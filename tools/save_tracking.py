# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import shutil
import pickle
import json
import torch.backends.cudnn as cudnn
import random
import copy
from functools import partial

import torchvision.transforms.functional as F

from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmaction.registry import RUNNERS
from mmengine.registry import DATASETS, MODELS, FUNCTIONS
from mmengine.structures import InstanceData

import imageio
import cv2
import subprocess
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMAction2 test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--checkpoint', type=str, default='work_dirs/alphachimp/alphachimp_res576.pth', help='checkpoint file path')
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='visualize per interval samples.')
    parser.add_argument("--is_distributed", type= int, default=1,help="Whether to produce samples from the model")
    parser.add_argument(
        '--dump',
        type=str,
        help='dump predictions to a pickle file for offline evaluation')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--output_dir',
        help='directory where the visualization images will be saved.', default='mmtracking/track_pkl/')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--no_obj', type=bool, default=False)
    parser.add_argument('--exp', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def merge_args(cfg, args):
    """Merge CLI arguments to config."""

    # -------------------- Dump predictions --------------------
    if args.dump is not None:
        assert args.dump.endswith(('.pkl', '.pickle')), \
            'The dump file must be a pkl file.'
        dump_metric = dict(type='DumpResults', out_file_path=args.dump)
        if isinstance(cfg.test_evaluator, (list, tuple)):
            cfg.test_evaluator = list(cfg.test_evaluator)
            cfg.test_evaluator.append(dump_metric)
        else:
            cfg.test_evaluator = [cfg.test_evaluator, dump_metric]

    return cfg

def run_vis(model, dataloader, out_file_path='out', is_distributed=True, no_obj=False):
    # Track instances to save detection results
    track_instances = defaultdict(dict)

    if is_distributed:
        print(f'RANK {os.environ["RANK"]}: Start Detection ...')

    # Iterate through data loader and process each batch
    for data_batch in tqdm(dataloader):
        with torch.no_grad():
            data = model.module.data_preprocessor(data_batch)
            input_h, input_w = data['inputs']['imgs'].size()[-2:]
            det_results = model.module.detector.predict(data['inputs']['imgs'], data['data_samples'])

            for det in det_results:
                video_id = det.metainfo['video_id']
                frame_id = det.metainfo['timestamp']
                det.metainfo['frame_id'] = frame_id
                det.pred_instances.labels = det.pred_instances.labels.detach().cpu()
                det.pred_instances.scores = det.pred_instances.scores.detach().cpu()
                if no_obj:
                    det.pred_instances.scores = torch.max(det.pred_instances.labels, dim=-1)[0]

                det.pred_instances.bboxes = det.pred_instances.bboxes.detach().cpu()
                track_instances[video_id][frame_id] = det

    if is_distributed:
        print(f'RANK {os.environ["RANK"]}: Complete Detection ...')

        torch.distributed.barrier()
        name = 'validate'
        save_path = osp.join(out_file_path, name)
        rank = torch.distributed.get_rank()
        path = save_path + '_' + str(rank) + '.pkl'
        with open(path, 'wb') as fid:
            pickle.dump(track_instances, fid, pickle.HIGHEST_PROTOCOL)
        print('dump the file', path)

        torch.distributed.barrier()
        if rank == 0:
            final_output = {}
            world_size = torch.distributed.get_world_size()
            for r in range(world_size):
                path = save_path + '_' + str(r) + '.pkl'
                with open(path, 'rb') as fid:
                    pred_i = pickle.load(fid)
                    print('load the file', path)
                for key in pred_i.keys():
                    if key not in final_output:
                        final_output[key] = pred_i[key]
                    else:
                        final_output[key].update(pred_i[key])
                os.remove(path)

            draw_vis(model, final_output, out_file_path, input_h, input_w)

    torch.distributed.barrier()


def draw_vis(model, track_instances, out_file_path='out', input_h = 576, input_w = 576, data_path='data/ChimpACT_processed'):
    pos_thr = 0.1

    pickle_data = {'det_bboxes':[],'track_bboxes':[],'info':[],'bbox_bboxes':[]}
    video_ids = sorted(track_instances.keys())
    for video_id in tqdm(video_ids):
        frame_ids = sorted(track_instances[video_id].keys())
        for frame_id in tqdm(frame_ids):
            data_sample = copy.deepcopy(track_instances[video_id][frame_id])
            image_path = os.path.join(data_path, video_id, str(frame_id).zfill(6)+'.jpg')
            image = cv2.imread(image_path)
            img_h, img_w = image.shape[:2]
            scale_h, scale_w = img_h / input_h, img_w / input_w

            det_bbox = data_sample.pred_instances.bboxes.detach().clone().cpu().numpy()
            det_score = data_sample.pred_instances.scores.detach().clone().cpu().numpy()
            select_id = det_score > pos_thr
            det_bbox = det_bbox[select_id]
            det_score = det_score[select_id].reshape(-1,1)
            det_bbox[:,0] = det_bbox[:,0] * scale_w
            det_bbox[:,1] = det_bbox[:,1] * scale_h
            det_bbox[:,2] = det_bbox[:,2] * scale_w
            det_bbox[:,3] = det_bbox[:,3] * scale_h
            det_data = np.concatenate([det_bbox,det_score],axis=1)
            pickle_data['det_bboxes'].append([det_data])

            pred_track_instances = model.module.tracker.track(data_sample)
            track_id = pred_track_instances.instances_id.detach().cpu().numpy().reshape(-1,1)
            track_score = pred_track_instances.scores.detach().cpu().numpy().reshape(-1,1)
            track_bbox = pred_track_instances.bboxes.detach().cpu().numpy()
            track_bbox[:,0] = track_bbox[:,0] * scale_w
            track_bbox[:,1] = track_bbox[:,1] * scale_h
            track_bbox[:,2] = track_bbox[:,2] * scale_w
            track_bbox[:,3] = track_bbox[:,3] * scale_h
            track_data = np.concatenate([track_id, track_bbox, track_score],axis=1)
            pickle_data['track_bboxes'].append([track_data])
            pickle_data['info'].append({'video_id':video_id,'frame_id':frame_id,'img_w':img_w,'img_h':img_h})


    with open(f"{out_file_path}/summary.pkl", 'wb') as f:
        pickle.dump(pickle_data, f)
    print(f"Summary JSON saved")


def init_distributed(args):
    if "WORLD_SIZE" not in os.environ or int(os.environ["WORLD_SIZE"]) < 1:
        return False

    torch.cuda.set_device(args.local_rank)

    assert os.environ["MASTER_PORT"], "set the MASTER_PORT variable or use pytorch launcher"
    assert os.environ["RANK"], "use pytorch launcher and explicityly state the rank of the process"

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
    #cudnn.enabled = False
    #torch.set_deterministric(True)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")

    return True

def _init_fn(worker_id):
    np.random.seed(0)
    random.seed(0)


def build_dataloader(eval_dataset, cfg, args):
    collate_fn_cfg = cfg.val_dataloader.pop('collate_fn',
                                            dict(type='pseudo_collate'))
    if isinstance(collate_fn_cfg, dict):
        collate_fn_type = collate_fn_cfg.pop('type')
        if isinstance(collate_fn_type, str):
            collate_fn = FUNCTIONS.get(collate_fn_type)
        else:
            collate_fn = collate_fn_type
        collate_fn = partial(collate_fn, **collate_fn_cfg)  # type: ignore
    elif callable(collate_fn_cfg):
        collate_fn = collate_fn_cfg
    else:
        raise TypeError(
            'collate_fn should be a dict or callable object, but got '
            f'{collate_fn_cfg}')

    sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset) if args.is_distributed else None
    dataloader = DataLoader(eval_dataset,
                            batch_size=cfg.val_dataloader.batch_size,
                            shuffle=False,
                            num_workers=cfg.val_dataloader.num_workers,
                            sampler=sampler,
                            drop_last=False,
                            pin_memory=False,
                            collate_fn=collate_fn,
                            worker_init_fn=_init_fn)

    return dataloader


def main():
    args = parse_args()
    if args.exp > 0:
        args.output_dir += '_' + str(args.exp)


    # load config
    cfg = Config.fromfile(args.config)
    cfg = merge_args(cfg, args)
    cfg.launcher = args.launcher


    cfg.load_from = args.checkpoint

    rank = 0
    if args.is_distributed:
        args.is_distributed = init_distributed(args)
        master = torch.distributed.get_rank()
        if args.is_distributed and os.environ["RANK"]:
            master = int(os.environ["RANK"]) == 0
            rank, world_size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
        else:
            rank = world_size = None

        args.world_size = world_size
        if args.is_distributed:
            args.device = torch.device(args.local_rank)
        else:
            args.device = torch.device(0)


    dataset = DATASETS.build(cfg.val_dataloader.dataset)
    print(len(dataset))
    model = MODELS.build(cfg.model)

    if rank == 0:
        print('Loading Checkpoints ...')
    if not args.checkpoint == '':
        ckpt = torch.load(args.checkpoint, map_location='cpu')
    else:
        if args.exp > 0:
            cfg.model.detector.init_cfg.checkpoint = cfg.model.detector.init_cfg.checkpoint.replace('exp_1', f'exp_{args.exp}')
            print(f'Detected Exp {args.exp} is loading')
        ckpt = torch.load(cfg.model.detector.init_cfg.checkpoint, map_location='cpu')
    model.detector.load_state_dict(ckpt['state_dict'], strict=True)

    model.eval()
    model.detector.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.is_distributed:
        print('Launching DDP ...')
        model = model.to(args.device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.device], output_device=args.local_rank)

        data_loader = build_dataloader(dataset, cfg, args)
        run_vis(model, data_loader, out_file_path=args.output_dir, no_obj=args.no_obj)
    else:
        data_loader = build_dataloader(dataset, cfg, args)
        run_vis(model, data_loader, out_file_path=args.output_dir, no_obj=args.no_obj)


if __name__ == '__main__':
    main()

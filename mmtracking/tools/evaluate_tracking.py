# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import time
import pickle as pkl

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmdet.apis import set_random_seed

from mmtrack.core import setup_multi_processes
from mmtrack.datasets import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='mmtrack test model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument('--eval', type=str, nargs='+', help='eval types', default=['bbox', 'track'])
    parser.add_argument('--exp', type=int, default=0)
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='pytorch',
        help='job launcher')
    parser.add_argument(
        '--pickle_path',
        type=str,
        help='pickle file path', default='track_pkl/summary.pkl')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()
    if args.exp > 0:
        idx = args.pickle_path.find('nq')
        args.pickle_path = args.pickle_path[:idx+5] + f'_{args.exp}' + args.pickle_path[idx+5:]

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if cfg.get('USE_MMDET', False):
        from mmdet.apis import multi_gpu_test, single_gpu_test
        from mmdet.datasets import build_dataloader
        from mmdet.models import build_detector as build_model
        if 'detector' in cfg.model:
            cfg.model = cfg.model.detector
    elif cfg.get('TRAIN_REID', False):
        from mmdet.apis import multi_gpu_test, single_gpu_test
        from mmdet.datasets import build_dataloader

        from mmtrack.models import build_reid as build_model
        if 'reid' in cfg.model:
            cfg.model = cfg.model.reid
    else:
        from mmtrack.apis import multi_gpu_test, single_gpu_test
        from mmtrack.datasets import build_dataloader
        from mmtrack.models import build_model
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set random seeds. Force setting fixed seed and deterministic=True in SOT
    # configs
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if cfg.get('seed', None) is not None:
        set_random_seed(
            cfg.seed, deterministic=cfg.get('deterministic', False))
    cfg.data.test.test_mode = True

    cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    rank, _ = get_dist_info()
    # allows not to create
    if args.work_dir is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        json_file = osp.join(args.work_dir, f'eval_{timestamp}.log.json')

    # build the dataloader
    dataset = build_dataset(cfg.data.test)

    outputs = mmcv.load(args.pickle_path)
    #with open(args.pickle_path, 'rb') as fid:
    #    outputs = pkl.load(fid)
    if 'info' in outputs:
        outputs.pop('info')

    rank, _ = get_dist_info()
    if rank == 0:
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            eval_hook_args = [
                'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                'rule', 'by_epoch'
            ]
            for key in eval_hook_args:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            metric = dataset.evaluate(outputs, **eval_kwargs)
            print(metric)
            metric_dict = dict(
                config=args.config, mode='test', epoch=cfg.total_epochs)
            metric_dict.update(metric)
            if args.work_dir is not None:
                mmcv.dump(metric_dict, json_file)


if __name__ == '__main__':
    main()

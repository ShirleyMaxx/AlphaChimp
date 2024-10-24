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
from functools import partial

import torchvision.transforms.functional as F

from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmaction.registry import RUNNERS
from mmaction.registry import DATASETS, MODELS
from mmengine.registry import FUNCTIONS
from mmengine.structures import InstanceData

import imageio
import cv2
import subprocess
from collections import defaultdict
import os

class ColorArranger:
    def __init__(self, cmap='Set3', cmap_len=12, abandon_n=10):
        self.cmap = plt.get_cmap(cmap)
        self.cmap_len = cmap_len

        self.various_cmap = plt.get_cmap('tab20')
        self.various_cmap_len = 20

        self.multi_seed = 937325691048103
        self.add_seed = 802374093583177
        self.rand_max = 65993

        self.remain_colors = set(list(range(self.cmap_len)))
        self.recorded_ids = {}

        self.abandon_n = abandon_n

    def lcg(self, track_id, cmap_len):
        return int(((track_id * self.multi_seed + self.add_seed) % self.rand_max) / self.rand_max * cmap_len)

    def get_color(self, track_id):
        if track_id in self.recorded_ids:
            self.recorded_ids[track_id][2] = 0
            return self.recorded_ids[track_id][1]

        color = self.lcg(track_id, self.cmap_len)
        if color in self.remain_colors:
            self.remain_colors.remove(color)
            self.recorded_ids[track_id] = [color, [int(c * 255) for c in self.cmap(color)[:3]], 0]
            return [int(c * 255) for c in self.cmap(color)[:3]]
        else:
            color = self.lcg(track_id, self.various_cmap_len)
            self.recorded_ids[track_id] = [-1, [int(c * 255) for c in self.various_cmap(color)[:3]], 0]
            return [int(c * 255) for c in self.various_cmap(color)[:3]]
        pass

    def frame_update(self):
        keys_to_remove = []
        for key in self.recorded_ids.keys():
            self.recorded_ids[key][2] += 1
            if self.recorded_ids[key][2] >= self.abandon_n:
                if self.recorded_ids[key][0] != -1:
                    self.remain_colors.add(self.recorded_ids[key][0])
                keys_to_remove.append(key)
        for key in keys_to_remove:
            self.recorded_ids.pop(key)
        pass


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMAction2 test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--checkpoint', type=str, default='work_dirs/alphachimp/alphachimp_res576.pth', help='checkpoint file path')
    parser.add_argument(
        '--gpus',
        type=int,
        default=4,
        help='visualize per interval samples.')
    parser.add_argument("--is_distributed", type= int, default=1,help="Whether to produce samples from the model")
    parser.add_argument("--temp_dir",type=str, default='infer_temps', help="temporary directory")
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
        default='infer_output',
        help='directory where the visualization images will be saved.')
    parser.add_argument(
        '--input_dir',
        default='infer_input',
        type=str,
        help='path of video.')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--vis_mode', type=str, default='mix', help="choose between 'det', 'act' and 'mix', which means to visualize detection bbox / action / both")
    parser.add_argument('--test_mode', type=bool, default=False, help="debug")
    parser.add_argument('--no_obj_mode', type=bool, default=True)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def merge_args(cfg, args):
    """Merge CLI arguments to config."""

    # -------------------- Dump predictions --------------------
    if args.dump is not None:
        assert argcs.dump.endswith(('.pkl', '.pickle')), \
            'The dump file must be a pkl file.'
        dump_metric = dict(type='DumpResults', out_file_path=args.dump)
        if isinstance(cfg.test_evaluator, (list, tuple)):
            cfg.test_evaluator = list(cfg.test_evaluator)
            cfg.test_evaluator.append(dump_metric)
        else:
            cfg.test_evaluator = [cfg.test_evaluator, dump_metric]

    return cfg

def merge_dicts(dict1, dict2):
    for key in dict2:
        if key in dict1:
            if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                merge_dicts(dict1[key], dict2[key])
            else:
                print('conflict key {}'.format(key))
                dict1[key] = dict2[key]
        else:
            dict1[key] = dict2[key]
    return dict1

def run_vis(model, dataloader, data_path, out_file_path='out', fps=25, is_distributed=True, vis_mode='det', no_obj_mode=True):
    input_h, input_w = None, None
    # Track instances to save detection results
    track_instances = defaultdict(dict)
    pos_thr = 0.30

    if is_distributed:
        print(f'RANK {os.environ["RANK"]}: Start Detection ...')

    # Iterate through data loader and process each batch
    for data_batch in tqdm(dataloader):
        with torch.no_grad():
            data = model.module.data_preprocessor(data_batch, training=False)
            if input_h is None or input_w is None:
                input_h, input_w = data['inputs']['imgs'].size()[-2:]

            det_results = model.module.detector.predict(data['inputs']['imgs'], data['data_samples'])
            for det in det_results:
                video_id = det.metainfo['video_id']
                frame_id = det.metainfo['timestamp']
                det.metainfo['frame_id'] = frame_id
                det.pred_instances.labels = det.pred_instances.labels.detach().cpu()
                det.pred_instances.scores = det.pred_instances.scores.detach().cpu()
                if no_obj_mode:
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
        # print('dump the file', path)

        torch.distributed.barrier()
        if rank == 0:
            final_output = {}
            world_size = torch.distributed.get_world_size()
            for r in range(world_size):
                path = save_path + '_' + str(r) + '.pkl'
                with open(path, 'rb') as fid:
                    pred_i = pickle.load(fid)
                    # print('load the file', path)
                # print(f"before {final_output.keys()}")
                for key in pred_i.keys():
                    if key not in final_output:
                        final_output[key] = pred_i[key]
                    else:
                        final_output[key].update(pred_i[key])
                        # print(final_output[key].keys())
                # print(f"after {final_output.keys()}")
                os.remove(path)

            if vis_mode == 'det':
                draw_vis_det(model, final_output, data_path, out_file_path, fps, input_h, input_w)
            elif vis_mode == 'mix':
                draw_vis_mix(model, final_output, data_path, out_file_path, fps, input_h, input_w)
            elif vis_mode == 'act':
                draw_vis_act(model, final_output, data_path, out_file_path, fps, input_h, input_w)
            else:
                draw_vis_det(model, final_output, data_path, out_file_path, fps, input_h, input_w)
                draw_vis_act(model, final_output, data_path, out_file_path, fps, input_h, input_w)
                draw_vis_mix(model, final_output, data_path, out_file_path, fps, input_h, input_w)

    torch.distributed.barrier()


def draw_vis_det(model, track_instances, data_path, out_file_path='out', fps=25, input_h=576, input_w=576):
    pos_thr = 0.25
    act_thr = 0.30

    action_class_names = ['other', 'moving', 'climbing', 'resting', 'sleeping',
                          'solitary object playing', 'eating', 'manipulating object',
                          'grooming', 'being groomed', 'aggressing', 'embracing', 'begging',
                          'being begged from', 'taking object',
                          'losing object', 'carrying', 'being carried', 'nursing', 'being nursed',
                          'playing', 'touching', 'erection',
                          'displaying']

    # Prepare color map for distinct colors in tracking visualization

    # cmap = plt.get_cmap('Set3')
    action_cmap = plt.get_cmap('Set2')
    summary_json_data = []
    for video_id in tqdm(track_instances.keys()):
        video_writer = imageio.get_writer(f"{out_file_path}/{video_id}_det.mp4", fps=fps)
        json_data = []

        color_arranger = ColorArranger()
        frame_ids = sorted(track_instances[video_id].keys())
        for frame_id in tqdm(frame_ids):
            data_sample = track_instances[video_id][frame_id]
            pred_track_instances = model.module.tracker.track(data_sample)

            # Load the image
            image_path = os.path.join(data_path, video_id, str(frame_id).zfill(6) + '.jpg')
            image = cv2.imread(image_path)
            if image is None:
                continue
            img_h, img_w = image.shape[:2]
            scale_h, scale_w = img_h / input_h, img_w / input_w

            color_arranger.frame_update()
            for idx, (bbox, label, score, track_id) in enumerate(zip(pred_track_instances.bboxes,
                                                                     data_sample.pred_instances.labels,
                                                                     pred_track_instances.scores,
                                                                     pred_track_instances.instances_id)):
                bbox = bbox.detach().cpu().numpy()
                label = label.detach().cpu().numpy()
                score = score.detach().cpu().numpy()
                track_id = track_id.detach().cpu().numpy()
                label_indices = (label >= act_thr).astype(int)  # Determine which labels exceed threshold
                label_names = [action_class_names[i] for i, is_active in enumerate(label_indices) if is_active]
                label_name_str = ', '.join(label_names)

                color = color_arranger.get_color(float(track_id))

                # Scale bounding box coordinates
                bbox_x1 = int(bbox[0] * scale_w)
                bbox_y1 = int(bbox[1] * scale_h)
                bbox_x2 = int(bbox[2] * scale_w)
                bbox_y2 = int(bbox[3] * scale_h)

                text_det = f"{score:.2f}"
                text_act = ""
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                thickness = 2
                left_txt_pad = 8
                left_rec_pad = 15
                bottom_txt_pad = 8
                bottom_rec_pad = 1
                top_txt_pad = 8
                right_txt_pad = 8
                txt_interval = 0
                bbox_thickness = 3

                position = (bbox_x1, bbox_y1)
                text_color = (0, 0, 0)

                # Draw bounding box and label
                cv2.rectangle(image, (bbox_x1, bbox_y1), (bbox_x2, bbox_y2), color[::-1], thickness=bbox_thickness)

                text_size_det = cv2.getTextSize(text_det, font, font_scale, thickness)[0]
                text_size_act = (0, 0)
                text_size = (max(text_size_det[0], text_size_act[0]), text_size_det[1] + text_size_act[1])

                top_left = (position[0] - left_rec_pad,
                            position[1] - bottom_rec_pad - bottom_txt_pad - text_size[1] - txt_interval - top_txt_pad)
                bottom_right = (
                position[0] - left_rec_pad + left_txt_pad + text_size[0] + right_txt_pad, position[1] - bottom_rec_pad)

                # Draw the rectangle
                cv2.rectangle(image, top_left, bottom_right, color[::-1], cv2.FILLED)
                cv2.putText(image, text_det, (
                top_left[0] + left_txt_pad, bottom_right[1] - bottom_txt_pad - txt_interval - text_size_act[1]), font,
                            font_scale, text_color, thickness)

                text_y = bottom_right[1] - bottom_txt_pad
                text_x = top_left[0] + left_txt_pad

                # Append detection data to json
                json_data.append({
                    'frame_id': frame_id,
                    'bbox': [bbox_x1, bbox_y1, bbox_x2, bbox_y2],
                    'label': label_names,
                    'score': float(score),
                    'track_id': int(track_id)
                })
                summary_json_data.append({
                    'video_id': video_id,
                    'frame_id': frame_id,
                    'bbox': [bbox_x1, bbox_y1, bbox_x2, bbox_y2],
                    'label': label_names,
                    'score': float(score),
                    'track_id': int(track_id)
                })
            # Save processed frame to video
            video_writer.append_data(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Close the video writer for the current video
        video_writer.close()

        # Save JSON data for the current video
        with open(f"{out_file_path}/{video_id}.json", 'w') as f:
            json.dump(json_data, f)
        print(f"Processed and saved video and JSON for {video_id}")

    with open(f"{out_file_path}/summary.json", 'w') as f:
        json.dump(summary_json_data, f)
    print(f"Summary JSON saved")

def draw_vis_act(model, track_instances, data_path, out_file_path='out', fps=25, input_h=576, input_w=576):
    pos_thr = 0.25
    act_thr = 0.30

    action_class_names = ['other', 'moving', 'climbing', 'resting', 'sleeping',
                          'solitary object playing', 'eating', 'manipulating object',
                          'grooming', 'being groomed', 'aggressing', 'embracing', 'begging',
                          'being begged from', 'taking object',
                          'losing object', 'carrying', 'being carried', 'nursing', 'being nursed',
                          'playing', 'touching', 'erection',
                          'displaying']

    # Prepare color map for distinct colors in tracking visualization

    # cmap = plt.get_cmap('Set3')
    action_cmap = plt.get_cmap('Set2')
    summary_json_data = []
    for video_id in tqdm(track_instances.keys()):
        video_writer = imageio.get_writer(f"{out_file_path}/{video_id}_act.mp4", fps=fps)
        json_data = []

        color_arranger = ColorArranger()
        frame_ids = sorted(track_instances[video_id].keys())
        for frame_id in tqdm(frame_ids):
            data_sample = track_instances[video_id][frame_id]
            pred_track_instances = model.module.tracker.track(data_sample)

            # Load the image
            image_path = os.path.join(data_path, video_id, str(frame_id).zfill(6) + '.jpg')
            image = cv2.imread(image_path)
            if image is None:
                continue
            img_h, img_w = image.shape[:2]
            scale_h, scale_w = img_h / input_h, img_w / input_w

            color_arranger.frame_update()
            for idx, (bbox, label, score, track_id) in enumerate(zip(pred_track_instances.bboxes,
                                                                     data_sample.pred_instances.labels,
                                                                     pred_track_instances.scores,
                                                                     pred_track_instances.instances_id)):
                bbox = bbox.detach().cpu().numpy()
                label = label.detach().cpu().numpy()
                score = score.detach().cpu().numpy()
                track_id = track_id.detach().cpu().numpy()
                label_indices = (label >= act_thr).astype(int)  # Determine which labels exceed threshold
                label_names = [action_class_names[i] for i, is_active in enumerate(label_indices) if is_active]
                label_name_str = ', '.join(label_names)

                color = color_arranger.get_color(float(track_id))

                # Scale bounding box coordinates
                bbox_x1 = int(bbox[0] * scale_w)
                bbox_y1 = int(bbox[1] * scale_h)
                bbox_x2 = int(bbox[2] * scale_w)
                bbox_y2 = int(bbox[3] * scale_h)

                text_det = f""
                text_act = ", ".join(label_names) + " "
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                thickness = 2
                left_txt_pad = 8
                left_rec_pad = 15
                bottom_txt_pad = 8
                bottom_rec_pad = 1
                top_txt_pad = 8
                right_txt_pad = 8
                txt_interval = 0
                bbox_thickness = 3

                position = (bbox_x1, bbox_y1)
                text_color = (0, 0, 0)

                # Draw bounding box and label
                cv2.rectangle(image, (bbox_x1, bbox_y1), (bbox_x2, bbox_y2), color[::-1], thickness=bbox_thickness)

                text_size_det = (0, 0)
                text_size_act = cv2.getTextSize(text_act, font, font_scale, thickness)[0]
                text_size = (max(text_size_det[0], text_size_act[0]), text_size_det[1] + text_size_act[1])

                top_left = (position[0] - left_rec_pad,
                            position[1] - bottom_rec_pad - bottom_txt_pad - text_size[1] - txt_interval - top_txt_pad)
                bottom_right = (
                position[0] - left_rec_pad + left_txt_pad + text_size[0] + right_txt_pad, position[1] - bottom_rec_pad)

                # Draw the rectangle
                cv2.rectangle(image, top_left, bottom_right, color[::-1], cv2.FILLED)

                text_y = bottom_right[1] - bottom_txt_pad
                text_x = top_left[0] + left_txt_pad
                for label_ids, label_name in enumerate(label_names):
                    action_color = [32, 32, 32]
                    # Calculate the size of the text
                    text_size = cv2.getTextSize(f"{label_name}, ", font, font_scale, thickness)[0]
                    # Adjust text position based on its size
                    # Draw each action label with its corresponding color
                    if label_ids == len(label_names) - 1:
                        cv2.putText(image, f"{label_name} ", (text_x, text_y), font, font_scale, action_color[::-1],
                                    thickness)
                    else:
                        cv2.putText(image, f"{label_name}, ", (text_x, text_y), font, font_scale, action_color[::-1],
                                    thickness)
                    text_x += text_size[0]

                # Append detection data to json
                json_data.append({
                    'frame_id': frame_id,
                    'bbox': [bbox_x1, bbox_y1, bbox_x2, bbox_y2],
                    'label': label_names,
                    'score': float(score),
                    'track_id': int(track_id)
                })
                summary_json_data.append({
                    'video_id': video_id,
                    'frame_id': frame_id,
                    'bbox': [bbox_x1, bbox_y1, bbox_x2, bbox_y2],
                    'label': label_names,
                    'score': float(score),
                    'track_id': int(track_id)
                })
            # Save processed frame to video
            video_writer.append_data(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Close the video writer for the current video
        video_writer.close()

        # Save JSON data for the current video
        with open(f"{out_file_path}/{video_id}.json", 'w') as f:
            json.dump(json_data, f)
        print(f"Processed and saved video and JSON for {video_id}")

    with open(f"{out_file_path}/summary.json", 'w') as f:
        json.dump(summary_json_data, f)
    print(f"Summary JSON saved")

def draw_vis_mix(model, track_instances, data_path, out_file_path='out', fps=25, input_h = 576, input_w = 576):
    pos_thr = 0.25
    act_thr = 0.30

    action_class_names = ['other', 'moving', 'climbing', 'resting', 'sleeping',
                          'solitary object playing', 'eating', 'manipulating object',
                          'grooming', 'being groomed', 'aggressing', 'embracing', 'begging',
                          'being begged from', 'taking object',
                          'losing object', 'carrying', 'being carried', 'nursing', 'being nursed',
                          'playing', 'touching', 'erection',
                          'displaying']

    # Prepare color map for distinct colors in tracking visualization

    #cmap = plt.get_cmap('Set3')
    action_cmap = plt.get_cmap('Set2')
    summary_json_data = []
    for video_id in tqdm(track_instances.keys()):
        video_writer = imageio.get_writer(f"{out_file_path}/{video_id}_mix.mp4", fps=fps)
        json_data = []

        color_arranger = ColorArranger()
        frame_ids = sorted(track_instances[video_id].keys())
        for frame_id in tqdm(frame_ids):
            data_sample = track_instances[video_id][frame_id]
            pred_track_instances = model.module.tracker.track(data_sample)

            # Load the image
            image_path = os.path.join(data_path, video_id, str(frame_id).zfill(6)+'.jpg')
            image = cv2.imread(image_path)
            if image is None:
                continue
            img_h, img_w = image.shape[:2]
            scale_h, scale_w = img_h / input_h, img_w / input_w

            color_arranger.frame_update()
            for idx, (bbox, label, score, track_id) in enumerate(zip(pred_track_instances.bboxes,
                                                                     data_sample.pred_instances.labels,
                                                                     pred_track_instances.scores,
                                                                     pred_track_instances.instances_id)):
                bbox = bbox.detach().cpu().numpy()
                label = label.detach().cpu().numpy()
                score = score.detach().cpu().numpy()
                track_id = track_id.detach().cpu().numpy()
                label_indices = (label >= act_thr).astype(int)  # Determine which labels exceed threshold
                label_names = [action_class_names[i] for i, is_active in enumerate(label_indices) if is_active]
                label_name_str = ', '.join(label_names)

                color = color_arranger.get_color(float(track_id))

                # Scale bounding box coordinates
                bbox_x1 = int(bbox[0] * scale_w)
                bbox_y1 = int(bbox[1] * scale_h)
                bbox_x2 = int(bbox[2] * scale_w)
                bbox_y2 = int(bbox[3] * scale_h)


                text_det = f"{score:.2f}"
                text_act = ", ".join(label_names) + " "
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                thickness = 2
                left_txt_pad = 8
                left_rec_pad = 15
                bottom_txt_pad = 8
                bottom_rec_pad = 1
                top_txt_pad = 8
                right_txt_pad = 8
                txt_interval = 8
                bbox_thickness = 3

                position = (bbox_x1, bbox_y1)
                text_color = (0, 0, 0)

                # Draw bounding box and label
                cv2.rectangle(image, (bbox_x1, bbox_y1), (bbox_x2, bbox_y2), color[::-1], thickness=bbox_thickness)

                text_size_det = cv2.getTextSize(text_det, font, font_scale, thickness)[0]
                text_size_act = cv2.getTextSize(text_act, font, font_scale, thickness)[0]
                text_size = (max(text_size_det[0], text_size_act[0]), text_size_det[1] + text_size_act[1])

                top_left = (position[0] - left_rec_pad, position[1] - bottom_rec_pad - bottom_txt_pad - text_size[1] - txt_interval - top_txt_pad)
                bottom_right = (position[0] - left_rec_pad + left_txt_pad + text_size[0] + right_txt_pad, position[1] - bottom_rec_pad)

                # Draw the rectangle
                cv2.rectangle(image, top_left, bottom_right, color[::-1], cv2.FILLED)
                cv2.putText(image, text_det, (top_left[0] + left_txt_pad, bottom_right[1] - bottom_txt_pad - txt_interval - text_size_act[1]), font, font_scale, text_color, thickness)

                text_y = bottom_right[1] - bottom_txt_pad
                text_x = top_left[0] + left_txt_pad
                for label_ids, label_name in enumerate(label_names):
                    action_color = [32, 32, 32]
                    # Calculate the size of the text
                    text_size = cv2.getTextSize(f"{label_name}, ", font, font_scale, thickness)[0]
                    # Adjust text position based on its size
                    # Draw each action label with its corresponding color
                    if label_ids == len(label_names)-1:
                        cv2.putText(image, f"{label_name} ", (text_x, text_y), font, font_scale, action_color[::-1], thickness)
                    else:
                        cv2.putText(image, f"{label_name}, ", (text_x, text_y), font, font_scale, action_color[::-1], thickness)
                    text_x += text_size[0]

                # Append detection data to json
                json_data.append({
                    'frame_id': frame_id,
                    'bbox': [bbox_x1, bbox_y1, bbox_x2, bbox_y2],
                    'label': label_names,
                    'score': float(score),
                    'track_id': int(track_id)
                })
                summary_json_data.append({
                    'video_id': video_id,
                    'frame_id': frame_id,
                    'bbox': [bbox_x1, bbox_y1, bbox_x2, bbox_y2],
                    'label': label_names,
                    'score': float(score),
                    'track_id': int(track_id)
                })
            # Save processed frame to video
            video_writer.append_data(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
        # Close the video writer for the current video
        video_writer.close()

        # Save JSON data for the current video
        with open(f"{out_file_path}/{video_id}.json", 'w') as f:
            json.dump(json_data, f)
        print(f"Processed and saved video and JSON for {video_id}")

    with open(f"{out_file_path}/summary.json", 'w') as f:
        json.dump(summary_json_data, f)
    print(f"Summary JSON saved")


def video2frames(vid_path, save_dir=None, do_ffmpeg=True):
    cap = cv2.VideoCapture(vid_path)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    vid_name = vid_path.split('/')[-1][:-4]
    if not do_ffmpeg:
        return frames, vid_name

    img_dir = osp.join(save_dir, 'images', vid_name)
    os.makedirs(img_dir, exist_ok=True)
    command = ['ffmpeg',
               '-i', vid_path,
               '-r', str(fps),
               '-f', 'image2',
               '-v', 'error',
               '-start_number', '0',
               f'{img_dir}/%06d.jpg']

    print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)
    print(f'Images saved to \"{img_dir}\"')
    return frames, vid_name

def process_data_per_video(video_dir, args, rank):
    video_list = sorted(os.listdir(video_dir))
    if args.test_mode:
        video_list = video_list[0:1]

    video_list_dist = []
    if args.is_distributed:
        for i in range(len(video_list)):
            if i % args.world_size == rank:
                video_list_dist.append(video_list[i])

    infer_data = []
    infer_path = []
    for video_name in video_list:
        vid_path = osp.join(video_dir, video_name)
        infer_data_video = []
        if video_name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.MP4')):
            frames, img_dir = video2frames(vid_path, video_dir, video_name in video_list_dist)
            infer_data_video.append({'video_name': img_dir, 'frame_len': frames})
            infer_path_video = osp.join(video_dir, f'{video_name}_infer.json')
            with open(infer_path_video, 'w') as f:
                json.dump(infer_data_video, f)
            infer_data.append(infer_data_video)
            infer_path.append(infer_path_video)
        else:
            print(f"Skipping non-video file: {video_name}")
            continue
    return infer_data, infer_path

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

    destination_dir = args.temp_dir
    if rank == 0:
        print('Deleting Exisiting Temp Folder ...')
        if os.path.exists(destination_dir):
            exist_file = set(os.listdir(destination_dir))
            tgt_file = set(os.listdir(args.input_dir))
            files_to_delete = exist_file - tgt_file
            files_to_add = tgt_file - exist_file
            for file in files_to_delete:
                if '.' in file:
                    os.remove(os.path.join(destination_dir, file))
                else:
                    shutil.rmtree(os.path.join(destination_dir, file))
            for file in files_to_add:
                shutil.copy(os.path.join(args.input_dir, file), destination_dir)
        else:
            os.makedirs(destination_dir, exist_ok=True)
            shutil.copytree(args.input_dir, destination_dir, dirs_exist_ok=True)
    args.input_dir = destination_dir

    if args.is_distributed:
        torch.distributed.barrier()

    # Process Data
    if rank == 0:
        print('Processing Data ...')
    infer_data, infer_path = process_data_per_video(args.input_dir, args, rank)

    if args.is_distributed:
        torch.distributed.barrier()

    cfg.val_dataloader.dataset['data_prefix'] = dict(img=osp.join(args.input_dir,'images'))
    data_loader_list = []
    for path in infer_path:
        cfg.val_dataloader.dataset['ann_file'] = path
        data_loader_list.append(DATASETS.build(cfg.val_dataloader.dataset))

    # start testing
    # runner.test()
    if rank == 0:
        print('Building Model ...')
    model = MODELS.build(cfg.model)

    if rank == 0:
        print('Loading Checkpoints ...')
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model.detector.load_state_dict(ckpt['state_dict'], strict=True)

    model.eval()
    if rank == 0:
        if os.path.exists(args.output_dir):
            shutil.rmtree(args.output_dir)
        os.makedirs(args.output_dir, exist_ok=True)

    if args.is_distributed:
        torch.distributed.barrier()

    if args.is_distributed:
        print('Launching DDP ...')
        model = model.to(args.device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.device], output_device=args.local_rank)

        for dataset in data_loader_list:
            data_loader = build_dataloader(dataset, cfg, args)
            run_vis(model, data_loader, out_file_path=args.output_dir, data_path=osp.join(args.temp_dir, 'images'), vis_mode=args.vis_mode, no_obj_mode=args.no_obj_mode)
    else:
        for dataset in data_loader_list:
            data_loader = build_dataloader(dataset, cfg, args)
            run_vis(model, data_loader, out_file_path=args.output_dir, data_path=osp.join(args.temp_dir, 'images'), vis_mode=args.vis_mode, no_obj_mode=args.no_obj_mode)


if __name__ == '__main__':
    main()

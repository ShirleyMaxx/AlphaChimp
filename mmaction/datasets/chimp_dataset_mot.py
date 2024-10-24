# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from collections import defaultdict
from typing import Callable, List, Optional, Union

import numpy as np
import json
from mmengine.fileio import exists, list_from_file, load
from mmengine.logging import MMLogger

from mmaction.evaluation import read_labelmap
from mmaction.registry import DATASETS
from mmaction.utils import ConfigType
from mmaction.datasets.base import BaseActionDataset

@DATASETS.register_module()
class ChimpDataset(BaseActionDataset):
    """LeipzigChimp dataset for spatial temporal detection.

    Based on official AVA annotation files, the dataset loads raw frames,
    bounding boxes, proposals and applies specified transformations to return
    a dict containing the frame tensors and other information.

    This datasets can load information from the following files:

    .. code-block:: txt

        ann_file -> ava_{train, val}_{v2.1, v2.2}.csv
        exclude_file -> ava_{train, val}_excluded_timestamps_{v2.1, v2.2}.csv
        label_file -> ava_action_list_{v2.1, v2.2}.pbtxt /
                      ava_action_list_{v2.1, v2.2}_for_activitynet_2019.pbtxt
        proposal_file -> ava_dense_proposals_{train, val}.FAIR.recall_93.9.pkl

    Particularly, the proposal_file is a pickle file which contains
    ``img_key`` (in format of ``{video_id},{timestamp}``). Example of a pickle
    file:

    .. code-block:: JSON

        {
            ...
            '0f39OWEqJ24,0902':
                array([[0.011   , 0.157   , 0.655   , 0.983   , 0.998163]]),
            '0f39OWEqJ24,0912':
                array([[0.054   , 0.088   , 0.91    , 0.998   , 0.068273],
                       [0.016   , 0.161   , 0.519   , 0.974   , 0.984025],
                       [0.493   , 0.283   , 0.981   , 0.984   , 0.983621]]),
            ...
        }

    Args:
        ann_file (str): Path to the annotation file like
            ``ava_{train, val}_{v2.1, v2.2}.csv``.
        exclude_file (str): Path to the excluded timestamp file like
            ``ava_{train, val}_excluded_timestamps_{v2.1, v2.2}.csv``.
        pipeline (List[Union[dict, ConfigDict, Callable]]): A sequence of
            data transforms.
        label_file (str): Path to the label file like
            ``ava_action_list_{v2.1, v2.2}.pbtxt`` or
            ``ava_action_list_{v2.1, v2.2}_for_activitynet_2019.pbtxt``.
            Defaults to None.
        filename_tmpl (str): Template for each filename.
            Defaults to 'img_{:05}.jpg'.
        start_index (int): Specify a start index for frames in consideration of
            different filename format. However, when taking frames as input,
            it should be set to 0, since frames from 0. Defaults to 0.
        proposal_file (str): Path to the proposal file like
            ``ava_dense_proposals_{train, val}.FAIR.recall_93.9.pkl``.
            Defaults to None.
        person_det_score_thr (float): The threshold of person detection scores,
            bboxes with scores above the threshold will be used.
            Note that 0 <= person_det_score_thr <= 1. If no proposal has
            detection score larger than the threshold, the one with the largest
            detection score will be used. Default: 0.9.
        num_classes (int): The number of classes of the dataset. Default: 81.
            (AVA has 80 action classes, another 1-dim is added for potential
            usage)
        custom_classes (List[int], optional): A subset of class ids from origin
            dataset. Please note that 0 should NOT be selected, and
            ``num_classes`` should be equal to ``len(custom_classes) + 1``.
        data_prefix (dict or ConfigDict): Path to a directory where video
            frames are held. Defaults to ``dict(img='')``.
        test_mode (bool): Store True when building test or validation dataset.
            Defaults to False.
        modality (str): Modality of data. Support ``RGB``, ``Flow``.
            Defaults to ``RGB``.
        num_max_proposals (int): Max proposals number to store.
            Defaults to 1000.
        timestamp_start (int): The start point of included timestamps. The
            default value is referred from the official website.
            Defaults to 902.
        timestamp_end (int): The end point of included timestamps. The default
            value is referred from the official website. Defaults to 1798.
        fps (int): Overrides the default FPS for the dataset. Defaults to 30.
    """

    def __init__(self,
                 ann_file: str,
                 exclude_file: str,
                 pipeline: List[Union[ConfigType, Callable]],
                 label_file: str,
                 filename_tmpl: str = '{:06}.jpg',
                 start_index: int = 0,
                 proposal_file: str = None,
                 person_det_score_thr: float = 0.9,
                 num_classes: int = 24,
                 custom_classes: Optional[List[int]] = None,
                 data_prefix: ConfigType = dict(img=''),
                 modality: str = 'RGB',
                 test_mode: bool = False,
                 num_max_proposals: int = 1000,
                 timestamp_start: int = 900,
                 timestamp_end: int = 1800,
                 num_max_person: int = 24,
                 fps: int = 25,
                 num_joints: int = 16,
                 **kwargs) -> None:
        fps = 1
        self._FPS = fps  # Keep this as standard
        self.custom_classes = custom_classes
        if custom_classes is not None:
            assert num_classes == len(custom_classes) + 1
            assert 0 not in custom_classes
            _, class_whitelist = read_labelmap(open(label_file))
            assert set(custom_classes).issubset(class_whitelist)

            self.custom_classes = list([0] + custom_classes)
        self.exclude_file = exclude_file
        self.label_file = label_file
        self.proposal_file = proposal_file
        assert 0 <= person_det_score_thr <= 1, (
            'The value of '
            'person_det_score_thr should in [0, 1]. ')
        self.person_det_score_thr = person_det_score_thr
        self.timestamp_start = timestamp_start
        self.timestamp_end = timestamp_end
        self.num_max_proposals = num_max_proposals
        self.num_max_person = num_max_person
        self.filename_tmpl = filename_tmpl
        self.num_joints = num_joints

        self.test_mode_debug = test_mode

        super().__init__(
            ann_file,
            pipeline=pipeline,
            data_prefix=data_prefix,
            test_mode=test_mode,
            num_classes=num_classes,
            start_index=start_index,
            modality=modality,
            **kwargs)

        if self.proposal_file is not None:
            self.proposals = load(self.proposal_file)
        else:
            self.proposals = None

    def parse_img_record(self, img_records: List[dict]) -> tuple:
        """Merge image records of the same entity at the same time.

        Args:
            img_records (List[dict]): List of img_records (lines in AVA
                annotations).

        Returns:
            Tuple(list): A tuple consists of lists of bboxes, action labels and
                entity_ids.
        """
        bboxes, labels, entity_ids, keypoints, keypoints_visible = [], [], [], [], []
        while len(img_records) > 0:
            img_record = img_records[0]
            num_img_records = len(img_records)

            selected_records = [
                x for x in img_records
                if np.array_equal(x['entity_box'], img_record['entity_box'])
            ]

            num_selected_records = len(selected_records)
            img_records = [
                x for x in img_records if
                not np.array_equal(x['entity_box'], img_record['entity_box'])
            ]

            assert len(img_records) + num_selected_records == num_img_records

            bboxes.append(img_record['entity_box'])
            valid_labels = []
            for selected_record in selected_records:
                valid_labels += selected_record['label']
            valid_labels = np.array(valid_labels).astype(np.int32)

            # The format can be directly used by BCELossWithLogits
            label = np.zeros(self.num_classes, dtype=np.float32)
            label[valid_labels] = 1.

            labels.append(label)
            entity_ids.append(img_record['entity_id'])
            keypoints.append(img_record['keypoints'])
            keypoints_visible.append(img_record['keypoints_visible'])

        bboxes = np.stack(bboxes)
        labels = np.stack(labels)
        entity_ids = np.stack(entity_ids)
        keypoints = np.stack(keypoints)
        keypoints_visible = np.stack(keypoints_visible)
        return bboxes, labels, entity_ids, keypoints, keypoints_visible

    def load_data_list(self) -> List[dict]:
        """Load AVA annotations."""
        exists(self.ann_file)
        data_list = []
        records_dict_by_img = {}
        with open(self.ann_file,'r') as fid:
            annot = json.load(fid)
            
        num_samples = len(annot)
        for idx in range(num_samples):
            ann = annot[idx].copy()
            
            # note: different label process
            behaviors = []
            if self.custom_classes is not None:
                for label in ann['behaviors']:
                    if label not in self.custom_classes:
                        continue
                    label = self.custom_classes.index(label)
                    behaviors.append(label)
            else:
                behaviors = ann['behaviors']
            valid_labels = np.array(behaviors)
            if 'label_mask' in ann.keys():
                label_mask = np.array(ann['label_mask'])
            else:
                label_mask = np.ones(valid_labels.reshape(-1,self.num_classes).shape[0])
            
            video_id = ann['video_name']
            timestamp = int(ann['frame_id'])
            img_key = f'{video_id},{timestamp:04d}'
            self.timestamp_start, self.timestamp_end = (video_id.split('/')[-1][video_id.split('/')[-1].find('clip')+5:]).split('_')
            self.timestamp_start, self.timestamp_end = int(self.timestamp_start), int(self.timestamp_end)
            self.timestamp_start, self.timestamp_end = 0, self.timestamp_end-self.timestamp_start-1

            entity_box = np.array(ann['bbox'])
            entity_id = np.array(ann['entity_id'])
            shot_info = (self.timestamp_start, self.timestamp_end)
            video_info = dict(
                video_id=video_id,
                timestamp=timestamp,
                entity_box=entity_box,
                label=valid_labels,
                label_mask = label_mask,
                entity_id=entity_id,
                shot_info=shot_info)
            records_dict_by_img[img_key] = video_info
            video_id, timestamp = img_key.split(',')
            ann_img = records_dict_by_img[img_key]
            ann = dict(
                gt_bboxes= ann_img['entity_box'], gt_labels=ann_img['label'],label_mask = ann_img['label_mask'],entity_ids=ann_img['entity_id'])
            frame_dir = video_id
            if self.data_prefix['img'] is not None:
                frame_dir = osp.join(self.data_prefix['img'], frame_dir)
            timestamp_start, timestamp_end = (frame_dir.split('/')[-1][video_id.split('/')[-1].find('clip')+5:]).split('_')
            timestamp_start, timestamp_end = int(timestamp_start), int(timestamp_end)
            timestamp_start, timestamp_end = 0, timestamp_end-timestamp_start-1
            video_info = dict(
                frame_dir=frame_dir,
                video_id=video_id,
                timestamp=int(timestamp),
                img_key=img_key,
                shot_info=(timestamp_start, timestamp_end),
                fps=self._FPS,
                ann=ann)
            data_list.append(video_info)

        return data_list

    def filter_data(self) -> List[dict]:
        """Filter out records in the exclude_file."""
        valid_indexes = []
        if self.exclude_file is None:
            valid_indexes = list(range(len(self.data_list)))
        else:
            exclude_video_infos = [
                x.strip().split(',') for x in open(self.exclude_file)
            ]
            for i, data_info in enumerate(self.data_list):
                valid_indexes.append(i)
                for video_id, timestamp in exclude_video_infos:
                    if (data_info['video_id'] == video_id
                            and data_info['timestamp'] == int(timestamp)):
                        valid_indexes.pop()
                        break

        logger = MMLogger.get_current_instance()
        logger.info(f'{len(valid_indexes)} out of {len(self.data_list)}'
                    f' frames are valid.')
        data_list = [self.data_list[i] for i in valid_indexes]

        return data_list

    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index."""
        data_info = super().get_data_info(idx)
        img_key = data_info['img_key']

        data_info['filename_tmpl'] = self.filename_tmpl

        video_id = data_info['video_id']
        timestamp_start, timestamp_end = (video_id.split('/')[-1][video_id.split('/')[-1].find('clip')+5:]).split('_')
        timestamp_start, timestamp_end = int(timestamp_start), int(timestamp_end)
        timestamp_start, timestamp_end = 0, timestamp_end-timestamp_start-1
        data_info['timestamp_start'] = timestamp_start
        data_info['timestamp_end'] = timestamp_end

        if self.proposals is not None:
            if img_key not in self.proposals:
                data_info['proposals'] = np.array([[0, 0, 1, 1]])
                data_info['scores'] = np.array([1])
            else:
                proposals = self.proposals[img_key]
                assert proposals.shape[-1] in [4, 5]
                if proposals.shape[-1] == 5:
                    thr = min(self.person_det_score_thr, max(proposals[:, 4]))
                    positive_inds = (proposals[:, 4] >= thr)
                    proposals = proposals[positive_inds]
                    proposals = proposals[:self.num_max_proposals]
                    data_info['proposals'] = proposals[:, :4]
                    data_info['scores'] = proposals[:, 4]
                else:
                    proposals = proposals[:self.num_max_proposals]
                    data_info['proposals'] = proposals

        ann = data_info.pop('ann')
        data_info['gt_bboxes'] = ann['gt_bboxes']
        data_info['gt_labels'] = ann['gt_labels']
        data_info['label'] = ann['gt_labels']
        data_info['entity_ids'] = ann['entity_ids']
        entity_mask = np.zeros(self.num_max_person).astype(bool)
        entity_mask[data_info['entity_ids']] = 1
        data_info['entity_mask'] = entity_mask
        data_info['label_mask'] = ann['label_mask']

        return data_info
# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import List, Union

from mmdet.registry import DATASETS
from .base_video_dataset import BaseVideoDataset


@DATASETS.register_module()
class MOTChallengeDatasetChimp(BaseVideoDataset):
    """Dataset for MOTChallenge.

    Args:
        visibility_thr (float, optional): The minimum visibility
            for the objects during training. Default to -1.
    """

    METAINFO = {
        'classes':
        ('chimpanzee')
    }

    def __init__(self, 
                visibility_thr: float = -1,
                fps: int = 25,
                filename_tmpl: str = '{:06}.jpg',
                num_classes: int = 24,
                modality: str = 'RGB',
                *args, **kwargs):
        self.visibility_thr = visibility_thr
        fps = 1
        self._FPS = fps
        self.filename_tmpl = filename_tmpl
        self.modality = modality
        self.num_classes = num_classes
        super().__init__(*args, **kwargs)

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format. The difference between this
        function and the one in ``BaseVideoDataset`` is that the parsing here
        adds ``visibility`` and ``mot_conf``.

        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``

        Returns:
            Union[dict, List[dict]]: Parsed annotation.
        """
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']
        data_info = {}

        data_info.update(img_info)
        if self.data_prefix.get('img_path', None) is not None:
            img_path = osp.join(self.data_prefix['img_path'],
                                img_info['file_name'])
        else:
            img_path = img_info['file_name']
        data_info['img_path'] = img_path

        data_info['timestamp'] = img_info['frame_id']
        data_info['fps'] = self._FPS
        video_id = data_info['file_name']
        timestamp_start, timestamp_end = (video_id.split('/')[0][video_id.split('/')[0].find('clip')+5:]).split('_')
        timestamp_start, timestamp_end = int(timestamp_start), int(timestamp_end)
        timestamp_start, timestamp_end = 0, timestamp_end-timestamp_start-1
        shot_info = (timestamp_start, timestamp_end)
        data_info['modality'] = self.modality
        data_info['timestamp_start'] = timestamp_start
        data_info['shot_info'] = shot_info
        data_info['filename_tmpl'] = self.filename_tmpl
        data_info['frame_dir'] = img_path[:img_path.rfind('/')]

        instances = []
        for i, ann in enumerate(ann_info):
            instance = {}

            if (not self.test_mode) and (ann['visibility'] <
                                         self.visibility_thr):
                continue
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]

            if ann.get('iscrowd', False):
                instance['ignore_flag'] = 1
            else:
                instance['ignore_flag'] = 0
            instance['bbox'] = bbox
            instance['bbox_label'] = self.cat2label[ann['category_id']]
            instance['instance_id'] = ann['instance_id']
            instance['category_id'] = ann['category_id']
            instance['mot_conf'] = ann['mot_conf']
            instance['visibility'] = ann['visibility']
            if len(instance) > 0:
                instances.append(instance)
        if not self.test_mode:
            assert len(instances) > 0, f'No valid instances found in ' \
                f'image {data_info["img_path"]}!'
        data_info['instances'] = instances
        return data_info

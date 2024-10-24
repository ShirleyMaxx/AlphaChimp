# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
from time import time
from datetime import datetime
from typing import Any, List, Optional, Sequence, Tuple
from collections import OrderedDict

import numpy
import json
import pickle
import numpy as np
import traceback
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from mmengine.evaluator import BaseMetric
from copy import deepcopy

from mmaction.evaluation import ava_eval, results2csv
from mmaction.registry import METRICS
from mmaction.structures import bbox2result


class COCOResults(object):
    METRICS = {
        "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        "box_proposal": [
            "AR@100",
            "ARs@100",
            "ARm@100",
            "ARl@100",
            "AR@1000",
            "ARs@1000",
            "ARm@1000",
            "ARl@1000",
        ],
        "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
    }

    def __init__(self, *iou_types):
        allowed_types = ("box_proposal", "bbox", "segm", "keypoints")
        assert all(iou_type in allowed_types for iou_type in iou_types)
        results = OrderedDict()
        for iou_type in iou_types:
            results[iou_type] = OrderedDict(
                [(metric, -1) for metric in COCOResults.METRICS[iou_type]]
            )
        self.results = results

    def update(self, coco_eval):
        if coco_eval is None:
            return
        from pycocotools.cocoeval import COCOeval

        assert isinstance(coco_eval, COCOeval)
        s = coco_eval.stats
        iou_type = coco_eval.params.iouType
        res = self.results[iou_type]
        metrics = COCOResults.METRICS[iou_type]
        for idx, metric in enumerate(metrics):
            res[metric] = s[idx]

    def __repr__(self):
        results = '\n'
        for task, metrics in self.results.items():
            results += 'Task: {}\n'.format(task)
            metric_names = metrics.keys()
            metric_vals = ['{:.4f}'.format(v) for v in metrics.values()]
            results += (', '.join(metric_names) + '\n')
            results += (', '.join(metric_vals) + '\n')
        return results

def xyxy2xywh(bbox: list):
    return [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]

def dump_to_coco(coco_gts:list, coco_preds: list, act_thr: float = 0.2, action_names=None, model_type='dino', save_suffix=''):
    if action_names is None:
        action_names = []
    coco_gt_dict = dict(images=[], annotations=[], categories=[])
    coco_pred_dict = []

    coco_gt_dict['categories'].append(dict(
        id=1,
        name='none',
        supercategory='none',
    ))
    for i in range(len(action_names)):
        coco_gt_dict['categories'].append(dict(
            id=i+2,
            name=action_names[i],
            supercategory='none',
        ))

    total_gt_box_ids = 0
    for id, ((gt_boxes, gt_labels), (pred_boxes, pred_scores, pred_labels)) in enumerate(zip(coco_gts, coco_preds)):
        coco_gt_dict['images'].append(dict(
            id=id,
        ))
        for id_box, gt_box in enumerate(gt_boxes):
            gt_label = gt_labels[id_box]
            coco_gt_dict['annotations'].append(dict(
                id=total_gt_box_ids,
                image_id=id,
                category_id=1,
                area=float((gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])),
                bbox=xyxy2xywh(gt_box.tolist()),
                iscrowd=0,
            ))
            total_gt_box_ids += 1

            for i in range(len(gt_label)):
                if gt_label[i] >= 0.5:
                    coco_gt_dict['annotations'].append(dict(
                        id=total_gt_box_ids,
                        image_id=id,
                        category_id=i+2,
                        area=float((gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])),
                        bbox=xyxy2xywh(gt_box.tolist()),
                        iscrowd=0,
                    ))
                    total_gt_box_ids += 1

        for pred_box, pred_score, pred_label in zip(pred_boxes, pred_scores, pred_labels):
            coco_pred_dict.append(dict(
                image_id=id,
                category_id=1,
                bbox=xyxy2xywh(pred_box.tolist()),
                score=float(pred_score),
            ))
            for i in range(len(pred_label)):
                if pred_label[i] >= act_thr:
                    coco_pred_dict.append(dict(
                        image_id=id,
                        category_id=i+2,
                        bbox=xyxy2xywh(pred_box.tolist()),
                        score=float(pred_label[i]) if not model_type == 'dino' else float(pred_label[i]) * float(pred_score),
                    ))

    with open(f'./work_dirs/metric_temp/gt_json{save_suffix}.json', 'w') as f:
        json.dump(coco_gt_dict, f)
    with open(f'./work_dirs/metric_temp/pred_json{save_suffix}.json', 'w') as f:
        json.dump(coco_pred_dict, f)

def calc_iou(box1, box2):
    x1 = np.max([box1[0], box2[0]])
    y1 = np.max([box1[1], box2[1]])
    x2 = np.min([box1[2], box2[2]])
    y2 = np.min([box1[3], box2[3]])
    area_all = np.max([(x2 - x1 + 1), 0]) * np.max([(y2 - y1 + 1), 0])
    areas = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1) + (box2[2] - box2[0] + 1) * (
                box2[3] - box2[1] + 1) - area_all

    iou = area_all / areas
    return iou

def max_iou_assign(pred_boxes, gt_boxes):
    gt_indexes = list(range(len(gt_boxes)))
    assign_results = [-1] * len(pred_boxes)
    for i, pbox in enumerate(pred_boxes):
        max_iou = -1
        max_index = -1
        for j in gt_indexes:
            gbox = gt_boxes[j]
            iou = calc_iou(pbox, gbox)
            if iou > max_iou:
                max_iou = iou
                max_index = j
        assign_results[i] = max_index
    return assign_results



@METRICS.register_module()
class ChimpMetric(BaseMetric):
    '''A metric for chimp ana, super simple, super neat XD.'''

    def __init__(self, threshold_pos: float = 0.5,
                 threshold_act: float = 0.2,
                 action_class_num: int = 5,
                 action_class_names: Sequence[str] = None,
                 model_type: str = 'dino',
                 save_suffix: str = '',
                 # threshold_ars: Sequence[float] = (0.5, ),
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.threshold_pos = threshold_pos
        self.threshold_act = threshold_act
        self.model_type = model_type

        self.action_class_num = action_class_num
        self.action_class_names = action_class_names
        assert len(self.action_class_names) == self.action_class_num
        # self.threshold_ars = threshold_ars

        self.save_suffix = save_suffix

        self.coco_gts = []
        self.coco_preds = []
        self.det_results = {}
        self.feat_results = []

    def process(self, data_batch: Sequence[Tuple[Any, dict]],
                data_samples: Sequence[dict]) -> None:
        '''Stuff anything into result list.

        :param data_batch: data batch inputs which will not be used.
        :param data_samples: data samples included predictions, which is used for evaluation.
        :return: literally nothing.
        '''
        # print(data_samples)
        for data_sample in data_samples:
            result = dict()
            pred = data_sample['pred_instances']

            pred_features = None
            pred_cls_features = None
            if self.model_type == 'dino':
                pred_boxes = pred['bboxes'].cpu().numpy()
                pred_scores = pred['scores'].cpu().numpy()
                pred_labels = pred['labels'].cpu().numpy()
                pred_features = pred['feats'].cpu().numpy()
                pred_cls_features = pred['cls_feats'].cpu().numpy()

            if self.model_type == 'dino_no_obj':
                pred_boxes = pred['bboxes'].cpu().numpy()
                pred_scores = torch.max(pred['labels'], dim=-1)[0].cpu().numpy()
                pred_labels = pred['labels'].cpu().numpy()
                pred_features = pred['feats'].cpu().numpy()
                pred_cls_features = pred['cls_feats'].cpu().numpy()

            elif self.model_type == 'slowfast':
                pred_boxes = data_sample['proposals']['bboxes'].cpu().numpy()
                pred_scores = torch.max(pred['scores'], dim=-1)[0].cpu().numpy()
                pred_labels = pred['scores'].cpu().numpy()

            pred_pos_idx = pred_scores >= self.threshold_pos
            pred_boxes = pred_boxes[pred_pos_idx]
            pred_scores = pred_scores[pred_pos_idx]
            pred_labels = pred_labels[pred_pos_idx]
            if not pred_features is None:
                pred_features = pred_features[pred_pos_idx]
                pred_cls_features = pred_cls_features[pred_pos_idx]

            feat_pred_pos_idx = pred_scores >= 0.1
            feat_pred_boxes = pred_boxes[feat_pred_pos_idx]
            feat_pred_labels = pred_labels[feat_pred_pos_idx]
            if not pred_features is None:
                pred_features = pred_features[feat_pred_pos_idx]
                pred_cls_features = pred_cls_features[feat_pred_pos_idx]

            gt_bboxes = data_sample['gt_instances']['bboxes'].cpu().numpy()
            gt_labels = data_sample['gt_instances']['labels'].cpu().numpy()
            ass_idxs = max_iou_assign(feat_pred_boxes, gt_bboxes)
            gt_labels_exp = np.concatenate([gt_labels, np.zeros((1, gt_labels.shape[1]))], axis=0)
            feat_gt_labels = gt_labels_exp[ass_idxs]

            result['coco_gts'] = (gt_bboxes, gt_labels)
            result['coco_preds'] = (pred_boxes, pred_scores, pred_labels)
            result['det_res_key'] = data_sample['img_key']
            result['det_results'] = np.array([(x[0] / data_sample['img_shape'][0]).tolist() + [x[1]] for x in
                                              list(zip(pred_boxes, pred_scores))])
            if not pred_features is None:
                result['pred_feats'] = list(zip(feat_gt_labels, feat_pred_labels, pred_features, pred_cls_features))
            else:
                result['pred_feats'] = []
            self.results.append(result)


    def compute_metrics(self, results: list) -> dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed resu
            lts of each batch.
        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        time_now = datetime.now().strftime('%m/%d %H:%M:%S')
        print(f'{time_now} - Start computing metrics')

        metric_dct = {}
        self.coco_gts.clear()
        self.coco_preds.clear()
        self.det_results.clear()
        self.feat_results.clear()

        for res in results:
            self.coco_gts.append(res['coco_gts'])
            self.coco_preds.append(res['coco_preds'])
            self.det_results[res['det_res_key']] = res['det_results']
            self.feat_results += res['pred_feats']

        dump_to_coco(self.coco_gts, self.coco_preds, self.threshold_act, self.action_class_names, self.model_type, self.save_suffix)
        coco = COCO(f'./work_dirs/metric_temp/gt_json{self.save_suffix}.json')
        coco_dt = coco.loadRes(f'./work_dirs/metric_temp/pred_json{self.save_suffix}.json')
        coco_eval = COCOeval(coco, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()

        def _summarize(cocoobj, ap=1, class_name='', class_idx=None, iouThr=None, areaRng='all', maxDets=100 ):
            p = cocoobj.params
            iStr = ' {:<18} {} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap==1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = cocoobj.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                if class_idx is None:
                    s = s[:, :, 1:, aind, mind]
                elif isinstance(class_idx, list):
                    s = s[:, :, class_idx, aind, mind]
                else:
                    s = s[:, :, class_idx:class_idx+1, aind, mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = cocoobj.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]

                if class_idx is None:
                    s = s[:, 1:, aind, mind]
                elif isinstance(class_idx, list):
                    s = s[:, class_idx, aind, mind]
                else:
                    s = s[:, class_idx:class_idx+1, aind, mind]
            if len(s[s>-1])==0:
                mean_s = -1
            else:
                tmp = deepcopy(s)
                tmp[tmp<0.0] = 0.0
                mean_s = np.mean(tmp)
            print(iStr.format(titleStr, typeStr, class_name, iouStr, areaRng, maxDets, mean_s))
            return mean_s

        _summarize(coco_eval, 1, 'detection', 0, None)
        _summarize(coco_eval, 1, 'detection', 0, 0.50)
        _summarize(coco_eval, 1, 'detection', 0, 0.75)
        for i, cls_name in enumerate(self.action_class_names):
            _summarize(coco_eval, 1, cls_name, i+1, 0.50)
        _summarize(coco_eval, 1, 'locomotion', [2, 3, 4, 5], 0.50)
        _summarize(coco_eval, 1, 'object interaction', [6, 7, 8], 0.50)
        _summarize(coco_eval, 1, 'social', [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22], 0.50)
        action_map = _summarize(coco_eval, 1, 'action mean', None, 0.50)
        final_res = {'mAP': action_map}

        metric_dct['coco_eval_precision'] = coco_eval.eval['precision'].tolist()
        metric_dct['coco_eval_recall'] = coco_eval.eval['recall'].tolist()
        with open(f'./work_dirs/chimp_metrics{self.save_suffix}.json', 'w') as f:
            f.write(json.dumps(metric_dct)+'\n')
        with open(f'./work_dirs/chimp_metrics_det_results{self.save_suffix}.pkl', 'wb') as f:
            pickle.dump(self.det_results, f)
        with open(f'./work_dirs/chimp_metrics_feat_results{self.save_suffix}.pkl', 'wb') as f:
            pickle.dump(self.feat_results[:100000] if len(self.feat_results) > 100000 else self.feat_results, f)
        print('Evaluation results saved.', flush=True)

        self.coco_gts.clear()
        self.coco_preds.clear()
        self.det_results.clear()
        self.feat_results.clear()
        return final_res

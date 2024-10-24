# Copyright (c) OpenMMLab. All rights reserved.
import os
from time import time
from datetime import datetime
from typing import Any, List, Optional, Sequence, Tuple

import numpy
import json
import numpy as np
from mmengine.evaluator import BaseMetric

from mmaction.evaluation import ava_eval, results2csv
from mmaction.registry import METRICS
from mmaction.structures import bbox2result

def iou_single_bbox(boxes1: np.ndarray, boxes2: np.ndarray) -> float:
    x1, y1 = boxes1[:2]
    x2, y2 = boxes1[2:]
    x3, y3 = boxes2[:2]
    x4, y4 = boxes2[2:]

    inter_x1 = max(x1, x3)
    inter_y1 = max(y1, y3)
    inter_x2 = min(x2, x4)
    inter_y2 = min(y2, y4)

    inter_area = max(0, inter_x2 - inter_x1 + 1) * max(0, inter_y2 - inter_y1 + 1)

    box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    box2_area = (x4 - x3 + 1) * (y4 - y3 + 1)

    iou = inter_area / (box1_area + box2_area - inter_area + 1e-6)
    return iou

def batch_process(data_sample: dict) -> dict:
    pred_boxes = data_sample['outputs']['bboxes']
    pred_scores = data_sample['outputs']['scores']
    gt_boxes = data_sample['gts']['gt_bboxes']

    sorted_idx = np.argsort(-pred_scores, kind='mergesort')
    pred_boxes = pred_boxes[sorted_idx]
    pred_scores = pred_scores[sorted_idx]

    gt_box_idxs = list(range(len(gt_boxes)))

    iou_max_pred = [0.0] * len(pred_boxes)
    iou_max_gt = [0.0] * len(gt_boxes)
    for i, pred_box in enumerate(pred_boxes):
        max_iou_pred = -1.0
        max_iou_idx = -1
        if len(gt_box_idxs) == 0:
            break

        for j, idx in enumerate(gt_box_idxs):
            gt_box = gt_boxes[idx]
            iou = iou_single_bbox(pred_box, gt_box)
            if iou > max_iou_pred:
                max_iou_idx = idx
                max_iou_pred = iou

        iou_max_gt[max_iou_idx] = max_iou_pred
        iou_max_pred[i] = max_iou_pred
        gt_box_idxs.remove(max_iou_idx)

    iou_max_gt = np.array(iou_max_gt)
    iou_max_pred = np.array(iou_max_pred)

    return {'iou_max_pred': iou_max_pred, 'iou_max_gt': iou_max_gt, 'pred_scores': pred_scores}

def calculate_average_precision(processed_res, thr=0.5) -> Tuple[float, list, list]:
    pred_ious = np.concatenate([x['iou_max_pred'] for x in processed_res], axis=0)
    pred_scores = np.concatenate([x['pred_scores'] for x in processed_res], axis=0)
    gt_ious = np.concatenate([x['iou_max_gt'] for x in processed_res], axis=0)
    num_positives = len(gt_ious)

    sorted_idx = np.argsort(-pred_scores, kind='mergesort')
    pred_ious = pred_ious[sorted_idx]

    preds_labels = (pred_ious >= thr).astype(int)
    # print('IOU THR:', thr, 'PREDS', preds_labels)
    tp, fp = 0, 0

    precisions = []
    recalls = []
    for pred in preds_labels:
        if pred == 1:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / (tp + fp))
        recalls.append(tp / num_positives)

    # print('PRECISIONS', precisions)
    # print('RECALLS', recalls)
    average_precision = np.trapz(precisions, recalls)
    return average_precision, precisions, recalls



@METRICS.register_module()
class ChimpDetMetric(BaseMetric):
    '''A metric for chimp detection, super simple, super neat XD.'''

    def __init__(self, threshold_aps: Sequence[float] = (0.5, ),
                 save_suffix: str = '',
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.threshold_aps = threshold_aps
        self.save_suffix = save_suffix

    def process(self, data_batch: Sequence[Tuple[Any, dict]],
                data_samples: Sequence[dict]) -> None:
        '''Stuff anything into result list.

        :param data_batch: data batch inputs which will not be used.
        :param data_samples: data samples included predictions, which is used for evaluation.
        :return: literally nothing.
        '''
        for data_sample in data_samples:
            result = dict()
            pred = data_sample['pred_instances']
            result['video_id'] = data_sample['video_id']
            result['timestamp'] = data_sample['timestamp']
            result['outputs'] = {'bboxes': pred['bboxes'].cpu().numpy(), 'scores': pred['scores'].cpu().numpy()}
            result['gts'] = {'gt_bboxes': data_sample['gt_instances']['bboxes'].cpu().numpy()}
            result = batch_process(result)
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
        # print(results[0])
        time_now = datetime.now().strftime('%m/%d %H:%M:%S')

        start_time = time()
        print(f'{time_now} - Start computing metrics')

        #process_res = []
        #for batch in results:
        #    process_res.append(batch_process(batch))


        eval_res = {}
        final_sum = 0.0

        ps_50, rs_50 = None, None
        for thr in self.threshold_aps:
            ap_thr, ps, rs = calculate_average_precision(results, thr)
            if thr == 0.5:
                ps_50, rs_50 = ps, rs

            eval_res[f'ap@thr={thr}'] = ap_thr
            if ap_thr != float('nan'):
                final_sum += ap_thr

        final_res = {'mAP':final_sum / len(self.threshold_aps)}

        end_time = time()
        print(f'Computation time: {end_time - start_time}')
        print(f'Compute results:')
        for k, v in eval_res.items():
            print(f'\t{k}: {v}')
            final_res[k] = v

        with open(f'./work_dirs/chimp_metrics_4class_pr_trapz{self.save_suffix}.json', 'w') as f:
            f.write(json.dumps(dict(precisions=ps_50, recalls=rs_50)))

        return final_res

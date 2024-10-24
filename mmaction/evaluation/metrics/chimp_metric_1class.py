# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
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

def assign_process(data_sample: dict, pos_thr=0.5) -> dict:
    pred_boxes = data_sample['outputs']['bboxes']
    pred_labels = data_sample['outputs']['labels']
    pred_scores = data_sample['outputs']['scores']

    pred_pos_idx = pred_scores >= pos_thr
    pred_boxes = pred_boxes[pred_pos_idx]
    pred_labels = pred_labels[pred_pos_idx]
    pred_scores = pred_scores[pred_pos_idx]

    if len(pred_boxes) == 0:
        return None

    #print('pred_labels:', pred_labels)
    #print('pred_scores:', pred_scores)
    gt_boxes = data_sample['gts']['gt_bboxes']
    gt_labels = data_sample['gts']['gt_labels']

    pred_box_indexes = list(range(len(pred_boxes)))
    gt_max_iou = [0] * len(gt_boxes)
    assign_max_iou = [0] * len(pred_boxes)

    unassigned_gt_labels = np.zeros_like(gt_labels)
    assigned_pred_gt_labels = []
    for i, gt_box in enumerate(gt_boxes):
        max_iou = 0.0
        max_iou_idx = 0
        if len(pred_box_indexes) == 0:
            unassigned_gt_labels = gt_labels[i:, :]
            break
        for j in pred_box_indexes:
            pred_box = pred_boxes[j]
            iou = iou_single_bbox(pred_box, gt_box)
            if iou > max_iou:
                max_iou = iou
                max_iou_idx = j
        if max_iou_idx in pred_box_indexes:
            pred_box_indexes.remove(max_iou_idx)
        assign_max_iou[max_iou_idx] = max_iou
        gt_max_iou[i] = max_iou

        # print('INDEX', i, 'GT_BOX', gt_box)
        # print('INDEX', i, 'PRED_BOX', pred_boxes[max_iou_idx])
        # print('INDEX', i, 'IOUUU', max_iou)

        assigned_pred_gt_labels.append((pred_labels[max_iou_idx], gt_labels[i]))

    assign_max_iou = np.array(assign_max_iou)
    gt_max_iou = np.array(gt_max_iou)

    return {'pred_gt_label_pairs': assigned_pred_gt_labels, 'unassigned_gt_labels': unassigned_gt_labels, 'pred_scores': pred_scores, 'pred_ious': assign_max_iou, 'gt_ious': gt_max_iou}

def calculate_det_average_precision(processed_res, thr=0.5) -> float:
    if len([x['pred_ious'] for x in processed_res if x is not None]) == 0:
        return 0.0
    pred_ious = np.concatenate([x['pred_ious'] for x in processed_res if x is not None], axis=0)
    pred_scores = np.concatenate([x['pred_scores'] for x in processed_res if x is not None], axis=0)
    # pred_labels = np.concatenate([x['pred_labels'] for x in processed_res if x is not None], axis=0)
    # gt_labels = np.concatenate([x['gt_labels'] for x in processed_res if x is not None], axis=0)

    # print('pred_scores: ', pred_scores)
    # print('pred_ious: ', pred_ious)
    gt_ious = np.concatenate([x['gt_ious'] for x in processed_res if x is not None], axis=0)
    num_positives = len(gt_ious)

    sorted_idx = np.flip(np.argsort(pred_scores, kind='mergesort'))
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
    # print('True Positives Num:', tp, flush=True)
    # print('False Positives Num:', fp, flush=True)
    # print('Real Positives Num:', num_positives, flush=True)

    # print('PRECISIONS', precisions)
    # print('RECALLS', recalls)
    average_precision = np.trapz(precisions, recalls)
    return average_precision

def calculate_act_average_precision(processed_res, thr=0.2, num_classes=24) -> tuple:
    label_pairs = []
    for res in processed_res:
        if res is not None:
            label_pairs += res['pred_gt_label_pairs']
    if len(label_pairs) == 0:
        return ([0] * num_classes, [0] * num_classes, [0] * num_classes, [0] * num_classes,
                [0] * num_classes, [0] * num_classes, [0] * num_classes, [0] * num_classes, [0] * num_classes)

    pred_labels = np.stack([x[0] for x in label_pairs])
    gt_labels = np.stack([x[1] for x in label_pairs])
    np_gt_labels = np.concatenate([x['unassigned_gt_labels'] for x in processed_res if x is not None], axis=0)

    pred_labels = (pred_labels >= thr).astype(int)
    gt_labels = gt_labels.astype(int)
    np_gt_labels = np_gt_labels.astype(int)

    tps = ((pred_labels + gt_labels) == 2).astype(int).sum(axis=0)
    fps = ((gt_labels - pred_labels) == -1).astype(int).sum(axis=0)
    tns = ((pred_labels + gt_labels) == 0).astype(int).sum(axis=0)
    fns = ((pred_labels - gt_labels) == -1).astype(int).sum(axis=0)
    nps = np_gt_labels.sum(axis=0)

    precisions_per_class = (tps) / (tps + fps)
    precisions_w_np_per_class = (tps) / (tps + fps + nps)
    recalls_per_class = (tps) / (tps + fns)
    recalls_w_np_per_class = (tps) / (tps + fns + nps)
    map_act = np.nansum(precisions_per_class) / len(precisions_per_class)
    mar_act = np.nansum(recalls_per_class) / len(recalls_per_class)

    return precisions_per_class, precisions_w_np_per_class, recalls_per_class, recalls_w_np_per_class, tps, fps, tns, fns, nps, map_act, mar_act


@METRICS.register_module()
class ChimpMetric1Class(BaseMetric):
    '''A metric for chimp ana, super simple, super neat XD.'''

    def __init__(self, threshold_pos: float = 0.5,
                 threshold_act: float = 0.2,
                 threshold_aps: Sequence[float] = (0.5, ),
                 # threshold_ars: Sequence[float] = (0.5, ),
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.threshold_pos = threshold_pos
        self.threshold_act = threshold_act
        self.threshold_aps = threshold_aps

        self.action_class_names = ['chimp']
        # self.threshold_ars = threshold_ars

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
            result['video_id'] = data_sample['video_id']
            result['timestamp'] = data_sample['timestamp']

            #print(pred)
            result['outputs'] = {'bboxes': pred['bboxes'].cpu().numpy(), 'labels':pred['labels'].cpu().numpy(), 'scores': pred['scores'].cpu().numpy()}
            result['gts'] = {'gt_bboxes': data_sample['gt_instances']['bboxes'].cpu().numpy(), 'gt_labels': data_sample['gt_instances']['labels'].cpu().numpy()}
            result = assign_process(result, self.threshold_pos)
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
        metric_dct = {}

        bbox_res = {}
        final_sum = 0.0
        for thr in self.threshold_aps:
            ap_thr = calculate_det_average_precision(results, thr)
            bbox_res[f'ap@thr={thr}'] = ap_thr
            metric_dct[f'det_ap@thr={thr}'] = ap_thr
            if ap_thr != float('nan'):
                final_sum += ap_thr
        map_det = final_sum / len(self.threshold_aps)
        metric_dct['det_map'] = map_det

        pc_act, pc_w_act, rc_act, rc_w_act, tps, fps, tns, fns, nps, map_act, mar_act = calculate_act_average_precision(results, self.threshold_act, len(self.action_class_names))
        #for thr in self.threshold_ars:
        #    ar_thr = cal_ar(process_res, thr)
        #    eval_res[f'ar@thr={thr}'] = ar_thr
        #    if ar_thr != float('nan'):
        #        final_sum += ar_thr
        #final_sum /= len(self.threshold_aps) + len(self.threshold_ars)
        final_res = {'mAP': (map_det + map_act + mar_act) / 3}
        metric_dct['act_map'] = map_act
        metric_dct['act_mar'] = mar_act

        end_time = time()
        print(f'Computation time: {end_time - start_time}')
        print(f'Compute results:')
        print(f'BBOX Results (w/o label):')
        for k, v in bbox_res.items():
            print(f'\t{k}: {v}')
        print(f'Action Results:')
        print(f'\tmAP: {map_act}')
        print(f'\tmAR: {mar_act}')

        for i in range(len(pc_act)):
            metric_dct[f'ap_{self.action_class_names[i]}'] = pc_act[i]
            metric_dct[f'ar_{self.action_class_names[i]}'] = rc_act[i]

            print(f'Class {self.action_class_names[i]}:')
            print('\tPrecision: w/o NP:'+format(pc_act[i], '.3f')+' w/ Np:'+format(pc_w_act[i], '.3f')+' - '+
                  'Recall: w/o NP:'+format(rc_act[i], '.3f')+' w/ Np:'+format(rc_w_act[i], '.3f'))
            print(f'\tTPs:{tps[i]}, FPs:{fps[i]}, TNs:{tns[i]}, FNs:{fns[i]}, NPs:{nps[i]}')

        with open('./work_dirs/chimp_metrics_4class.json', 'a') as f:
            f.write(json.dumps(metric_dct)+'\n')

        return final_res

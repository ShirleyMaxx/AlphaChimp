# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.bbox import (bbox_cxcywh_to_xyxy, bbox_overlaps,
                                   bbox_xyxy_to_cxcywh)
from mmdet.utils import InstanceList, OptInstanceList, reduce_mean
from ..losses import QualityFocalLoss
from ..utils import multi_apply
from .deformable_detr_head import DeformableDETRHead

class HierarchyFC(nn.Module):
    def __init__(self, embed_dims = 512,
                 num_classes = 5,
                 num_cls_fc = 4,
                 hierarchy_class_nums = None,
                 hierarchy_class_indexes = None):
        super(HierarchyFC, self).__init__()
        self.embed_dims = embed_dims
        self.num_classes = num_classes
        self.hierarchy_class_nums = hierarchy_class_nums
        self.hierarchy_class_indexes = hierarchy_class_indexes

        assert hierarchy_class_nums is not None and hierarchy_class_indexes is not None
        assert len(hierarchy_class_indexes) == len(hierarchy_class_nums)


        cls_branch = []
        for num in hierarchy_class_nums:
            cls_model = []
            for _ in range(num_cls_fc):
                cls_model.append(Linear(self.embed_dims, self.embed_dims))
                cls_model.append(nn.ReLU())
            cls_model.append(Linear(self.embed_dims, num))
            cls_model = nn.Sequential(*cls_model)

            cls_branch.append(cls_model)
        self.cls_branches = nn.ModuleList(cls_branch)

    def forward(self, hidden_state, cls_result):
        '''Do hierarchy classification.

        :param hidden_state: bs, num_queries, embed_dims
        :param cls_result: bs, num_queries, num_classes
        :return: hierarchy cls result, (bs, num_queries, num_classes + sum of hierarchy class nums).
        '''
        ret_list = [cls_result]
        for branch in range(len(self.hierarchy_class_indexes)):
            ret_list.append(self.cls_branches[branch](hidden_state))
        return torch.cat(ret_list, dim=2)


@MODELS.register_module()
class DINOHeadHierarchy(DeformableDETRHead):
    r"""Head of the DINO: DETR with Improved DeNoising Anchor Boxes
    for End-to-End Object Detection

    Code is modified from the `official github repo
    <https://github.com/IDEA-Research/DINO>`_.

    More details can be found in the `paper
    <https://arxiv.org/abs/2203.03605>`_ .
    """
    def __init__(self,
                 *args,
                 num_cls_fcs = 4,
                 hierarchy_class_nums = None,
                 hierarchy_class_index: list = None,
                 **kwargs) -> None:
        if hierarchy_class_nums is None:
            hierarchy_class_nums = [5, 6, 2]
            hierarchy_class_index = [2, 3, 4]
        self.num_cls_fcs = num_cls_fcs
        self.hierarchy_class_nums = hierarchy_class_nums
        self.hierarchy_class_index = hierarchy_class_index

        super().__init__(*args, **kwargs)

    def _init_layers(self) -> None:
        """Initialize classification branch and regression branch of head."""
        cls_branch = []
        for _ in range(self.num_cls_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.ReLU())
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        cls_branch = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, 4))
        reg_branch = nn.Sequential(*reg_branch)

        hierarchy_branch = HierarchyFC(self.embed_dims, self.num_classes, self.num_cls_fcs, self.hierarchy_class_nums, self.hierarchy_class_index)

        if self.share_pred_layer:
            self.cls_branches = nn.ModuleList(
                [cls_branch for _ in range(self.num_pred_layer)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(self.num_pred_layer)])
            self.hierarchy_branches = nn.ModuleList(
                [hierarchy_branch for _ in range(self.num_pred_layer)])
        else:
            self.cls_branches = nn.ModuleList(
                [copy.deepcopy(cls_branch) for _ in range(self.num_pred_layer)])
            self.reg_branches = nn.ModuleList([
                copy.deepcopy(reg_branch) for _ in range(self.num_pred_layer)])
            self.hierarchy_branches = nn.ModuleList(
                [copy.deepcopy(hierarchy_branch) for _ in range(self.num_pred_layer)])

    def init_weights(self) -> None:
        """Initialize weights of the Deformable DETR head."""
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, bias_init)
        for m in self.reg_branches:
            constant_init(m[-1], 0, bias=0)
        nn.init.constant_(self.reg_branches[0][-1].bias.data[2:], -2.0)
        if self.as_two_stage:
            for m in self.reg_branches:
                nn.init.constant_(m[-1].bias.data[2:], 0.0)

    def forward(self, hidden_states: Tensor,
                references: List[Tensor]) -> Tuple[Tensor, Tensor]:
        """Forward function.

        Args:
            hidden_states (Tensor): Hidden states output from each decoder
                layer, has shape (num_decoder_layers, bs, num_queries, dim).
            references (list[Tensor]): List of the reference from the decoder.
                The first reference is the `init_reference` (initial) and the
                other num_decoder_layers(6) references are `inter_references`
                (intermediate). The `init_reference` has shape (bs,
                num_queries, 4) when `as_two_stage` of the detector is `True`,
                otherwise (bs, num_queries, 2). Each `inter_reference` has
                shape (bs, num_queries, 4) when `with_box_refine` of the
                detector is `True`, otherwise (bs, num_queries, 2). The
                coordinates are arranged as (cx, cy) when the last dimension is
                2, and (cx, cy, w, h) when it is 4.

        Returns:
            tuple[Tensor]: results of head containing the following tensor.

            - all_layers_outputs_classes (Tensor): Outputs from the
              classification head, has shape (num_decoder_layers, bs,
              num_queries, cls_out_channels).
            - all_layers_outputs_coords (Tensor): Sigmoid outputs from the
              regression head with normalized coordinate format (cx, cy, w,
              h), has shape (num_decoder_layers, bs, num_queries, 4) with the
              last dimension arranged as (cx, cy, w, h).
        """
        all_layers_outputs_classes = []
        all_layers_outputs_coords = []

        for layer_id in range(hidden_states.shape[0]):
            reference = inverse_sigmoid(references[layer_id])
            # NOTE The last reference will not be used.
            hidden_state = hidden_states[layer_id]
            outputs_class = self.cls_branches[layer_id](hidden_state)
            outputs_class = self.hierarchy_branches[layer_id](hidden_state, outputs_class)
            tmp_reg_preds = self.reg_branches[layer_id](hidden_state)
            if reference.shape[-1] == 4:
                # When `layer` is 0 and `as_two_stage` of the detector
                # is `True`, or when `layer` is greater than 0 and
                # `with_box_refine` of the detector is `True`.
                tmp_reg_preds += reference
            else:
                # When `layer` is 0 and `as_two_stage` of the detector
                # is `False`, or when `layer` is greater than 0 and
                # `with_box_refine` of the detector is `False`.
                assert reference.shape[-1] == 2
                tmp_reg_preds[..., :2] += reference
            outputs_coord = tmp_reg_preds.sigmoid()
            all_layers_outputs_classes.append(outputs_class)
            all_layers_outputs_coords.append(outputs_coord)

        all_layers_outputs_classes = torch.stack(all_layers_outputs_classes)
        all_layers_outputs_coords = torch.stack(all_layers_outputs_coords)

        return all_layers_outputs_classes, all_layers_outputs_coords

    def loss(self, hidden_states: Tensor, references: List[Tensor],
             enc_outputs_class: Tensor, enc_outputs_coord: Tensor,
             batch_data_samples: SampleList, dn_meta: Dict[str, int]) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the queries of the upstream network.

        Args:
            hidden_states (Tensor): Hidden states output from each decoder
                layer, has shape (num_decoder_layers, bs, num_queries_total,
                dim), where `num_queries_total` is the sum of
                `num_denoising_queries` and `num_matching_queries` when
                `self.training` is `True`, else `num_matching_queries`.
            references (list[Tensor]): List of the reference from the decoder.
                The first reference is the `init_reference` (initial) and the
                other num_decoder_layers(6) references are `inter_references`
                (intermediate). The `init_reference` has shape (bs,
                num_queries_total, 4) and each `inter_reference` has shape
                (bs, num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h).
            enc_outputs_class (Tensor): The score of each point on encode
                feature map, has shape (bs, num_feat_points, cls_out_channels).
            enc_outputs_coord (Tensor): The proposal generate from the
                encode feature map, has shape (bs, num_feat_points, 4) with the
                last dimension arranged as (cx, cy, w, h).
            batch_data_samples (list[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'. It will be used for split outputs of
              denoising and matching parts and loss calculation.

        Returns:
            dict: A dictionary of loss components.
        """
        batch_gt_instances = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)

        outs = self(hidden_states, references)
        loss_inputs = outs + (enc_outputs_class, enc_outputs_coord,
                              batch_gt_instances, batch_img_metas, dn_meta)
        losses = self.loss_by_feat(*loss_inputs)
        return losses

    def loss_by_feat(
        self,
        all_layers_cls_scores: Tensor,
        all_layers_bbox_preds: Tensor,
        enc_cls_scores: Tensor,
        enc_bbox_preds: Tensor,
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        dn_meta: Dict[str, int],
        batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, Tensor]:
        """Loss function.

        Args:
            all_layers_cls_scores (Tensor): Classification scores of all
                decoder layers, has shape (num_decoder_layers, bs,
                num_queries_total, cls_out_channels), where
                `num_queries_total` is the sum of `num_denoising_queries`
                and `num_matching_queries`.
            all_layers_bbox_preds (Tensor): Regression outputs of all decoder
                layers. Each is a 4D-tensor with normalized coordinate format
                (cx, cy, w, h) and has shape (num_decoder_layers, bs,
                num_queries_total, 4).
            enc_cls_scores (Tensor): The score of each point on encode
                feature map, has shape (bs, num_feat_points, cls_out_channels).
            enc_bbox_preds (Tensor): The proposal generate from the encode
                feature map, has shape (bs, num_feat_points, 4) with the last
                dimension arranged as (cx, cy, w, h).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            dn_meta (Dict[str, int]): The dictionary saves information about
                group collation, including 'num_denoising_queries' and
                'num_denoising_groups'. It will be used for split outputs of
                denoising and matching parts and loss calculation.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # extract denoising and matching part of outputs
        (all_layers_matching_cls_scores, all_layers_matching_bbox_preds,
         all_layers_denoising_cls_scores, all_layers_denoising_bbox_preds) = \
            self.split_outputs(
                all_layers_cls_scores, all_layers_bbox_preds, dn_meta)

        loss_dict = self.loss_by_feat_hierarchy(all_layers_matching_cls_scores, all_layers_matching_bbox_preds,
            batch_gt_instances, batch_img_metas, batch_gt_instances_ignore)
        # NOTE DETRHead.loss_by_feat but not DeformableDETRHead.loss_by_feat
        # is called, because the encoder loss calculations are different
        # between DINO and DeformableDETR.

        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            # NOTE The enc_loss calculation of the DINO is
            # different from that of Deformable DETR.
            enc_loss_cls, enc_loss_hierarchy, enc_losses_bbox, enc_losses_iou = \
                self.loss_by_feat_single_hierarchy(
                    enc_cls_scores, enc_bbox_preds,
                    batch_gt_instances=batch_gt_instances,
                    batch_img_metas=batch_img_metas)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_hierarchy'] = enc_loss_hierarchy
            loss_dict['enc_loss_bbox'] = enc_losses_bbox
            loss_dict['enc_loss_iou'] = enc_losses_iou

        if all_layers_denoising_cls_scores is not None:
            # calculate denoising loss from all decoder layers
            dn_losses_cls, dn_losses_hierarchy, dn_losses_bbox, dn_losses_iou = self.loss_dn(
                all_layers_denoising_cls_scores,
                all_layers_denoising_bbox_preds,
                batch_gt_instances=batch_gt_instances,
                batch_img_metas=batch_img_metas,
                dn_meta=dn_meta)
            # collate denoising loss
            loss_dict['dn_loss_cls'] = dn_losses_cls[-1]
            loss_dict['dn_loss_hierarchy'] = dn_losses_hierarchy[-1]
            loss_dict['dn_loss_bbox'] = dn_losses_bbox[-1]
            loss_dict['dn_loss_iou'] = dn_losses_iou[-1]
            for num_dec_layer, (loss_cls_i, loss_hierarchy_i, loss_bbox_i, loss_iou_i) in \
                    enumerate(zip(dn_losses_cls[:-1], dn_losses_hierarchy[:-1], dn_losses_bbox[:-1],
                                  dn_losses_iou[:-1])):
                loss_dict[f'd{num_dec_layer}.dn_loss_cls'] = loss_cls_i
                loss_dict[f'd{num_dec_layer}.dn_loss_hierarchy'] = loss_hierarchy_i
                loss_dict[f'd{num_dec_layer}.dn_loss_bbox'] = loss_bbox_i
                loss_dict[f'd{num_dec_layer}.dn_loss_iou'] = loss_iou_i
        return loss_dict

    def loss_dn(self, all_layers_denoising_cls_scores: Tensor,
                all_layers_denoising_bbox_preds: Tensor,
                batch_gt_instances: InstanceList, batch_img_metas: List[dict],
                dn_meta: Dict[str, int]) -> Tuple[List[Tensor]]:
        """Calculate denoising loss.

        Args:
            all_layers_denoising_cls_scores (Tensor): Classification scores of
                all decoder layers in denoising part, has shape (
                num_decoder_layers, bs, num_denoising_queries,
                cls_out_channels).
            all_layers_denoising_bbox_preds (Tensor): Regression outputs of all
                decoder layers in denoising part. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and has shape
                (num_decoder_layers, bs, num_denoising_queries, 4).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'. It will be used for split outputs of
              denoising and matching parts and loss calculation.

        Returns:
            Tuple[List[Tensor]]: The loss_dn_cls, loss_dn_bbox, and loss_dn_iou
            of each decoder layers.
        """
        return multi_apply(
            self._loss_dn_single,
            all_layers_denoising_cls_scores,
            all_layers_denoising_bbox_preds,
            batch_gt_instances=batch_gt_instances,
            batch_img_metas=batch_img_metas,
            dn_meta=dn_meta)

    def _loss_dn_single(self, dn_cls_scores: Tensor, dn_bbox_preds: Tensor,
                        batch_gt_instances: InstanceList,
                        batch_img_metas: List[dict],
                        dn_meta: Dict[str, int]) -> Tuple[Tensor]:
        """Denoising loss for outputs from a single decoder layer.

        Args:
            dn_cls_scores (Tensor): Classification scores of a single decoder
                layer in denoising part, has shape (bs, num_denoising_queries,
                cls_out_channels).
            dn_bbox_preds (Tensor): Regression outputs of a single decoder
                layer in denoising part. Each is a 4D-tensor with normalized
                coordinate format (cx, cy, w, h) and has shape
                (bs, num_denoising_queries, 4).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'. It will be used for split outputs of
              denoising and matching parts and loss calculation.

        Returns:
            Tuple[Tensor]: A tuple including `loss_cls`, `loss_box` and
            `loss_iou`.
        """
        num_imgs = dn_cls_scores.size(0)
        cls_reg_targets = self.get_dn_targets(batch_gt_instances,
                                              batch_img_metas, dn_meta)
        (labels_list, label_obj_weights_list, label_cls_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_obj_weights = torch.cat(label_obj_weights_list, 0)
        label_cls_weights = torch.cat(label_cls_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = dn_cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        #cls_avg_factor = \
        #    num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        #if self.sync_cls_avg_factor:
        #    cls_avg_factor = reduce_mean(
        #        cls_scores.new_tensor([cls_avg_factor]))
        #cls_avg_factor = max(cls_avg_factor, 1)
        cls_avg_factor = 1 / num_imgs

        if len(cls_scores) > 0:
            loss_cls = self.loss_cls(
                cls_scores,
                labels,
                obj_weight=label_obj_weights,
                cls_weight=label_cls_weights,
                avg_factor=cls_avg_factor)
        else:
            loss_cls = torch.zeros(
                1, dtype=cls_scores.dtype, device=cls_scores.device)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(batch_img_metas, dn_bbox_preds):
            img_h, img_w = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = dn_bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)

        # regression L1 loss
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)
        return loss_cls, loss_bbox, loss_iou

    def get_dn_targets(self, batch_gt_instances: InstanceList,
                       batch_img_metas: dict, dn_meta: Dict[str,
                                                            int]) -> tuple:
        """Get targets in denoising part for a batch of images.

        Args:
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'. It will be used for split outputs of
              denoising and matching parts and loss calculation.

        Returns:
            tuple: a tuple containing the following targets.

            - labels_list (list[Tensor]): Labels for all images.
            - label_weights_list (list[Tensor]): Label weights for all images.
            - bbox_targets_list (list[Tensor]): BBox targets for all images.
            - bbox_weights_list (list[Tensor]): BBox weights for all images.
            - num_total_pos (int): Number of positive samples in all images.
            - num_total_neg (int): Number of negative samples in all images.
        """
        (labels_list, label_obj_weights_list, label_cls_weights_list, bbox_targets_list, bbox_weights_list,
         pos_inds_list, neg_inds_list) = multi_apply(
             self._get_dn_targets_single,
             batch_gt_instances,
             batch_img_metas,
             dn_meta=dn_meta)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_obj_weights_list, label_cls_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def _get_dn_targets_single(self, gt_instances: InstanceData,
                               img_meta: dict, dn_meta: Dict[str,
                                                             int]) -> tuple:
        """Get targets in denoising part for one image.

        Args:
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It should includes ``bboxes`` and ``labels``
                attributes.
            img_meta (dict): Meta information for one image.
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'. It will be used for split outputs of
              denoising and matching parts and loss calculation.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

            - labels (Tensor): Labels of each image.
            - label_weights (Tensor]): Label weights of each image.
            - bbox_targets (Tensor): BBox targets of each image.
            - bbox_weights (Tensor): BBox weights of each image.
            - pos_inds (Tensor): Sampled positive indices for each image.
            - neg_inds (Tensor): Sampled negative indices for each image.
        """
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        num_groups = dn_meta['num_denoising_groups']
        num_denoising_queries = dn_meta['num_denoising_queries']
        num_queries_each_group = int(num_denoising_queries / num_groups)
        device = gt_bboxes.device

        if len(gt_labels) > 0:
            t = torch.arange(len(gt_labels), dtype=torch.long, device=device)
            t = t.unsqueeze(0).repeat(num_groups, 1)
            pos_assigned_gt_inds = t.flatten()
            pos_inds = torch.arange(
                num_groups, dtype=torch.long, device=device)
            pos_inds = pos_inds.unsqueeze(1) * num_queries_each_group + t
            pos_inds = pos_inds.flatten()
        else:
            pos_inds = pos_assigned_gt_inds = \
                gt_bboxes.new_tensor([], dtype=torch.long)

        neg_inds = pos_inds + num_queries_each_group // 2

        # label targets
        labels = gt_bboxes.new_full((num_denoising_queries, self.num_classes + 1),
                                    0,
                                    dtype=torch.long)
        gt_labels_exp = torch.cat(
            [torch.ones((gt_labels.size(0), 1), device=gt_labels.device, dtype=gt_labels.dtype), gt_labels], dim=1)
        labels[pos_inds] = gt_labels_exp[pos_assigned_gt_inds].long()
        label_obj_weights = gt_bboxes.new_ones(num_denoising_queries)
        label_obj_weights[pos_inds] = 2.0
        label_cls_weights = gt_bboxes.new_full((num_denoising_queries,), 0, dtype=torch.float)
        label_cls_weights[pos_inds] = 1.0

        # bbox targets
        bbox_targets = torch.zeros(num_denoising_queries, 4, device=device)
        bbox_weights = torch.zeros(num_denoising_queries, 4, device=device)
        bbox_weights[pos_inds] = 1.0
        img_h, img_w = img_meta['img_shape']

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        factor = gt_bboxes.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)
        gt_bboxes_normalized = gt_bboxes / factor
        gt_bboxes_targets = bbox_xyxy_to_cxcywh(gt_bboxes_normalized)
        bbox_targets[pos_inds] = gt_bboxes_targets.repeat([num_groups, 1])

        return (labels, label_obj_weights, label_cls_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds)

    @staticmethod
    def split_outputs(all_layers_cls_scores: Tensor,
                      all_layers_bbox_preds: Tensor,
                      dn_meta: Dict[str, int]) -> Tuple[Tensor]:
        """Split outputs of the denoising part and the matching part.

        For the total outputs of `num_queries_total` length, the former
        `num_denoising_queries` outputs are from denoising queries, and
        the rest `num_matching_queries` ones are from matching queries,
        where `num_queries_total` is the sum of `num_denoising_queries` and
        `num_matching_queries`.

        Args:
            all_layers_cls_scores (Tensor): Classification scores of all
                decoder layers, has shape (num_decoder_layers, bs,
                num_queries_total, cls_out_channels).
            all_layers_bbox_preds (Tensor): Regression outputs of all decoder
                layers. Each is a 4D-tensor with normalized coordinate format
                (cx, cy, w, h) and has shape (num_decoder_layers, bs,
                num_queries_total, 4).
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'.

        Returns:
            Tuple[Tensor]: a tuple containing the following outputs.

            - all_layers_matching_cls_scores (Tensor): Classification scores
              of all decoder layers in matching part, has shape
              (num_decoder_layers, bs, num_matching_queries, cls_out_channels).
            - all_layers_matching_bbox_preds (Tensor): Regression outputs of
              all decoder layers in matching part. Each is a 4D-tensor with
              normalized coordinate format (cx, cy, w, h) and has shape
              (num_decoder_layers, bs, num_matching_queries, 4).
            - all_layers_denoising_cls_scores (Tensor): Classification scores
              of all decoder layers in denoising part, has shape
              (num_decoder_layers, bs, num_denoising_queries,
              cls_out_channels).
            - all_layers_denoising_bbox_preds (Tensor): Regression outputs of
              all decoder layers in denoising part. Each is a 4D-tensor with
              normalized coordinate format (cx, cy, w, h) and has shape
              (num_decoder_layers, bs, num_denoising_queries, 4).
        """
        num_denoising_queries = dn_meta['num_denoising_queries']
        if dn_meta is not None:
            all_layers_denoising_cls_scores = \
                all_layers_cls_scores[:, :, : num_denoising_queries, :]
            all_layers_denoising_bbox_preds = \
                all_layers_bbox_preds[:, :, : num_denoising_queries, :]
            all_layers_matching_cls_scores = \
                all_layers_cls_scores[:, :, num_denoising_queries:, :]
            all_layers_matching_bbox_preds = \
                all_layers_bbox_preds[:, :, num_denoising_queries:, :]


        else:
            all_layers_denoising_cls_scores = None
            all_layers_denoising_bbox_preds = None
            all_layers_matching_cls_scores = all_layers_cls_scores
            all_layers_matching_bbox_preds = all_layers_bbox_preds
        return (all_layers_matching_cls_scores, all_layers_matching_bbox_preds,
                all_layers_denoising_cls_scores, all_layers_denoising_bbox_preds)

    def get_targets_widx(self, cls_scores_list: List[Tensor],
                    bbox_preds_list: List[Tensor],
                    batch_gt_instances: InstanceList,
                    batch_img_metas: List[dict]) -> tuple:
        (labels_list, label_obj_weights_list, label_cls_weights_list, bbox_targets_list, bbox_weights_list,
         pos_inds_list,
         neg_inds_list, pos_gt_inds_list) = multi_apply(self.get_targets_widx_single,
                                      cls_scores_list, bbox_preds_list,
                                      batch_gt_instances, batch_img_metas)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_obj_weights_list, label_cls_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg, pos_inds_list, neg_inds_list, pos_gt_inds_list)

    def get_targets_widx_single(self, cls_score: Tensor, bbox_pred: Tensor,
                            gt_instances: InstanceData,
                            img_meta: dict) -> tuple:
        img_h, img_w = img_meta['img_shape']
        factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)
        num_bboxes = bbox_pred.size(0)
        # convert bbox_pred from xywh, normalized to xyxy, unnormalized
        bbox_pred = bbox_cxcywh_to_xyxy(bbox_pred)
        bbox_pred = bbox_pred * factor

        pred_instances = InstanceData(scores=cls_score, bboxes=bbox_pred)
        # assigner and sampler
        assign_result = self.assigner.assign(
            pred_instances=pred_instances,
            gt_instances=gt_instances,
            img_meta=img_meta)

        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        pos_inds = torch.nonzero(
            assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(
            assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1
        pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds.long(), :]

        # label targets
        labels = gt_bboxes.new_full((num_bboxes, self.num_classes + 1),
                                    0,
                                    dtype=torch.long)
        gt_labels_exp = torch.cat(
            [torch.ones((gt_labels.size(0), 1), device=gt_labels.device, dtype=gt_labels.dtype), gt_labels], dim=1)
        labels[pos_inds] = gt_labels_exp[pos_assigned_gt_inds].long()
        label_obj_weights = gt_bboxes.new_ones(num_bboxes)
        label_obj_weights[pos_inds] = 2.0
        label_cls_weights = gt_bboxes.new_full((num_bboxes, ), 0, dtype=torch.float)
        label_cls_weights[pos_inds] = 2.0

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred, dtype=gt_bboxes.dtype)
        bbox_weights = torch.zeros_like(bbox_pred, dtype=gt_bboxes.dtype)
        bbox_weights[pos_inds] = 1.0

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        pos_gt_bboxes_normalized = pos_gt_bboxes / factor
        pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)
        bbox_targets[pos_inds] = pos_gt_bboxes_targets
        return (labels, label_obj_weights, label_cls_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds, pos_assigned_gt_inds)

    def get_assigned_feat(self, hidden_states: Tensor,
                  references: List[Tensor],
                  batch_data_samples: SampleList) -> dict:
        batch_gt_instances = []
        batch_gt_instances_pair = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)
            batch_gt_instances_pair.append(data_sample.gt_instances_pair)

        last_dec_feat = hidden_states[-1]
        all_layers_out_classes, all_layers_out_coords = self(hidden_states, references)
        last_dec_classes = all_layers_out_classes[-1]
        last_dec_coords = all_layers_out_coords[-1]

        last_dec_feat, last_dec_feat_pair = torch.chunk(last_dec_feat, chunks=2, dim=0)
        last_dec_classes, last_dec_classes_pair = torch.chunk(last_dec_classes, chunks=2, dim=0)
        last_dec_coords, last_dec_coords_pair = torch.chunk(last_dec_coords, chunks=2, dim=0)

        pred_inputs = (last_dec_classes, last_dec_coords, batch_gt_instances, batch_img_metas)
        pred_inputs_pair = (last_dec_classes_pair, last_dec_coords_pair, batch_gt_instances_pair, batch_img_metas)

        assigned_feats = self.get_assign_by_feat_single(*pred_inputs)
        (labels, label_obj_weights, label_cls_weights, bbox_targets, bbox_weights,
         pos_inds_list, neg_inds_list, pos_gt_inds_list) = assigned_feats

        assigned_feats_pair = self.get_assign_by_feat_single(*pred_inputs_pair)
        (labels_pair, label_obj_weights_pair, label_cls_weights_pair, bbox_targets_pair, bbox_weights_pair,
         pos_inds_list_pair, neg_inds_list_pair, pos_gt_inds_list_pair) = assigned_feats_pair

        assigned_feats = []
        assigned_feats_pair = []
        assigned_attn_mask = []

        gt_bboxes = [x.bboxes for x in batch_gt_instances]
        gt_bboxes_pair = [x.bboxes for x in batch_gt_instances_pair]
        for i, (dec_feat, dec_feat_pair, bbox, bbox_pair) in enumerate(zip(last_dec_feat, last_dec_feat_pair, gt_bboxes, gt_bboxes_pair)):
            gt_inds = torch.scatter(input=torch.zeros_like(pos_gt_inds_list[i]), dim=0,
                                    index=pos_gt_inds_list[i], src=torch.arange(start=0, end=len(bbox), step=1, device=dec_feat.device))
            gt_inds_pair = torch.scatter(input=torch.zeros_like(pos_gt_inds_list_pair[i]), dim=0,
                                    index=pos_gt_inds_list_pair[i], src=torch.arange(start=0, end=len(bbox_pair), step=1, device=dec_feat.device))
            attn_mask = torch.cat([torch.zeros_like(pos_inds_list[i]), torch.ones_like(pos_inds_list_pair[i])],
                                  dim=0).unsqueeze(0).repeat(len(pos_inds_list[i]), 1)
            attn_mask_pair = torch.cat([torch.ones_like(pos_inds_list[i]), torch.zeros_like(pos_inds_list_pair[i])],
                                  dim=0).unsqueeze(0).repeat(len(pos_inds_list_pair[i]), 1)


            det_feat_assigned = dec_feat[pos_inds_list[i]][gt_inds]
            det_feat_assigned_pair = dec_feat_pair[pos_inds_list_pair[i]][gt_inds_pair]
            attn_mask_assigned = torch.cat([attn_mask, attn_mask_pair], dim=0).float()
            assigned_feats.append(det_feat_assigned)
            assigned_feats_pair.append(det_feat_assigned_pair)
            assigned_attn_mask.append(attn_mask_assigned)

        return {'assigned_feats': assigned_feats, 'assigned_feats_pair': assigned_feats_pair, 'assigned_attn_mask': assigned_attn_mask,
                'loss_dicts': ((last_dec_classes, last_dec_classes_pair), (last_dec_coords, last_dec_coords_pair), (labels, labels_pair),
                               (label_obj_weights, label_obj_weights_pair), (label_cls_weights, label_cls_weights_pair),
                               (bbox_targets, bbox_targets_pair), (bbox_weights, bbox_weights_pair))}

    def get_assign_by_feat_single(self, cls_scores: Tensor, bbox_preds: Tensor,
                            batch_gt_instances: InstanceList,
                            batch_img_metas: List[dict]) -> Tuple[Tensor]:
        num_imgs = cls_scores.size(0)
        num_queries = cls_scores.size(1)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]

        assigned_results = self.get_targets_widx(cls_scores_list, bbox_preds_list,
                                           batch_gt_instances, batch_img_metas)
        (labels_list, label_obj_weights_list, label_cls_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg, pos_inds_list, neg_inds_list, pos_gt_inds_list) = assigned_results

        #print(len(labels_list), labels_list[0].size())
        labels = torch.cat(labels_list, 0).reshape(num_imgs, num_queries, -1)
        label_obj_weights = torch.cat(label_obj_weights_list, 0).reshape(num_imgs, num_queries, 1)
        label_cls_weights = torch.cat(label_cls_weights_list, 0).reshape(num_imgs, num_queries, 1)
        bbox_targets = torch.cat(bbox_targets_list, 0).reshape(num_imgs, num_queries, -1)
        bbox_weights = torch.cat(bbox_weights_list, 0).reshape(num_imgs, num_queries, 4)

        return (labels, label_obj_weights, label_cls_weights, bbox_targets, bbox_weights,
                pos_inds_list, neg_inds_list, pos_gt_inds_list)

    def get_predicted_feat(self, hidden_states: Tensor,
                  references: List[Tensor],
                  batch_data_samples: SampleList, conf_threshold=0.2) -> list:
        batch_gt_instances = []
        batch_gt_instances_pair = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)
            batch_gt_instances_pair.append(data_sample.gt_instances_pair)

        last_dec_feat = hidden_states[-1]
        all_layers_out_classes, all_layers_out_coords = self(hidden_states, references)
        last_dec_classes = all_layers_out_classes[-1]
        last_dec_coords = all_layers_out_coords[-1]

        last_dec_feat, last_dec_feat_pair = torch.chunk(last_dec_feat, chunks=2, dim=0)
        last_dec_classes, last_dec_classes_pair = torch.chunk(last_dec_classes, chunks=2, dim=0)
        last_dec_coords, last_dec_coords_pair = torch.chunk(last_dec_coords, chunks=2, dim=0)

        feat_list = []
        feat_list_pair = []
        classes_list = []
        classes_list_pair = []
        coords_list = []
        coords_list_pair = []

        for (feat, feat_pair, classes, classes_pair,
             coords, coords_pair) in zip(last_dec_feat, last_dec_feat_pair,
             last_dec_classes, last_dec_classes_pair, last_dec_coords, last_dec_coords_pair):

            pos_idx = classes[:, 0] >= conf_threshold
            pos_idx_pair = classes_pair[:, 0] >= conf_threshold

            feat_list.append(feat[pos_idx])
            feat_list_pair.append(feat_pair[pos_idx_pair])
            classes_list.append(classes[pos_idx])
            classes_list_pair.append(classes_pair[pos_idx_pair])
            coords_list.append(coords[pos_idx])
            coords_list_pair.append(coords_pair[pos_idx_pair])

        return [dict(feat=feat_list[i], feat_pair=feat_list_pair[i], classes=classes_list[i],
                           classes_pair=classes_list_pair[i], coords=coords_list[i], coords_pair=coords_list_pair[i]) for i in range(len(batch_data_samples))]

    def get_predicted_feat_wo_pair(self, hidden_states: Tensor,
                  references: List[Tensor],
                  batch_data_samples: SampleList, conf_threshold=0.2) -> list:
        last_dec_feat = hidden_states[-1]
        all_layers_out_classes, all_layers_out_coords = self(hidden_states, references)
        last_dec_classes = all_layers_out_classes[-1]
        last_dec_coords = all_layers_out_coords[-1]

        feat_list = []
        classes_list = []
        coords_list = []

        for (feat, classes, coords) in zip(last_dec_feat, last_dec_classes, last_dec_coords):
            pos_idx = classes[:, 0] >= conf_threshold
            feat_list.append(feat[pos_idx])
            classes_list.append(classes[pos_idx])
            coords_list.append(coords[pos_idx])

        return [dict(feat=feat_list[i], classes=classes_list[i], coords=coords_list[i]) for i in range(len(batch_data_samples))]
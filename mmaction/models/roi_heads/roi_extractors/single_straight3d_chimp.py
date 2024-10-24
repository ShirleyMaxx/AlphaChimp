# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SingleRoIExtractor3DChimp(nn.Module):
    """Extract RoI features from a single level feature map.

    Args:
        roi_layer_type (str): Specify the RoI layer type.
            Defaults to ``RoIAlign``.
        featmap_stride (int): Strides of input feature maps. Defaults to 16.
        output_size (int or tuple): Size or (Height, Width). Defaults to 16.
        sampling_ratio (int): number of inputs samples to take for each
            output sample. 0 to take samples densely for current models.
            Defaults to 0.
        pool_mode (str): pooling mode in each bin. Choices are ``avg`` or
            ``max``. Defaults to ``avg``.
        aligned (bool): if False, use the legacy implementation in
            MMDetection. If True, align the results more perfectly.
            Defaults to True.
        with_temporal_pool (bool): if True, avgpool the temporal dim.
            Defaults to True.
        with_global (bool): if True, concatenate the RoI feature with global
            feature. Defaults to False.

    Note that sampling_ratio, pool_mode, aligned only apply when roi_layer_type
    is set as RoIAlign.
    """

    def __init__(self,
                 roi_layer_type: str = 'RoIAlign',
                 featmap_stride: int = 16,
                 output_size: int = 16,
                 sampling_ratio: int = 0,
                 max_chimp: int = 5,
                 pool_mode: str = 'avg',
                 aligned: bool = True,
                 with_temporal_pool: bool = False,
                 temporal_pool_mode: str = 'avg',
                 with_global: bool = False) -> None:
        super().__init__()
        self.roi_layer_type = roi_layer_type
        assert self.roi_layer_type in ['RoIPool', 'RoIAlign']
        self.featmap_stride = featmap_stride
        self.spatial_scale = 1. / self.featmap_stride

        self.output_size = output_size
        self.sampling_ratio = sampling_ratio
        self.max_chimp = max_chimp
        self.pool_mode = pool_mode
        self.aligned = aligned

        self.with_temporal_pool = with_temporal_pool
        self.temporal_pool_mode = temporal_pool_mode

        self.with_global = with_global

        try:
            from mmcv.ops import RoIAlign, RoIPool
        except (ImportError, ModuleNotFoundError):
            raise ImportError('Failed to import `RoIAlign` and `RoIPool` from '
                              '`mmcv.ops`. The two modules will be used in '
                              '`SingleRoIExtractor3D`! ')

        if self.roi_layer_type == 'RoIPool':
            self.roi_layer = RoIPool(self.output_size, self.spatial_scale)
        else:
            self.roi_layer = RoIAlign(
                self.output_size,
                self.spatial_scale,
                sampling_ratio=self.sampling_ratio,
                pool_mode=self.pool_mode,
                aligned=self.aligned)
        self.global_pool = nn.AdaptiveAvgPool2d(self.output_size)

    def forward(self, feat: Union[Tensor, Tuple[Tensor]],
                rois: list,
                entity_mask: Tensor) -> tuple:
        """Forward function for extract roi features.

        Args:
            feat (Tensor or Tuple[Tensor]): The image features extracted by
                the upstream network. The shape of feat is N, C, T, H, W.
            rois (List): Input RoIs, shape (n), (t), (k_t, 5).
            entity_mask (Tensor): shape N, 1, M, T

        Returns:
            tuple: A tuple of roi features and global features.

                - roi_feats (Tensor): Extracted bbox RoI features, shape N, T, C, H, W.
                - feat (Tensor): Global features of the video clip.
        """
        if not isinstance(feat, tuple):
            feat = (feat, )

        if len(feat) >= 2:
            maxT = max([x.shape[2] for x in feat])
            max_shape = (maxT, ) + feat[0].shape[3:]
            # resize each feat to the largest shape (w. nearest)
            feat = [F.interpolate(x, max_shape).contiguous() for x in feat]

        if self.with_temporal_pool:
            if self.temporal_pool_mode == 'avg':
                feat = [torch.mean(x, 2, keepdim=True) for x in feat]
            elif self.temporal_pool_mode == 'max':
                feat = [torch.max(x, 2, keepdim=True)[0] for x in feat]
            else:
                raise NotImplementedError

        feat = torch.cat(feat, axis=1).contiguous()

        roi_feats = []
        roi_batch_feats = []
        for t in range(feat.size(2)):
            for n in range(feat.size(0)):
                frame_feat = feat[n, :, t].unsqueeze(0).contiguous()
                roi_feat = self.roi_layer(frame_feat, rois[n])
                roi_feat = torch.cat([roi_feat, 
            torch.zeros(self.max_chimp-rois[n].size(0), feat.size(1), self.output_size, self.output_size, device=feat.device)]).unsqueeze(0).unsqueeze(0)
                roi_batch_feats.append(roi_feat)

            roi_feats.append(torch.cat(roi_batch_feats, dim=0))
            roi_batch_feats.clear()

        roi_feats = torch.cat(roi_feats, dim=1)
        return roi_feats, feat


class SingleRoIExtractor3DChimpAVG(nn.Module):
    """Extract RoI features from a single level feature map.

    Args:
        roi_layer_type (str): Specify the RoI layer type.
            Defaults to ``RoIAlign``.
        featmap_stride (int): Strides of input feature maps. Defaults to 16.
        output_size (int or tuple): Size or (Height, Width). Defaults to 16.
        sampling_ratio (int): number of inputs samples to take for each
            output sample. 0 to take samples densely for current models.
            Defaults to 0.
        pool_mode (str): pooling mode in each bin. Choices are ``avg`` or
            ``max``. Defaults to ``avg``.
        aligned (bool): if False, use the legacy implementation in
            MMDetection. If True, align the results more perfectly.
            Defaults to True.
        with_temporal_pool (bool): if True, avgpool the temporal dim.
            Defaults to True.
        with_global (bool): if True, concatenate the RoI feature with global
            feature. Defaults to False.

    Note that sampling_ratio, pool_mode, aligned only apply when roi_layer_type
    is set as RoIAlign.
    """

    def __init__(self,
                 roi_layer_type: str = 'RoIAlign',
                 featmap_stride: int = 16,
                 output_size: int = 16,
                 sampling_ratio: int = 0,
                 max_chimp: int = 24,
                 pool_mode: str = 'avg',
                 aligned: bool = True,
                 with_temporal_pool: bool = False,
                 temporal_pool_mode: str = 'avg',
                 with_global: bool = False) -> None:
        super().__init__()
        self.roi_layer_type = roi_layer_type
        assert self.roi_layer_type in ['RoIPool', 'RoIAlign']
        self.featmap_stride = featmap_stride
        self.spatial_scale = 1. / self.featmap_stride

        self.output_size = output_size
        self.sampling_ratio = sampling_ratio
        self.max_chimp = max_chimp
        self.pool_mode = pool_mode
        self.aligned = aligned

        self.with_temporal_pool = with_temporal_pool
        self.temporal_pool_mode = temporal_pool_mode

        self.with_global = with_global

        try:
            from mmcv.ops import RoIAlign, RoIPool
        except (ImportError, ModuleNotFoundError):
            raise ImportError('Failed to import `RoIAlign` and `RoIPool` from '
                              '`mmcv.ops`. The two modules will be used in '
                              '`SingleRoIExtractor3D`! ')

        if self.roi_layer_type == 'RoIPool':
            self.roi_layer = RoIPool(self.output_size, self.spatial_scale)
        else:
            self.roi_layer = RoIAlign(
                self.output_size,
                self.spatial_scale,
                sampling_ratio=self.sampling_ratio,
                pool_mode=self.pool_mode,
                aligned=self.aligned)
        self.global_pool = nn.AdaptiveAvgPool2d(self.output_size)

    def forward(self, feat: Union[Tensor, Tuple[Tensor]],
                rois: Tensor) -> tuple:
        """Forward function for extract roi features.

        Args:
            feat (Tensor or Tuple[Tensor]): The image features extracted by
                the upstream network. The shape of feat is N, C, T, H, W.
            rois (Tensor): Input RoIs, shape (k, 5).

        Returns:
            tuple: A tuple of roi features and global features.

                - roi_feats (Tensor): Extracted bbox RoI features.
                - feat (Tensor): Global features of the video clip.
        """
        batch_size = feat.shape[0]
        if not isinstance(feat, tuple):
            feat = (feat, )

        if len(feat) >= 2:
            maxT = max([x.shape[2] for x in feat])
            max_shape = (maxT, ) + feat[0].shape[3:]
            # resize each feat to the largest shape (w. nearest)
            feat = [F.interpolate(x, max_shape).contiguous() for x in feat]

        if self.with_temporal_pool:
            if self.temporal_pool_mode == 'avg':
                feat = [torch.mean(x, 2, keepdim=True) for x in feat]
            elif self.temporal_pool_mode == 'max':
                feat = [torch.max(x, 2, keepdim=True)[0] for x in feat]
            else:
                raise NotImplementedError

        feat = torch.cat(feat, axis=1).contiguous()

        roi_feats = []
        for t in range(feat.size(2)):
            frame_feat = feat[:, :, t].contiguous()
            roi_feat = self.roi_layer(frame_feat, rois)
            if self.with_global:
                global_feat = self.global_pool(frame_feat.contiguous())
                inds = rois[:, 0].type(torch.int64)
                global_feat = global_feat[inds]
                roi_feat = torch.cat([roi_feat, global_feat], dim=1)
                roi_feat = roi_feat.contiguous()
            roi_feats.append(roi_feat)
        roi_feats = torch.stack(roi_feats, dim=1).view((feat.size(2),batch_size,self.max_chimp)+roi_feat.size()[-3:])
        return roi_feats, feat


class SingleRoIExtractor3DChimpAVGV3(nn.Module):
    """Extract RoI features from a single level feature map.

    Args:
        roi_layer_type (str): Specify the RoI layer type.
            Defaults to ``RoIAlign``.
        featmap_stride (int): Strides of input feature maps. Defaults to 16.
        output_size (int or tuple): Size or (Height, Width). Defaults to 16.
        sampling_ratio (int): number of inputs samples to take for each
            output sample. 0 to take samples densely for current models.
            Defaults to 0.
        pool_mode (str): pooling mode in each bin. Choices are ``avg`` or
            ``max``. Defaults to ``avg``.
        aligned (bool): if False, use the legacy implementation in
            MMDetection. If True, align the results more perfectly.
            Defaults to True.
        with_temporal_pool (bool): if True, avgpool the temporal dim.
            Defaults to True.
        with_global (bool): if True, concatenate the RoI feature with global
            feature. Defaults to False.

    Note that sampling_ratio, pool_mode, aligned only apply when roi_layer_type
    is set as RoIAlign.
    """

    def __init__(self,
                 roi_layer_type: str = 'RoIAlign',
                 featmap_stride: int = 16,
                 output_size: int = 16,
                 sampling_ratio: int = 0,
                 max_chimp: int = 24,
                 pool_mode: str = 'avg',
                 aligned: bool = True,
                 with_temporal_pool: bool = False,
                 temporal_pool_mode: str = 'avg',
                 with_global: bool = False) -> None:
        super().__init__()
        self.roi_layer_type = roi_layer_type
        assert self.roi_layer_type in ['RoIPool', 'RoIAlign']
        self.featmap_stride = featmap_stride
        self.spatial_scale = 1. / self.featmap_stride

        self.output_size = output_size
        self.sampling_ratio = sampling_ratio
        self.max_chimp = max_chimp
        self.pool_mode = pool_mode
        self.aligned = aligned

        self.with_temporal_pool = with_temporal_pool
        self.temporal_pool_mode = temporal_pool_mode

        self.with_global = with_global

        try:
            from mmcv.ops import RoIAlign, RoIPool
        except (ImportError, ModuleNotFoundError):
            raise ImportError('Failed to import `RoIAlign` and `RoIPool` from '
                              '`mmcv.ops`. The two modules will be used in '
                              '`SingleRoIExtractor3D`! ')

        if self.roi_layer_type == 'RoIPool':
            self.roi_layer = RoIPool(self.output_size, self.spatial_scale)
        else:
            self.roi_layer = RoIAlign(
                self.output_size,
                self.spatial_scale,
                sampling_ratio=self.sampling_ratio,
                pool_mode=self.pool_mode,
                aligned=self.aligned)
        self.global_pool = nn.AdaptiveAvgPool2d(self.output_size)

    def forward(self, feat: Tensor,
                rois: Tensor) -> tuple:
        """Forward function for extract roi features.

        Args:
            feat (Tensor or Tuple[Tensor]): The image features extracted by
                the upstream network. The shape of feat is N, C, T, H, W.
            rois (Tensor): Input RoIs, shape (k, 5).

        Returns:
            tuple: A tuple of roi features and global features.

                - roi_feats (Tensor): Extracted bbox RoI features.
                - feat (Tensor): Global features of the video clip.
        """
        batch_size = feat.size(0)

        roi_feats = []
        for t in range(feat.size(2)):
            frame_feat = feat[:, :, t].contiguous()
            roi_feat = self.roi_layer(frame_feat, rois)
            # print(roi_feat.size())
            if self.with_global:
                global_feat = self.global_pool(frame_feat.contiguous())
                inds = rois[:, 0].type(torch.int64)
                global_feat = global_feat[inds]
                roi_feat = torch.cat([roi_feat, global_feat], dim=1)
                roi_feat = roi_feat.contiguous()
            roi_feat = roi_feat.reshape((batch_size, -1)+roi_feat.size()[-3:])
            roi_feats.append(roi_feat)
        roi_feats = torch.stack(roi_feats, dim=1)
        return roi_feats, feat

# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.model.weight_init import normal_init
from torch import Tensor

from mmaction.registry import MODELS
from mmaction.utils import ConfigType
from typing import Dict, Optional, Tuple, Union
from .base import BaseHead

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

from mmaction.evaluation import top_k_accuracy
from mmaction.utils import ForwardResults, SampleList


@MODELS.register_module()
class I3DHeadChimp(BaseHead):
    """Classification head for I3D.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict or ConfigDict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 loss_cls: ConfigType = dict(type='CrossEntropyLoss'),
                 spatial_type: str = 'avg',
                 dropout_ratio: float = 0.5,
                 init_std: float = 0.01,
                 **kwargs) -> None:
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)

        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool3d` to adaptively match the in_channels.
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool = None

    def init_weights(self) -> None:
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std)

    def loss_by_feat(self, cls_scores: torch.Tensor,
                     data_samples: SampleList) -> Dict:
        """Calculate the loss based on the features extracted by the head.

        Args:
            cls_scores (torch.Tensor): Classification prediction results of
                all class, has shape (batch_size, num_classes).
            data_samples (list[:obj:`ActionDataSample`]): The batch
                data samples.

        Returns:
            dict: A dictionary of loss components.
        """
        labels = [x.gt_instances.labels for x in data_samples]
        losses = dict()

        loss_cls = self.loss_cls(cls_scores, labels)
        # loss_cls may be dictionary or single tensor
        if isinstance(loss_cls, dict):
            losses.update(loss_cls)
        else:
            losses['loss_cls'] = loss_cls
        return losses

    def predict_by_feat(self, cls_scores: torch.Tensor,
                        data_samples: SampleList) -> SampleList:
        """Transform a batch of output features extracted from the head into
        prediction results.

        Args:
            cls_scores (torch.Tensor): Classification scores, has a shape
                (B*num_segs, num_classes)
            data_samples (list[:obj:`ActionDataSample`]): The
                annotation data of every samples. It usually includes
                information such as `gt_label`.

        Returns:
            List[:obj:`ActionDataSample`]: Recognition results wrapped
                by :obj:`ActionDataSample`.
        """
        
        cls_scores = [x.squeeze(0) for x in torch.chunk(cls_scores, len(data_samples), dim=0)]
        for data_sample, score in zip(data_samples, cls_scores):
            num_bboxes = data_sample.gt_instances.bboxes.size(0)
            score = torch.sigmoid(score[:num_bboxes, :])

            data_sample.set_field({'bboxes':data_sample.gt_instances.bboxes, 'scores':score}, 'pred_instances')
            
        return data_samples

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        """Defines the computation performed at every call.

        Args:
            x (Tensor): The input data.

        Returns:
            Tensor: The classification scores for input samples.
        """
        N, R, C, T, D = x.size()
        #print('FFFF', x[0, 0, 0])

        x = x.reshape(N * R, C, T, D, 1)
        # [N, in_channels, 4, 7, 7]
        if self.avg_pool is not None:
            x = self.avg_pool(x)
        # [N, in_channels, 1, 1, 1]
        if self.dropout is not None:
            x = self.dropout(x)
        # [N, in_channels, 1, 1, 1]
        x = x.view(x.shape[0], -1)
        # [N, in_channels]
        cls_score = self.fc_cls(x).reshape(N, R, self.num_classes)
        # [N, num_classes]
        return cls_score

@MODELS.register_module()
class I3DHeadChimpAVG(BaseHead):
    """Classification head for I3D.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict or ConfigDict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 loss_cls: ConfigType = dict(type='CrossEntropyLoss'),
                 spatial_type: str = 'avg',
                 dropout_ratio: float = 0.5,
                 init_std: float = 0.01,
                 **kwargs) -> None:
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)

    def init_weights(self) -> None:
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std)

    def loss_by_feat(self, cls_scores: torch.Tensor,
                     data_samples: SampleList) -> Dict:
        """Calculate the loss based on the features extracted by the head.

        Args:
            cls_scores (torch.Tensor): Classification prediction results of
                all class, has shape (batch_size, num_classes).
            data_samples (list[:obj:`ActionDataSample`]): The batch
                data samples.

        Returns:
            dict: A dictionary of loss components.
        """
        labels = [x.gt_instances.labels for x in data_samples]
        score_list = []
        for data_sample, score in zip(data_samples, cls_scores):
            score = score[data_sample.entity_mask.bboxes, :]
            score_list.append(score)
        
        labels = torch.cat(labels,dim=0).to(labels[0].device)
        scores = torch.cat(score_list,dim=0).to(score_list[0].device)

        losses = dict()

        loss_cls = self.loss_cls(scores, labels)
        # loss_cls may be dictionary or single tensor
        if isinstance(loss_cls, dict):
            losses.update(loss_cls)
        else:
            losses['loss_cls'] = loss_cls
        return losses

    def predict_by_feat(self, cls_scores: torch.Tensor,
                        data_samples: SampleList) -> SampleList:
        """Transform a batch of output features extracted from the head into
        prediction results.

        Args:
            cls_scores (torch.Tensor): Classification scores, has a shape
                (B*num_segs, num_classes)
            data_samples (list[:obj:`ActionDataSample`]): The
                annotation data of every samples. It usually includes
                information such as `gt_label`.

        Returns:
            List[:obj:`ActionDataSample`]: Recognition results wrapped
                by :obj:`ActionDataSample`.
        """
        
        cls_scores = [x.squeeze(0) for x in torch.chunk(cls_scores, len(data_samples), dim=0)]
        for data_sample, score in zip(data_samples, cls_scores):
            num_bboxes = data_sample.gt_instances.bboxes.size(0)
            score = torch.sigmoid(score[data_sample.entity_mask.bboxes, :])

            data_sample.set_field({'bboxes':data_sample.gt_instances.bboxes, 'scores':score}, 'pred_instances')
            
        return data_samples

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        """Defines the computation performed at every call.

        Args:
            x (Tensor): The input data.

        Returns:
            Tensor: The classification scores for input samples.
        """

        N, R, C = x.size()

        x = x.reshape(N * R, C)
        # [N, in_channels, 4, 7, 7]
        if self.dropout is not None:
            x = self.dropout(x)
        # [N, in_channels]
        cls_score = self.fc_cls(x).reshape(N, R, self.num_classes)
        # [N, num_classes]
        return cls_score


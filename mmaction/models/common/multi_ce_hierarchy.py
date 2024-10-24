import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Optional, Tuple, Union
from mmaction.registry import MODELS
from mmengine.model import BaseModule


@MODELS.register_module()
class MultilableCrossEntropyHierarchy(BaseModule):
    """ Criss-Cross Attention Module"""

    def __init__(self, use_sigmoid=True, loss_weight=1.0, num_classes=24, mask_cls=False, init_cfg=None, **kwargs):
        super().__init__(init_cfg=init_cfg)
        self.use_sigmoid = use_sigmoid
        self.loss_weight = loss_weight
        self.num_classes = num_classes
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.weighted_class = torch.tensor([[1., 5., 5., 1., 4., 1.5, 5., 8., 5., 5., 12., 4., 12., 12., 20., 20., 10., 10., 8., 8., 4., 3.5, 30., 7.]],
                                           dtype=torch.float, requires_grad=False)
        self.mask_cls = mask_cls
        self.cls_mask_prob = 0.1

    def forward(self, x: Tensor, target: Union[Tuple[Tensor, Tensor], Tensor], obj_weight=None, cls_weight=None, avg_factor=None, cls_masks=None, **kwargs):
        if isinstance(target, tuple):
            labels, scores = target
        else:
            labels = target

        x = x.float()
        labels = labels.float()
        loss = self.criterion(x, labels)

        if x.size(1) != self.num_classes:
            loss_obj = loss[:, 0]
            loss_cls = loss[:, 1:]
        else:
            loss_obj = 0
            loss_cls = loss

        if obj_weight is not None and cls_weight is not None:
            obj_weight = obj_weight.reshape(loss.size(0), ).float()
            cls_weight = cls_weight.reshape(loss.size(0), 1).float()
            loss_obj = loss_obj * obj_weight
            loss_cls = loss_cls * cls_weight

        #if self.weighted_class.device != loss_cls.device:
        #    self.weighted_class = self.weighted_class.to(loss_cls.device)
        #loss_cls = loss_cls * self.weighted_class / self.weighted_class.mean()

        if cls_masks is not None and self.mask_cls:
            cls_masks = cls_masks.reshape(len(loss_cls), 1)
            cls_masks = 1 - ((torch.rand_like(cls_masks) * (1 - cls_masks)) >= self.cls_mask_prob).float()
            loss_cls = loss_cls * cls_masks

        loss_cls = loss_cls.mean(dim=1)
        if avg_factor is not None:
            loss = (loss_cls + loss_obj).mean() / avg_factor
        else:
            loss = (loss_cls + loss_obj).mean()

        return loss * self.loss_weight
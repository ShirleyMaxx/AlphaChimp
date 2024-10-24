import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Optional, Tuple, Union
from mmaction.registry import MODELS
from mmengine.model import BaseModule


class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"

    def __init__(self, alpha=.25, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1 - alpha])
        self.gamma = gamma

    def forward(self, inputs, targets):
        self.alpha = self.alpha.to(inputs.device)

        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        targets = targets.long()
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at * (1 - pt) ** self.gamma * BCE_loss
        return F_loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)  # 使用交叉熵损失函数计算基础损失
        pt = torch.exp(-ce_loss)  # 计算预测的概率
        focal_loss = (1 - pt) ** self.gamma * ce_loss  # 根据Focal Loss公式计算Focal Loss
        return focal_loss


@MODELS.register_module()
class MultilableCrossEntropy(BaseModule):
    """ Criss-Cross Attention Module"""

    def __init__(self, use_sigmoid=True, loss_weight=1.0, num_classes=24, mask_cls=False, no_obj_mode=False, extra_obj_mode=False, focal=True, init_cfg=None, **kwargs):
        super().__init__(init_cfg=init_cfg)
        self.use_sigmoid = use_sigmoid
        self.loss_weight = loss_weight
        self.num_classes = num_classes
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.focal_loss = FocalLoss()
        self.mask_cls = mask_cls
        self.no_obj_mode = no_obj_mode
        self.extra_obj_mode = extra_obj_mode
        self.cls_mask_prob = 0.1
        self.focal = focal

    def forward(self, x: Tensor, target: Union[Tuple[Tensor, Tensor], Tensor], obj_weight=None, cls_weight=None, avg_factor=None, cls_masks=None, **kwargs):
        if isinstance(target, tuple):
            labels, scores = target
        else:
            labels = target

        x = x.float()
        labels = labels.float()
        #print(x.size(), labels.size())
        if self.extra_obj_mode:
            loss_obj = self.criterion(x[:, 0], labels[:, 0])
            if self.focal:
                loss_cls = self.focal_loss(x[:, 1:], labels[:, 1:])
            else:
                loss_cls = self.criterion(x[:, 1:], labels[:, 1:])
        else:
            if self.focal:
                loss = self.focal_loss(x, labels)
            else:
                loss = self.criterion(x, labels)
            if x.size(1) != self.num_classes:
                loss_obj = loss[:, 0]
                loss_cls = loss[:, 1:]
            else:
                loss_obj = 0
                loss_cls = loss

        if obj_weight is not None and cls_weight is not None:
            if not self.no_obj_mode:
                obj_weight = obj_weight.reshape(loss.size(0), ).float()
                cls_weight = cls_weight.reshape(loss.size(0), 1).float()
            else:
                obj_weight = 0 if not self.extra_obj_mode else 1
                cls_weight = 1
            loss_obj = loss_obj * obj_weight
            loss_cls = loss_cls * cls_weight

        if cls_masks is not None and self.mask_cls:
            cls_masks = cls_masks.reshape(len(loss_cls), 1)
            cls_masks = 1 - ((torch.rand_like(cls_masks) * (1 - cls_masks)) >= self.cls_mask_prob).float()
            loss_cls = loss_cls * cls_masks

        loss_cls = loss_cls.mean(dim=1)
        if avg_factor is not None:
            loss = (loss_cls + loss_obj).sum() * avg_factor
        else:
            loss = (loss_cls + loss_obj).mean()

        return loss * self.loss_weight
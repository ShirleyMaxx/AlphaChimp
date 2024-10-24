import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Optional, Tuple, Union
from mmaction.registry import MODELS
from mmengine.model import BaseModule


@MODELS.register_module()
class TrackingBCELoss(BaseModule):
    """ Criss-Cross Attention Module"""

    def __init__(self, loss_weight=1.0, init_cfg=None, **kwargs):
        super().__init__(init_cfg=init_cfg)
        self.loss_weight = loss_weight
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, preds: list, batch_samples: list):
        num_batch = len(batch_samples)
        assert len(preds) == num_batch, 'Size must match.'

        loss = 0.0
        for i in range(num_batch):
            # print(sample.matrix.bboxes.size())
            pred = preds[i]
            target = batch_samples[i].matrix.bboxes.float()
            loss += self.criterion(pred, target).sum()

        loss = loss / num_batch

        return loss * self.loss_weight
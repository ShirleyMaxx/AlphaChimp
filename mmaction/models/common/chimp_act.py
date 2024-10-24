import inspect
import warnings
from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
from mmengine.model import BaseModel, merge_dict

from mmaction.registry import MODELS
from mmaction.utils import (ConfigType, ForwardResults, OptConfigType,
                            OptSampleList, SampleList)


@MODELS.register_module()
class ChimpAct(BaseModel, metaclass=ABCMeta):
    def __init__(self,
                 backbone: ConfigType,
                 cls_head: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None) -> None:
        if data_preprocessor is None:
            # This preprocessor will only stack batch data samples.
            data_preprocessor = dict(type='ActionDataPreprocessor')

        super(ChimpAct, self).__init__(data_preprocessor=data_preprocessor)
        
        self.backbone = MODELS.build(backbone)
        self.cls_head = MODELS.build(cls_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self,inputs):
        feat = self.backbone(**inputs)
        return feat
    
    def loss(self,
            inputs: dict,
            data_samples: dict):
        feats = self.extract_feat(inputs)
        loss_cls = self.cls_head.loss(feats, data_samples)
        return loss_cls
    
    def predict(self,
                inputs: dict,
                data_samples: dict):
        feats = self.extract_feat(inputs)
        predictions = self.cls_head.predict(feats, data_samples)
        return predictions


    def forward(self,
                inputs: dict,
                data_samples: dict,
                mode: str = 'loss'):
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        else:
            raise NotImplementedError
        



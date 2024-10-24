import inspect
import warnings
from abc import ABCMeta, abstractmethod

import os
import torch
import torch.nn as nn
from torch import Tensor
from mmengine.model import BaseModel, merge_dict

from mmaction.registry import MODELS
from mmaction.utils import (ConfigType, ForwardResults, OptConfigType,
                            OptSampleList, SampleList)

@MODELS.register_module()
class AlphaChimp(BaseModel, metaclass=ABCMeta):
    def __init__(self,
                 model_cfg: ConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptConfigType = None) -> None:
        super(AlphaChimp, self).__init__(data_preprocessor=data_preprocessor,init_cfg=init_cfg)

        self.model = MODELS.build(model_cfg)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def loss(self, imgs: Tensor,
             data_samples: dict, **kargs) -> dict:
        losses = self.model.loss(imgs, data_samples)
        return losses

    def predict(self, imgs: Tensor,
             data_samples: dict, **kargs) -> dict:
        #self.model.eval()
        predictions = self.model.predict(imgs, data_samples)
        return predictions

    def forward(self, inputs: dict,
                data_samples: dict,
                mode: str = 'loss'):
        if mode == 'loss':
            return self.loss(**inputs, data_samples=data_samples)
        elif mode == 'predict':
            return self.predict(**inputs, data_samples=data_samples)
        else:
            raise NotImplementedError


@MODELS.register_module()
class ChimpAnalyserWithTracking(BaseModel, metaclass=ABCMeta):
    def __init__(self,
                 ana_model_cfg: ConfigType = None,
                 track_head_cfg: ConfigType = None,
                 pred_conf_threshold: float = 0.2,

                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None) -> None:
        super(ChimpAnalyserWithTracking, self).__init__(data_preprocessor=data_preprocessor)

        self.ana_model = MODELS.build(ana_model_cfg)
        self.track_head = MODELS.build(track_head_cfg)

        self.pred_conf_threshold = pred_conf_threshold
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def loss(self, imgs: Tensor,
             imgs_pair: Tensor,
             data_samples: dict) -> dict:
        if self.ana_model.training:
            self.ana_model.eval()
        with torch.no_grad():
            assigned_dict = self.ana_model.model.get_assigned_feat(imgs, imgs_pair, data_samples)
        losses = self.track_head.loss(**assigned_dict, data_samples=data_samples)
        return losses

    def predict(self, imgs: Tensor,
             imgs_pair: Tensor,
             data_samples: list, **kargs) -> dict:
        predict_feats = self.ana_model.model.get_predicted_feat(imgs, imgs_pair, data_samples, conf_threshold=self.pred_conf_threshold)
        predictions = self.track_head.predict(predict_feats, data_samples=data_samples)
        return predictions

    def predict_feat_only(self, imgs: Tensor, data_samples: list):
        predict_feats = self.ana_model.model.get_predicted_feat_wo_pair(imgs, data_samples, conf_threshold=self.pred_conf_threshold)
        return predict_feats
    
    def predict_attn_only(self, feat_dict: list, data_samples: list):
        predictions = self.track_head.predict(feat_dict, data_samples=data_samples)
        return predictions

    def forward(self, inputs: dict,
                data_samples: dict,
                mode: str = 'loss'):
        if mode == 'loss':
            return self.loss(**inputs, data_samples=data_samples)
        elif mode == 'predict':
            return self.predict(**inputs, data_samples=data_samples)
        else:
            raise NotImplementedError




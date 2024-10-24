import inspect
import warnings
from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
from torch import Tensor
from mmengine.model import BaseModel, merge_dict

from mmaction.registry import MODELS
from mmaction.utils import (ConfigType, ForwardResults, OptConfigType,
                            OptSampleList, SampleList)


@MODELS.register_module()
class ChimpDet(BaseModel, metaclass=ABCMeta):
    def __init__(self,
                 preprocess_model: ConfigType,
                 det_model: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None) -> None:
        if data_preprocessor is None:
            # This preprocessor will only stack batch data samples.
            data_preprocessor = dict(type='ActionDataPreprocessor')

        super(ChimpDet, self).__init__(data_preprocessor=data_preprocessor)

        self.preprocess_model = MODELS.build(preprocess_model)
        self.det_model = MODELS.build(det_model)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def fusion_temporal_feat(self, imgs: Tensor):
        '''This function merges temporal features
        with preprocess model.

        :param imgs: input images, of shape (N, C, T, H, W)
        :param kargs: key-word arguments.
        :return:
        '''
        feats = self.preprocess_model(imgs)

        torch.cuda.empty_cache()
        return feats

    def loss(self, inputs: Tensor,
             data_samples: dict):
        # print(data_samples)
        feats = self.fusion_temporal_feat(inputs)
        losses = self.det_model.loss(feats, data_samples)
        return losses

    def predict(self,
                inputs: Tensor,
                data_samples: dict):
        feats = self.fusion_temporal_feat(inputs)
        predictions = self.det_model.predict(feats, data_samples)
        return predictions

    def forward(self, inputs: Tensor,
                data_samples: dict,
                mode: str = 'loss'):
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        else:
            raise NotImplementedError




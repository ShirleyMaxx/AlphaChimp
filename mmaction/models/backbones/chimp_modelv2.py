import warnings
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union
from torch import Tensor

import torch
import torch.nn as nn
import numpy as np
from mmcv.cnn import ConvModule
from mmaction.registry import MODELS
from mmengine.logging import MMLogger, print_log
from mmengine.model import BaseModule
from mmengine.model.weight_init import kaiming_init
from mmengine.runner.checkpoint import _load_checkpoint, load_checkpoint

from mmdet.structures.bbox import bbox2roi

from mmaction.registry import MODELS
from .resnet3d_slowfast import ResNet3dSlowFast
from .stgcn import STGCN
from ..roi_heads.roi_extractors.single_straight3d_chimp import SingleRoIExtractor3DChimp, SingleRoIExtractor3DChimpAVG
from ..common.criss_cross_attn import CrissCrossAttention


@MODELS.register_module()
class ChimpModelV2(BaseModule):
    def __init__(self,
                 num_frame: int = 80,
                 num_keypoint: int = 16,
                 max_chimp: int = 7,
                 d_model: int = 256,
                 out_channels = 32,
                 img_model_cfg: Dict = None,
                 roi_extractor_cfg: Dict = None,
                 init_cfg: Optional[Union[Dict, List[Dict]]] = None) -> None:
        '''Backbone for chimp action feature extraction.

        Args:
            num_frame: Numbers of frames input.
            num_keypoints: Numbers of skeleton keypoints.
            max_chimp: Maximum chimp processed, related to roi & spatio-temporal attention.
                be aware that increasing this may cause performance problems.
            d_model: Indicates the feature dim inside model. This depends on the backbones and
                forward pipeline, changing this may cause runtime error.
            
        
        '''
        super().__init__(init_cfg=init_cfg)

        if img_model_cfg is not None:
            self.img_model = MODELS.build(img_model_cfg)
        else:
            self.img_model = ResNet3dSlowFast()

        if roi_extractor_cfg is not None:
            self.roi_extractor = SingleRoIExtractor3DChimpAVG(**roi_extractor_cfg)
        else:
            self.roi_extractor = SingleRoIExtractor3DChimpAVG(max_chimp=max_chimp)
        
        self.num_frame = num_frame
        self.num_keypoint = num_keypoint
        self.max_chimp = max_chimp
        self.d_model = d_model
        self.out_channels = out_channels

        self.roi_size = self.roi_extractor.output_size

        self.slow_fast_fusion_t = nn.Sequential(
            nn.Conv2d(self.num_frame//self.img_model.speed_ratio, self.num_frame, 3, 1, 1), nn.BatchNorm2d(self.num_frame), nn.LeakyReLU(inplace=True)
        )
        self.slow_fast_fusion_c = nn.Sequential(
            nn.Conv2d(self.d_model*8, self.d_model, 3, 1, 1), nn.BatchNorm2d(self.d_model), nn.LeakyReLU(inplace=True)
        )

        self.roi_downsample = nn.Sequential(
            nn.Conv2d(self.d_model*2, self.d_model, 3, 1, 1), nn.BatchNorm2d(self.d_model), nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.d_model, self.d_model, 3, 2, 1), nn.BatchNorm2d(self.d_model), nn.LeakyReLU(inplace=True),
            nn.AvgPool2d(4, 4),
            nn.LeakyReLU(inplace=True),
        )

        self.final_d = self.d_model * self.roi_size//8 * self.roi_size//8
        self.cc_attn = CrissCrossAttention(self.final_d)

        self.feature_upsample = nn.Sequential(
            nn.Conv2d(self.final_d, self.out_channels, 1, 1, 0), nn.BatchNorm2d(self.out_channels), nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 1, 1, 0)
        )
        self.temperol_downsample = nn.Sequential(nn.Conv2d(self.num_frame, 1, 1, 1, 0))
            

    def forward(self,
                imgs: Tensor,
                proposals: Tensor,
                entity_mask: Tensor = None):
        '''Forward function. Pipeline: img-hmap feature extraction => roi & ske feature extration => spatio-temporal attention.

        Args:
            img: tensor, shape N, C, T, H, W. RGB.
            rois: list, List[Tensor], List[N]. List of rois in eatch batch, no need to pad.
            roi_mask: tensor, shape N, 1, M, T.
        '''
        #print('PROPOSALS', proposals)
        #print('ENTITY MASK', entity_mask)

        # Get batch size.
        batch_size = imgs.size(0)

        # Image features, out N, C, T, H, W
        img_feat = self.img_model(imgs)

        
        # Cat features.
        img_feat_slow = img_feat[0].transpose(1, 2)

        img_feat_slow = self.slow_fast_fusion_c(img_feat_slow.reshape((-1, ) + img_feat_slow.size()[2:]))
        img_feat_slow = img_feat_slow.reshape((batch_size, -1) + img_feat_slow.size()[1:]).transpose(1, 2)

        img_feat_slow = self.slow_fast_fusion_t(img_feat_slow.reshape((-1, ) + img_feat_slow.size()[2:]))
        global_feat = torch.cat((img_feat_slow.reshape((batch_size, -1) + img_feat_slow.size()[1:]), img_feat[1]), dim=1)
        # Extract roi feature.
        # TODO optimize complexity.

        rois = bbox2roi(proposals) 
        roi_feat, __ = self.roi_extractor(global_feat, rois)
        roi_feat = self.roi_downsample(roi_feat.reshape((-1, ) + roi_feat.size()[3:]))
        roi_feat = roi_feat.reshape(self.num_frame, batch_size, self.max_chimp, -1).permute(1, 3, 2, 0)

        # Compute spatio-temporal criss-cross attention, out N, T, R, D.
        # We do twice criss cross attention to fully crossing the space and time dim feature.
        attn_mask = entity_mask.unsqueeze(-1).unsqueeze(1).repeat(1,1,1,self.num_frame)
        out = self.cc_attn(roi_feat, attn_mask)
        out = self.cc_attn(out, attn_mask)

        # Get size.
        N, D, R, T = out.size()

        # Temporal downsample to predict as contexted, out N, R, C, T, D.
        out = self.feature_upsample(out).reshape(batch_size, self.out_channels, R, T)

        out = self.temperol_downsample(out.permute(0,3,2,1)).permute(0,2,3,1).reshape(N,R,self.out_channels)

        return out

        

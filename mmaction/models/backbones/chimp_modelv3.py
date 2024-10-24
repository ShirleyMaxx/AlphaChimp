import warnings
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union
from torch import Tensor

import math
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
from ..roi_heads.roi_extractors.single_straight3d_chimp import *
from ..common.criss_cross_attn import CrissCrossAttention


class ResBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock3D, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(out_channels),
        )
        self.conv3 = None
        if in_channels != out_channels:
            self.conv3 = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.activation = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.conv3 is not None:
            out = out + self.conv3(x)
        else:
            out = out + x
        out = self.activation(out)
        return out

class ResBottleNeck3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBottleNeck3D, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(out_channels),
        )
        self.conv3 = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))
        self.activation = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + self.conv3(x)
        out = self.activation(out)
        return out


@MODELS.register_module()
class ChimpModelV3(BaseModule):
    def __init__(self,
                 num_frame: int = 80,
                 num_keypoint: int = 16,
                 max_chimp: int = 7,
                 d_model: int = 256,
                 out_channels=1024,
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
            self.roi_extractor = SingleRoIExtractor3DChimpAVGV3(**roi_extractor_cfg)
        else:
            self.roi_extractor = SingleRoIExtractor3DChimpAVGV3(max_chimp=max_chimp)

        self.num_frame = num_frame
        self.num_keypoint = num_keypoint
        self.max_chimp = max_chimp
        self.d_model = d_model
        self.out_channels = out_channels

        self.roi_size = self.roi_extractor.output_size
        self.mid_d = self.num_frame // self.img_model.speed_ratio

        sf_fusion_t_models = []
        for i in range(int(math.log2(self.img_model.speed_ratio) + 1e-4)):
            sf_fusion_t_models.append(ResBottleNeck3D(self.d_model, self.d_model))
        self.slow_fast_fusion_t = nn.Sequential(*sf_fusion_t_models)

        sf_fusion_c_models = []
        channel_coefficient = self.img_model.channel_ratio
        for i in range(int(math.log2(self.img_model.channel_ratio) + 1e-4)):
            sf_fusion_c_models.append(ResBlock3D(self.d_model * channel_coefficient, self.d_model * channel_coefficient // 2))
            channel_coefficient = channel_coefficient // 2
        self.slow_fast_fusion_c = nn.Sequential(*sf_fusion_c_models)

        self.roi_downsample = nn.Sequential(
            nn.Conv3d(self.mid_d, self.mid_d,
                      kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(self.num_frame // self.img_model.speed_ratio),
            nn.LeakyReLU(inplace=True),

            nn.Conv3d(self.mid_d, self.mid_d,
                      kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(self.num_frame // self.img_model.speed_ratio),
            nn.LeakyReLU(inplace=True),

            nn.Conv3d(self.mid_d, self.mid_d,
                      kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(self.num_frame // self.img_model.speed_ratio),
            nn.LeakyReLU(inplace=True),
        )

        self.final_d = self.d_model * self.roi_size // 8 * self.roi_size // 8
        self.cc_attn = CrissCrossAttention(self.final_d)

        self.feature_upsample = nn.Sequential(
            nn.Conv2d(self.final_d, self.out_channels, 1, 1, 0, bias=False), nn.BatchNorm2d(self.out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 1, 1, 0)
        )
        self.temperol_downsample = nn.Sequential(
            nn.Conv1d(self.mid_d, self.mid_d // 2, 3, 1, 1, bias=False), nn.BatchNorm1d(self.mid_d // 2), nn.LeakyReLU(inplace=True),
            nn.Conv1d(self.mid_d // 2, 1, 3, 1, 1)
        )

    def forward(self, imgs: Tensor,
                proposals: Tensor,
                entity_mask: Tensor = None):
        '''Forward function. Pipeline: img-hmap feature extraction => roi & ske feature extration => spatio-temporal attention.

        Args:
            img: tensor, shape N, C, T, H, W. RGB.
            rois: list, List[Tensor], List[N]. List of rois in eatch batch, no need to pad.
            roi_mask: tensor, shape N, 1, M, T.
        '''
        # print('PROPOSALS', proposals)
        # print('ENTITY MASK', entity_mask)

        # Get batch size.
        batch_size = imgs.size(0)

        # Image features, out N, C, T, H, W
        img_feat = self.img_model(imgs)
        # print(img_feat[0].size())
        # print(img_feat[1].size())

        # Cat features.
        img_feat_slow, img_feat_fast = img_feat
        img_feat_slow = self.slow_fast_fusion_c(img_feat_slow)
        img_feat_fast = self.slow_fast_fusion_t(img_feat_fast)
        global_feat = torch.cat((img_feat_slow, img_feat_fast), dim=1)
        # Extract roi feature.
        # TODO optimize complexity.

        rois = bbox2roi(proposals)
        # This output N, M, T, C, D, D. where D is roi feature size.
        # Be noticed that the T here is actually T_in // slow_fast_ratio.
        roi_feat, _ = self.roi_extractor(global_feat, rois)
        roi_feat: Tensor = roi_feat.transpose(1, 2)
        N, M, T, C, D, _ = roi_feat.size()

        # Simply downsample it due to the large scale feature size.
        roi_feat = self.roi_downsample(roi_feat.reshape(N * M, T, C, D, D)).reshape(N, M, T, C //2 * D // 8 * D // 8)
        N, M, T, C = roi_feat.size()

        # We permute roi_feat to do attentions.
        roi_feat = roi_feat.permute(0, 3, 1, 2)

        # Compute spatio-temporal criss-cross attention, out N, C, M, T.
        # We do twice criss-cross attention to fully crossing the space and time dim feature.
        attn_mask = entity_mask.unsqueeze(-1).unsqueeze(1).repeat(1, 1, 1, T)
        out = self.cc_attn(roi_feat, attn_mask)
        out = self.cc_attn(out, attn_mask)

        # Feature upsample to out_channels, out N, C, M, T.
        out = self.feature_upsample(out).reshape(batch_size, self.out_channels, M, T)

        # Temporal downsample.
        # We multiply entity mask to zero grad masked chimp pos.
        out = out.permute(0, 2, 3, 1).reshape(N * M, T, self.out_channels)
        out = self.temperol_downsample(out).reshape(N, M, self.out_channels) * entity_mask.reshape(N, M, 1)

        return out



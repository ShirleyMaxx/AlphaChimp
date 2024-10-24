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
class ChimpModel(BaseModule):
    def __init__(self,
                 num_frame: int = 80,
                 num_keypoint: int = 16,
                 max_chimp: int = 7,
                 d_model: int = 256,
                 out_channels = 32,

                 img_model_cfg: Dict = None,
                 hmap_model_cfg: Dict = None,
                 ske_model_cfg: Dict = None,
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

        if hmap_model_cfg is not None:
            self.hmap_model = MODELS.build(hmap_model_cfg)
        else:
            self.hmap_model = ResNet3dSlowFast()

        if ske_model_cfg is not None:
            self.ske_model = MODELS.build(ske_model_cfg)
        else:
            self.ske_model = STGCN(graph_cfg=dict(layout='chimp', mode='stgcn_spatial'), num_person=max_chimp)

        if roi_extractor_cfg is not None:
            self.roi_extractor = SingleRoIExtractor3DChimp(**roi_extractor_cfg)
        else:
            self.roi_extractor = SingleRoIExtractor3DChimp(max_chimp=max_chimp)
        
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
            nn.Conv2d(self.d_model*4, self.d_model, 3, 1, 1), nn.BatchNorm2d(self.d_model), nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.d_model, self.d_model, 3, 2, 1), nn.BatchNorm2d(self.d_model), nn.LeakyReLU(inplace=True),
            nn.AvgPool2d(4, 4),
            nn.LeakyReLU(inplace=True),
        )
        self.ske_spatio_downsample = nn.Sequential(
            nn.Conv2d(self.d_model, self.d_model//4, 1, 1, 0), nn.BatchNorm2d(self.d_model//4), nn.LeakyReLU(inplace=True)
        )
        self.ske_temporal_upsample = nn.Sequential(
            nn.Conv2d(self.num_frame//4, self.num_frame, 1, 1, 0), nn.BatchNorm2d(self.num_frame), nn.LeakyReLU(inplace=True)
        )

        self.final_d = self.d_model//4 * self.num_keypoint + self.d_model * self.roi_size//8 * self.roi_size//8
        self.cc_attn = CrissCrossAttention(self.final_d)

        self.feature_upsample = nn.Sequential(
            nn.Conv2d(1, self.out_channels, 3, 1, 1), nn.BatchNorm2d(self.out_channels), nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1), nn.BatchNorm2d(self.out_channels), nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1), nn.BatchNorm2d(self.out_channels), nn.LeakyReLU(inplace=True)
        )
            

    def forward(self,
                imgs: Tensor,
                heatmap_imgs: Tensor,
                keypoint: Tensor,
                proposals: List,
                entity_mask: Tensor = None):
        '''Forward function. Pipeline: img-hmap feature extraction => roi & ske feature extration => spatio-temporal attention.

        Args:
            img: tensor, shape N, C, T, H, W. RGB.
            hmap: tensor, shape N, C, T, H, W. Expected to have 3 channels.
            ske: tensor, shape N, M, T, J, 3. In which M is max_chimp, J is num_keypoints.
            rois: list, List[Tensor], List[N]. List of rois in eatch batch, no need to pad.
            roi_mask: tensor, shape N, 1, M, T.
        '''

        # Get batch size.
        batch_size = heatmap_imgs.size(0)

        # Image features, out N, C, T, H, W
        img_feat = self.img_model(imgs)

        # heatmap_imgs features, out N, C, T, H, W
        hmap_feat = self.hmap_model(heatmap_imgs)

        
        # Cat features.
        img_feat_slow, hmap_feat_slow = img_feat[0].transpose(1, 2), hmap_feat[0].transpose(1, 2)

        img_feat_slow, hmap_feat_slow = self.slow_fast_fusion_c(img_feat_slow.reshape((-1, ) + img_feat_slow.size()[2:])), self.slow_fast_fusion_c(hmap_feat_slow.reshape((-1, ) + hmap_feat_slow.size()[2:]))
        img_feat_slow = img_feat_slow.reshape((batch_size, -1) + img_feat_slow.size()[1:]).transpose(1, 2)
        hmap_feat_slow = hmap_feat_slow.reshape((batch_size, -1) + hmap_feat_slow.size()[1:]).transpose(1, 2)

        img_feat_slow, hmap_feat_slow = self.slow_fast_fusion_t(img_feat_slow.reshape((-1, ) + img_feat_slow.size()[2:])), self.slow_fast_fusion_t(hmap_feat_slow.reshape((-1, ) + hmap_feat_slow.size()[2:]))
        global_feat = torch.cat((img_feat_slow.reshape((batch_size, -1) + img_feat_slow.size()[1:]), img_feat[1], hmap_feat_slow.reshape((batch_size, -1) + hmap_feat_slow.size()[1:]), hmap_feat[1]), dim=1)
        # Extract roi feature.
        # TODO optimize complexity.

        rois = bbox2roi(proposals) 
        bbox_feats, __ = self.roi_extractor(global_feat, proposals)
        roi_feat = self.roi_downsample(roi_feat.reshape((-1, ) + roi_feat.size()[3:])).reshape(batch_size, self.num_frame, self.max_chimp, self.d_model * self.roi_size//8 * self.roi_size//8)
        roi_feat = roi_feat.permute(0, 3, 2, 1)

        # Skeleton feature, out N, M, C, T, V
        if keypoint.size(1) < self.max_chimp:
            ske = torch.cat([keypoint, torch.zeros(batch_size, self.max_chimp-keypoint.size(1), self.num_frame, self.num_keypoint, 3, device=keypoint.device)], dim=1)
        else:
            ske = keypoint
        ske_feat = self.ske_model(ske).reshape(batch_size * self.max_chimp, self.d_model, self.num_frame//4, self.num_keypoint)
        ske_feat = self.ske_temporal_upsample(ske_feat.transpose(1, 2)).transpose(1, 2)
        ske_feat = self.ske_spatio_downsample(ske_feat).transpose(1, 2).reshape(batch_size, self.max_chimp, self.num_frame, self.d_model//4 * self.num_keypoint)
        ske_feat = ske_feat.permute(0, 3, 1, 2)

        # Chimp feature.
        out = torch.cat((roi_feat, ske_feat), dim=1)

        # Compute spatio-temporal criss-cross attention, out N, T, R, D.
        # We do twice criss cross attention to fully crossing the space and time dim feature.
        attn_mask = []
        for b in proposals:
            attn_mask.append(torch.from_numpy(np.fromfunction(lambda i, j, k: (j < b.size(0)).astype(np.float32), shape=(1, self.max_chimp, self.num_frame))).to(imgs.device))
        attn_mask = torch.cat(attn_mask, dim=0).unsqueeze(1)

        out = self.cc_attn(out, attn_mask)
        out = self.cc_attn(out, attn_mask).permute(0, 3, 2, 1)

        # Get size.
        T, R, D = out.size()[1:]

        # Temporal downsample to predict as contexted, out N, R, C, T, D.
        out = self.feature_upsample(out.reshape(batch_size, 1, T, R*D)).reshape(batch_size, self.out_channels, T, R, D).permute(0, 3, 1, 2, 4)

        # Get size.
        T, R, D = out.size()[1:]

        # Temporal downsample to predict as contexted, out N, R, C, T, D.
        out = self.feature_upsample(out.reshape(batch_size, 1, T, R*D)).reshape(batch_size, self.out_channels, T, R, D).permute(0, 3, 1, 2, 4)

        return out

        
@MODELS.register_module()
class ChimpModelAVG(BaseModule):
    def __init__(self,
                 num_frame: int = 80,
                 num_keypoint: int = 16,
                 max_chimp: int = 7,
                 d_model: int = 256,
                 out_channels = 32,

                 img_model_cfg: Dict = None,
                 hmap_model_cfg: Dict = None,
                 ske_model_cfg: Dict = None,
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

        if hmap_model_cfg is not None:
            self.hmap_model = MODELS.build(hmap_model_cfg)
        else:
            self.hmap_model = ResNet3dSlowFast()

        if ske_model_cfg is not None:
            self.ske_model = MODELS.build(ske_model_cfg)
        else:
            self.ske_model = STGCN(graph_cfg=dict(layout='chimp', mode='stgcn_spatial'), num_person=max_chimp)

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
            nn.Conv2d(2048, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.LeakyReLU(inplace=True)
        )

        self.roi_downsample = nn.Sequential(
            nn.Conv2d(in_channels=self.d_model*4, 
                      out_channels =self.d_model, 
                      kernel_size = 1,
                      stride = 1, 
                      padding = 0), 
            nn.BatchNorm2d(self.d_model), 
            nn.LeakyReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.ske_spatio_downsample = nn.Sequential(
            nn.Conv2d(self.d_model, self.d_model//4, 1, 1, 0), nn.BatchNorm2d(self.d_model//4), nn.LeakyReLU(inplace=True)
        )

        self.final_d = self.d_model//4 * self.num_keypoint + self.d_model
        self.cc_attn = CrissCrossAttention(self.final_d)

        self.feature_upsample = nn.Linear(self.final_d,self.out_channels)
            

    def forward(self,
                imgs: Tensor,
                heatmap_imgs: Tensor,
                keypoint: Tensor,
                proposals: List,
                entity_mask: Tensor = None):
        '''Forward function. Pipeline: img-hmap feature extraction => roi & ske feature extration => spatio-temporal attention.

        Args:
            img: tensor, shape N, C, T, H, W. RGB.
            hmap: tensor, shape N, C, T, H, W. Expected to have 3 channels.
            ske: tensor, shape N, M, T, J, 3. In which M is max_chimp, J is num_keypoints.
            rois: list, List[Tensor], List[N]. List of rois in eatch batch, no need to pad.
            roi_mask: tensor, shape N, M.
        '''

        # Get batch size.
        batch_size = heatmap_imgs.size(0)

        # Image features, out N, C, T, H, W
        img_feat = self.img_model(imgs)

        # heatmap_imgs features, out N, C, T, H, W
        hmap_feat = self.hmap_model(heatmap_imgs)

        
        # Cat features.
        img_feat_slow, hmap_feat_slow = img_feat[0].transpose(1, 2), hmap_feat[0].transpose(1, 2)

        img_feat_slow, hmap_feat_slow = self.slow_fast_fusion_c(img_feat_slow.reshape((-1, ) + img_feat_slow.size()[2:])), self.slow_fast_fusion_c(hmap_feat_slow.reshape((-1, ) + hmap_feat_slow.size()[2:]))
        img_feat_slow = img_feat_slow.reshape((batch_size, -1) + img_feat_slow.size()[1:]).transpose(1, 2)
        hmap_feat_slow = hmap_feat_slow.reshape((batch_size, -1) + hmap_feat_slow.size()[1:]).transpose(1, 2)

        img_feat_slow, hmap_feat_slow = self.slow_fast_fusion_t(img_feat_slow.reshape((-1, ) + img_feat_slow.size()[2:])), self.slow_fast_fusion_t(hmap_feat_slow.reshape((-1, ) + hmap_feat_slow.size()[2:]))
        global_feat = torch.cat((img_feat_slow.reshape((batch_size, -1) + img_feat_slow.size()[1:]), img_feat[1], hmap_feat_slow.reshape((batch_size, -1) + hmap_feat_slow.size()[1:]), hmap_feat[1]), dim=1)
        # Extract roi feature.
        # TODO optimize complexity.

        rois = bbox2roi(proposals) 
        # roi_feat N,M,C,H',W'
        roi_feat, __ = self.roi_extractor(global_feat, rois)
        roi_feat = self.roi_downsample(roi_feat.reshape((-1, ) + roi_feat.size()[2:]))
        roi_feat = roi_feat.reshape(batch_size, self.max_chimp, -1)

        if keypoint.size(1) < self.max_chimp:
            ske = torch.cat([keypoint, torch.zeros(batch_size, self.max_chimp-keypoint.size(1), self.num_frame, self.num_keypoint, 3, device=keypoint.device)], dim=1)
        else:
            ske = keypoint
        ske_feat = self.ske_model(ske).reshape(batch_size,self.max_chimp, self.d_model, self.num_frame//4, self.num_keypoint)
        ske_feat = ske_feat.mean(dim=-2).transpose(2,1)
        ske_feat = self.ske_spatio_downsample(ske_feat).permute(0,2,3,1).reshape(batch_size,self.max_chimp,-1)

        feat = torch.cat((roi_feat, ske_feat), dim=-1).permute(0,2,1).unsqueeze(-1)
        out_feat = self.cc_attn(feat,entity_mask.unsqueeze(1).unsqueeze(-1))
        out_feat = out_feat.squeeze(-1).permute(0,2,1)

        out = self.feature_upsample(out_feat)


        return out

        


    

    
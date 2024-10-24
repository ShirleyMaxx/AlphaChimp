import inspect
import warnings
from abc import ABCMeta, abstractmethod

import math
import torch
import torch.nn as nn
from mmengine.model import BaseModule, merge_dict

from mmaction.registry import MODELS
from mmaction.utils import (ConfigType, ForwardResults, OptConfigType,
                            OptSampleList, SampleList)

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
class Resnet3DChimp(BaseModule):
    '''This is the Resnet3DChimp backbone for temporal feature extraction.
    Be noticed that the backbone may not really implement the 'Resnet3D'
    structure due to practical reasons.

    '''
    def __init__(self, out_size: tuple = (512, 512),
                 out_channels: int = 128,
                 depth: int = 10,
                 temporal_len: int = 32) -> None:
        super(Resnet3DChimp, self).__init__(init_cfg=None)

        self.out_channels = out_channels

        model_list = [ResBottleNeck3D(3, out_channels), ResBottleNeck3D(out_channels, out_channels)]
        for d in range(depth):
            model_list.append(ResBlock3D(out_channels, out_channels))
        for d in range(int(math.log2(temporal_len // 4))):
            model_list.append(nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1), bias=False))
            model_list.append(nn.BatchNorm3d(out_channels))
            model_list.append(nn.LeakyReLU(inplace=True))
        model_list.append(nn.AdaptiveAvgPool3d((1, ) + out_size))

        self.conv_seq = nn.Sequential(*model_list)
        model_list.clear()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, T, H, W = x.size()
        x = self.conv_seq(x).reshape(N, self.out_channels, H, W)
        return x




# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Superresolution network architectures from the paper
"Efficient Geometry-aware 3D Generative Adversarial Networks"."""

import torch
import torch.nn as nn
import math
import numpy as np
from .utils import misc
from .utils.stylegan2_official_ops import upfirdn2d
# from .stylegan2_generator import ConvLayer, ModulateConvLayer
from .layers import ConvRenderBlock2d, ModResBlock, ConvLayer2d



#----------------------------------------------------------------------------
# Resolutions allowed.
_SUPER_RES_MULTIPLE_ALLOWED = [1, 2 ,4]
# for 512x512 generation

class SuperresolutionHybrid(torch.nn.Module):
    """2D rendering refinement module from EG3D.
    """

    def __init__(self, 
                 in_channel, 
                 in_res, 
                 out_res,
                 style_channel,
                 hidden_channel = 128, 
                 block_num = 2,
                 mode='blur', deep=False, **kwargs):
        super().__init__()

        log_size_in = int(math.log(in_res, 2))
        log_size_out = int(math.log(out_res, 2))
        sr_multiple =  (log_size_out - log_size_in)
        assert 2 ** sr_multiple in _SUPER_RES_MULTIPLE_ALLOWED 
        # self.render_blocks = nn.ModuleList()
        assert block_num >= sr_multiple 
        self.sr_multiple = sr_multiple
        self.block_num = block_num

        c_in, c_out = in_channel, min(in_channel*2, hidden_channel)
        for i in range(block_num):
            if i < sr_multiple: 
                layer_name = f'modRes_block{i} with upsampling'
                self.add_module(layer_name,
                    ModResBlock(in_channel=c_in, out_channel=c_out, z_dim=style_channel, upsample_factor=2)
                )
            else:
                layer_name = f'modRes_block{i} without upsampling'
                self.add_module(layer_name,
                    ModResBlock(in_channel=c_in, out_channel=c_out,z_dim=style_channel, upsample_factor=1))

            c_in = c_in * 2
            c_out = min(c_in*2, hidden_channel)
    def forward(self, x, rgb = None, z = None):
        """Forward pass.

        Input:
        -----
        x: torch.Tensor
            Input feature maps of shape [B, in_channel, in_res, in_res].

        Return:
        ------
        rgb: torch.Tensor
            RGB images of shape [B, 3, out_res, out_res].

        """
        # rgb = None
        for i in range(self.block_num):
            if i < self.sr_multiple:
                x, rgb = getattr(self, f'modRes_block{i} with upsampling')(x,z, rgb)
            else:
                x, rgb = getattr(self, f'modRes_block{i} without upsampling')(x,z, rgb)

        rgb = torch.tanh(rgb)
        return rgb

#--------------------------------------------------------------------------------------------------------------------------------------------
class SuperresolutionBase(torch.nn.Module):
    """2D rendering refinement module from urbanGIRAFFE.
    """

    def __init__(self, 
                 in_channel, 
                 in_res, 
                 out_res,
                 style_channel,
                 hidden_channel = 128, 
                 block_num = 2,
                 **kwargs):
        super().__init__()

        log_size_in = int(math.log(in_res, 2))
        log_size_out = int(math.log(out_res, 2))
        sr_multiple =  (log_size_out - log_size_in)
        assert 2 ** sr_multiple in _SUPER_RES_MULTIPLE_ALLOWED 
        # self.render_blocks = nn.ModuleList()
        assert block_num >= sr_multiple 
        self.sr_multiple = sr_multiple
        self.block_num = block_num

        c_in, c_out = in_channel, min(in_channel*2, hidden_channel)
        for i in range(block_num):
            if i < sr_multiple: 
                layer_name = f'modRes_block{i} with upsampling'
                self.add_module(layer_name,
                    ModResBlock(in_channel=c_in, out_channel=c_out, z_dim=style_channel, upsample_factor=2)
                )
            else:
                layer_name = f'modRes_block{i} without upsampling'
                self.add_module(layer_name,
                    ModResBlock(in_channel=c_in, out_channel=c_out,z_dim=style_channel, upsample_factor=1))
            c_in = c_in * 2
            c_out = min(c_in*2, hidden_channel)

        self.toRGB = ConvLayer2d(hidden_channel, 3, 1, bias=True, activate=False)
    def forward(self, x, rgb = None, z = None):
        """Forward pass.

        Input:
        -----
        x: torch.Tensor
            Input feature maps of shape [B, in_channel, in_res, in_res].

        Return:
        ------
        rgb: torch.Tensor
            RGB images of shape [B, 3, out_res, out_res].

        """
        # rgb = None
        for i in range(self.block_num):
            if i < self.sr_multiple:
                x, _ = getattr(self, f'modRes_block{i} with upsampling')(x = x, z = z, img = None)
            else:
                x, _ = getattr(self, f'modRes_block{i} without upsampling')(x = x,z = z, img = None)
        rgb = self.toRGB(x)
        rgb = torch.tanh(rgb)
        return rgb

# @persistence.persistent_class
class RenderNet2d(nn.Module):
    """2D rendering refinement module.

    Inspired by GIRAFFE: https://arxiv.org/abs/2011.12100

    This module takes as input a set of feature maps and upsamples them to higher resolution RGB outputs. Skips
    connections are used to aggregate RGB outputs at each layer for more stable training.

    Args:
    ----
    in_channel: int
        Input channel.
    in_res: int
        Input resolution.
    out_res: int
        Output resolution.
    mode: str
        Which mode to use for the render block. Options are 'original' and 'blur'. Original mode is implemented
        as described in the GIRAFFE paper using nearest neighbour upsampling + conv. Blur mode is closer to the
        skip generator in StyleGAN2, and uses transposed convolution with stride for upsampling.
    deep: bool
        Each block in the rendering network uses two convolutional layers. Otherwise, a single 3x3 conv is used
        per resolution.

    """

    def __init__(self, 
                 in_channel, 
                 in_res, 
                 out_res, 
                 mode='blur', deep=False, **kwargs):
        super().__init__()

        log_size_in = int(math.log(in_res, 2))
        log_size_out = int(math.log(out_res, 2))

        # self.render_blocks = nn.ModuleList()
        block_num = log_size_out - log_size_in
        self.block_num = block_num
        for i in range(block_num):
            layer_name = f'sr_block{i}'
            self.add_module(layer_name,
                ConvRenderBlock2d(in_channel=in_channel, out_channel=in_channel // 2, mode=mode, deep=deep)
            )
            in_channel = in_channel // 2

    def forward(self, x, rgb = None, z = None):
        """Forward pass.

        Input:
        -----
        x: torch.Tensor
            Input feature maps of shape [B, in_channel, in_res, in_res].

        Return:
        ------
        rgb: torch.Tensor
            RGB images of shape [B, 3, out_res, out_res].

        """
        # rgb = None
        for i in range(self.block_num):
            x, rgb = getattr(self, f'sr_block{i}')(x, rgb)

        rgb = torch.tanh(rgb)
        return rgb



#-------------------------------------------------------------------------
#----------------------------Legacy------------------------------------------
#-------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class ResnetBlock(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # norm_layer = nn.BatchNorm2d(num_features=dim, affine=True ,track_running_stats = False)

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channel = dim, out_channel = dim, kernel_size=kernel_size,stride=1, padding=1),
            nn.BatchNorm2d(num_features=dim, affine=True ,track_running_stats = False),
            self.activation,
            nn.Conv2d(in_channel = dim, out_channel = dim, kernel_size=kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(num_features=dim, affine=True ,track_running_stats = False)
        )

    def forward(self, x):
        y = self.conv_block(x)
        out = self.activation(x + y)
        return out
class CNNRender_legacy(nn.Module):
    r"""CNN converting intermediate feature map to final image."""

    def __init__(self, 
                 in_channel = 64, 
                 hidden_channel=128, 
                 style_channel = 64,
                 leaky_relu=True, sr_multiple = 1, **kwagrs):
        super(CNNRender_legacy, self).__init__()
        #self.fc_z_cond = nn.Linear(style_dim, 2 * 2 * hidden_channel)
        self.conv_in = nn.Conv2d(in_channel, hidden_channel, 1, stride=1, padding=0)
        AdainResBlk(hidden_channel,hidden_channel,style_channel)
        self.res0 = AdainResBlk(hidden_channel,hidden_channel,style_channel, upsample = (sr_multiple >= 4))
        self.res1 = AdainResBlk(hidden_channel, hidden_channel,style_channel, upsample =(sr_multiple >= 2))

        self.toRGB = nn.Conv2d(hidden_channel, 3, 1, stride=1, padding=0)

        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x, z = None, rgb = None):
        r"""Forward network.

        Args:
            x (N x in_channel x H x W tensor): Intermediate feature map
            z (N x style_dim tensor): Style codes.
        """
        # z = self.fc_z_cond(z)
        # adapt = torch.chunk(z, 2 * 2, dim=-1)

        x = self.act(self.conv_in(x))
        x = self.res0(x, z)
        x = self.res1(x, z)
        rgb = self.toRGB(x)
        # if True:
        #     rgb = torch.tanh(rgb)

        return rgb



def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()


class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class AdainResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64, w_hpf=0,
                 actv=nn.LeakyReLU(0.2), upsample=False):
        super().__init__()
        self.w_hpf = w_hpf
        self.actv = actv
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)

    def _build_weights(self, dim_in, dim_out, style_dim=64):
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        # a = x.detach().cpu().numpy()
        x = self.actv(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        if self.w_hpf == 0:
            out = (out + self._shortcut(x)) / math.sqrt(2)
        return out




import math
import numpy as np
import torch
from torch import nn
from torch import pi
import torch.nn.functional as F
import os
# from .stylegan2_generator import DenseLayer as Linear
# from .utils.stylegan2_official_ops import bias_act

from .layers import EqualLinear as Linear
from .layers import PositionalEncoding

class StuffDecoder(nn.Module):
    r""""""
    def __init__(self, 
                in_channel = 3,
                w_channel = 16,
                seg_channel = 16,
                hidden_channel=256,
                out_channel = 64,
                style_channel = 512,
                n_block_num = 4,
                use_seg = True,
                use_viewdirs=False,
                use_positonal_encoding=True,
                n_freq_posenc_pts=10,
                n_freq_posenc_w=0,
                n_freq_posenc_views=4,
                # Settings 
                final_tanh=False,
                demodulate=True,
                use_wscale=True,
                wscale_gain=1.0,
                lr_mul=1.0,
                eps=1e-8,
                **kwargs
                ):
        super(StuffDecoder, self).__init__()
        assert n_block_num >= 1
        self.in_channel = in_channel
        self.use_viewdirs = use_viewdirs
        self.use_posotonal_encoding = use_positonal_encoding
        self.n_block_num = n_block_num
        self.final_tanh = final_tanh
        self.use_seg = use_seg


        self.in_channel_x = in_channel
        self.in_channel_w = w_channel 
        if self.use_posotonal_encoding:
            self.pe_pts = PositionalEncoding(in_dim=in_channel, frequency_bands=n_freq_posenc_pts, include_input= True)
            self.pe_w = PositionalEncoding(in_dim=w_channel, frequency_bands=n_freq_posenc_w, include_input= False)

        in_channel_x = getattr(self, f'in_channel_x')
        in_channel_w =  getattr(self, f'in_channel_w')
        if use_positonal_encoding:
            self.add_module('pe_pts',
                PositionalEncoding(in_dim=in_channel, frequency_bands=n_freq_posenc_pts, include_input= True))
            self.add_module('pe_w',
                PositionalEncoding(in_dim=w_channel, frequency_bands=n_freq_posenc_w, include_input= True))
            in_channel_x = self.pe_pts.out_dim
            in_channel_w = self.pe_w.out_dim
        if self.use_seg:
            in_channel_w += seg_channel
        self.add_module('fc_x',
                Linear(
                in_channel=in_channel_x,
                out_channel=hidden_channel // 2,
                bias=True,
                activate=True))
        self.add_module('fc_w',
                Linear(
                in_channel=in_channel_w,
                out_channel=hidden_channel // 2,
                bias=True,
                activate=True))
        
        for i in range(n_block_num):
            layer_name = f'main_layer{i}'
            self.add_module(layer_name,
                Linear(in_channel=hidden_channel,
                                    out_channel=hidden_channel,
                                    activate=True))

        self.add_module('fc_sigma',
                Linear(in_channel=hidden_channel,
                out_channel=1,
                bias=True,
                activate=False))

        # Feat part
        if self.use_viewdirs:
            dir_channel_pe = getattr(self, f'in_channel_dir')
            if self.use_posotonal_encoding:
                self.add_module('pe_w',
                    PositionalEncoding(in_dim=w_channel, frequency_bands=n_freq_posenc_w, include_input= True))
                dir_channel_pe = self.pe_w.out_dim
            else:
                dir_channel_pe
            self.add_module('fc_dir',
                    Linear(in_channel=dir_channel_pe,
                    out_channel=hidden_channel,
                    bias=True,
                    activate=True))

        self.add_module('fc_feat',
                Linear(in_channel=hidden_channel,
                out_channel= out_channel,
                activate=False,
                bias=True))


    def forward(self, pts, w, raydir, seg = None):
        r""" Forward network

        Args:
            w (N x H x W x M x in_channel tensor): Projected features.
            pts (N x H x W x M x in_channel tensor): Projected features.
            raydir (N x H x W x 1 x viewdir_dim tensor): Ray directions.
            seg (N x H x W x M x mask_dim tensor): One-hot segmentation maps.
        """
        # Position endoding
        pts = pts[...,:self.in_channel]
        if self.use_posotonal_encoding:
            w = self.pe_w(w)
            x = self.pe_pts(pts)
        else:
            w = w
            x = pts
        if self.use_seg:
            w = torch.cat((w, seg) ,dim = -1)
        # Common MLP
        x = torch.concat((self.fc_x(x), self.fc_w(w)), dim = 1)
        for i in range(self.n_block_num):
            layer = getattr(self, f'main_layer{i}')
            x = layer(x)
            if (i + 2) == self.n_block_num:
                sigma = self.fc_sigma(x)
        # Color MLP
        if self.use_viewdirs:
            if self.use_posotonal_encoding:
                r = self.pe_dirs(raydir)
            x = x + self.fc_dir(r)
        feat = self.fc_feat(x)
        if self.final_tanh:
            feat = torch.tanh(feat)
        return feat ,sigma


class StuffDecoder_legacy(nn.Module):
    r""" MLP"""

    def __init__(self, 
                in_channel = 3,
                w_channel = 16,
                out_channel_s=1, 
                out_channel_c=3, 
                out_channel = 64,
                style_channel = 64,
                n_block_num = 2,
                hidden_channel=256,
                semantic_dim=16,
                use_pts= True,
                use_viewdirs=False,
                use_seg = False,
                use_final_sigmoid=True,
                use_positonal_encoding=False,
                use_density_modulate= False,
                n_freq_posenc_w=4,
                n_freq_posenc_pts=10,
                n_freq_posenc_views=4,

                use_wscale=True,
                wscale_gain=1.0,
                lr_mul=1.0,
                **kwargs
                ):
        super(StuffDecoder_legacy, self).__init__()
        assert n_block_num >= 2
        self.in_channel = in_channel
        self.w_channel = w_channel
        self.use_final_sigmoid = use_final_sigmoid
        self.use_seg= use_seg
        self.use_pts = use_pts
        self.use_viewdirs = use_viewdirs
        self.use_posotonal_encoding = use_positonal_encoding
        self.use_density_modulate = use_density_modulate
        self.n_block_num = n_block_num
        if use_seg:
            self.fc_m_a = nn.Linear(semantic_dim, hidden_channel)


        self.pe_pts = PositionalEncoding(in_dim=in_channel, frequency_bands=n_freq_posenc_pts, include_input= False)
        self.pe_z = PositionalEncoding(in_dim=self.w_channel//2, frequency_bands=n_freq_posenc_w, include_input= False)
        self.pe_dirs = PositionalEncoding(in_dim=3, frequency_bands=n_freq_posenc_views, include_input= False)

        #self.fc_viewdir = None
        self.fc_viewdir = nn.Linear(self.pe_dirs.out_dim, hidden_channel)

        # self.fc_in = nn.Linear(self.pe_x.out_dim + self.pe_pts.out_dim * use_pts + (in_channel // 2) + use_seg * semantic_dim, hidden_channel)
        # self.fc_in = nn.Linear(self.pe_x.out_dim + self.in_channel//2 + use_seg * semantic_dim, hidden_channel)


        
        if use_positonal_encoding:
            z_in_dim = self.w_channel//2 + self.pe_z.out_dim + semantic_dim * use_seg
            self.fc_z = nn.Linear(z_in_dim, hidden_channel // 2)
            self.fc_pts = nn.Linear(self.pe_pts.out_dim, hidden_channel // 2)
        else:
            z_in_dim = self.w_channel
            self.fc_z = nn.Linear(z_in_dim, hidden_channel // 2)
            self.fc_pts = nn.Linear(3, hidden_channel // 2)
        
        self.fc_main = nn.ModuleList([])
       
        for i in range(n_block_num):
            self.fc_main.append(nn.Linear(hidden_channel, hidden_channel))
            

        self.fc_sigma = nn.Linear(hidden_channel, out_channel_s)

        # Feat part
        if self.use_viewdirs:
            in_dim_dir = self.pe_dirs.out_dim if use_viewdirs else 3
            self.fc_dir = nn.Linear(in_dim_dir, hidden_channel)

        self.fc_feat = nn.Linear(hidden_channel, out_channel)
        self.fc_rgb = nn.Linear(out_channel, out_channel_c)

        self.act = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, w, pts = None, raydir = None, seg = None):
        r""" Forward network

        Args:
            z (N x H x W x M x in_channel tensor): Projected features.
            raydir (N x H x W x 1 x viewdir_dim tensor): Ray directions.
            seg (N x H x W x M x mask_dim tensor): One-hot segmentation maps.
        """
        pts = pts[...,:self.in_channel]
        p_num, c_in = w.size()
        z = w
        # partial position endoding
        if self.use_posotonal_encoding:
            z = torch.cat((self.pe_z(z[...,:self.w_channel//2]), z[...,self.w_channel//2:]), dim = -1)
            pts = self.pe_pts(pts)

        if self.use_seg:
            assert seg != None
            z = torch.cat((z, seg), dim = -1)

        if self.use_pts:
            assert pts != None
            f = self.act(torch.cat((self.fc_pts(pts), self.fc_z(z)), dim = -1))
        else: 
            f = self.act(self.fc_feat(z))
        # Common MLP
        for i, layer in enumerate(self.fc_main):
            f = self.act(layer(f))
            if (i + 2) == self.n_block_num:
                # Sigma MLP
                sigma = self.fc_sigma(f)

        if self.use_density_modulate:
            a = z.detach().cpu().numpy()
            sigma = torch.sigmoid(z[...,0:1]) * sigma

        # Color MLP
        if self.use_viewdirs:
            r =self.pe_dirs(raydir)
            f = self.fc_dir(f)
            f = f + self.fc_viewdir(r)

        feat = self.fc_feat(f)
        if True: 
            feat = torch.tanh(feat)

        rgb = self.fc_rgb(feat)
        # if self.use_final_sigmoid:
        #     rgb= torch.sigmoid(rgb)
        # feat = torch.cat((feat, rgb), dim = -1)
        return feat ,sigma

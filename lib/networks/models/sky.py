import torch.nn as nn
from .layers import PositionalEncoding
from .layers import EqualLinear as Linear
# from .layers import ModulationLinear


class SkyDecoder(nn.Module):
    r""""""
    def __init__(self, 
                in_channel = 3,
                hidden_channel=128,
                out_channel = 64,
                style_channel = 512,
                n_block_num = 4,
                use_positonal_encoding=True,
                n_freq_posenc=10,
                # Settings 
                final_tanh=True,
                demodulate=True,
                use_wscale=True,
                wscale_gain=1.0,
                lr_mul=1.0,
                eps=1e-8,**kwargs
                ):
        super(SkyDecoder, self).__init__()
        assert n_block_num >= 1
        self.in_channel = in_channel
        self.use_posotonal_encoding = use_positonal_encoding
        self.n_block_num = n_block_num
        self.final_tanh = final_tanh

        self.in_channel_pe = in_channel
        in_channel_pe = getattr(self, f'in_channel_pe')
        if use_positonal_encoding:
            self.add_module('pe',
                PositionalEncoding(in_dim=3, frequency_bands=n_freq_posenc, include_input= True))
            in_channel_pe = self.pe.out_dim
        
        self.fc_x = Linear(in_channel=in_channel_pe,
            out_channel=hidden_channel // 2,bias=True,activate=False)
        self.fc_style = Linear(in_channel=style_channel,
            out_channel=hidden_channel - in_channel_pe,
            bias=True,
            activate=False)

        for i in range(n_block_num):
            layer_name = f'main_layer{i}'
            activate= True
            in_dim = out_dim = hidden_channel
            if i == (n_block_num - 1):
                out_dim = out_channel # feat & rgb
                activate = False
            self.add_module(layer_name,
                Linear(in_channel=in_dim,
                        out_channel=out_dim,
                        bias=True,
                        activate=activate))

    def forward(self, x, z, **kwargs):
        r""" Forward network

        Args:
            w (N x H x W x M x in_channel tensor): Projected features.
            pts (N x H x W x M x in_channel tensor): Projected features.
            raydir (N x H x W x 1 x viewdir_dim tensor): Ray directions.
            seg (N x H x W x M x mask_dim tensor): One-hot segmentation maps.
        """
        # Position endoding
        B, N, C = x.shape
        if self.use_posotonal_encoding:
            x = self.pe(x)
        while z.dim() < x.dim():
            z = z.unsqueeze(1).repeat(1,x.shape[1],1)
        x = x.reshape(B * N,-1)
        z = z.reshape(B * N,-1)

        feat = torch.cat((x, self.fc_style(z)), dim = 1)
        for i in range(self.n_block_num):
            layer = getattr(self, f'main_layer{i}')
            feat = layer(feat)

        if self.final_tanh:
            feat = torch.tanh(feat)
        
        feat = feat.reshape(B * N,-1)
        return feat



#------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.config import cfg
from numpy import pi


class SkyDecoder_legacy(nn.Module):
    r"""MLP converting ray directions to sky features."""

    def __init__(self, 
                 style_channel = 256,in_channel = 3,frequency_bands = 6, 
    out_channel_c=3,
    out_channel = 64,
    hidden_channel=256, leaky_relu=True, use_final_sigmoid = True,
    **kwargs):
        super(SkyDecoder_legacy, self).__init__()
        self.use_final_sigmoid = use_final_sigmoid

        self.positional_encode = PositionalEncoding(in_dim= in_channel, frequency_bands=frequency_bands ) 

        self.fc_z_a = nn.Linear(style_channel, hidden_channel, bias=False)

        self.fc1 = nn.Linear(in_channel + frequency_bands * 6 + hidden_channel , hidden_channel)
        self.fc2 = nn.Linear(hidden_channel, hidden_channel)
        self.fc3 = nn.Linear(hidden_channel, hidden_channel)
        self.fc4 = nn.Linear(hidden_channel, hidden_channel)
        self.fc5 = nn.Linear(hidden_channel, hidden_channel)

        self.fc_out_c = nn.Linear(out_channel, out_channel_c)
        self.fc_out_feat = nn.Linear(hidden_channel, out_channel)

        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x, z):
        r"""Forward network

        Args:
            x (... x in_channel tensor): Ray direction embeddings.
            z (... x style_dim tensor): Style codes.
        """

        x  = self.positional_encode(x)
        z = self.fc_z_a(z)
        while z.dim() < x.dim():
            z = z.unsqueeze(1).repeat(1,x.shape[1],1)
        # z = z.repeat(1,x.shape[1],1)
        y = self.fc1(torch.cat((x, z), dim = -1))
        y = self.act(y)
        y = self.act(self.fc2(y))
        y = self.act(self.fc3(y))
        y = self.act(self.fc4(y))
        y = self.act(self.fc5(y))



        feat = self.fc_out_feat(y)
        if True: 
            feat = torch.tanh(feat)
        # rgb = self.fc_out_c(feat)
        # if self.use_final_sigmoid:
        #     rgb = torch.sigmoid(rgb)
        # feat = torch.cat((feat, rgb), dim = -1)
        return feat


import imp
import torch 
import torch.nn as nn
import torch.nn.functional as F 

import math
import numpy as np 


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


class HighPass(nn.Module):
    def __init__(self, w_hpf, device):
        super(HighPass, self).__init__()
        self.register_buffer('filter',
                             torch.tensor([[-1, -1, -1],
                                           [-1, 8., -1],
                                           [-1, -1, -1]]) / w_hpf)

    def forward(self, x):
        filter = self.filter.unsqueeze(0).unsqueeze(1).repeat(x.size(1), 1, 1, 1)
        return F.conv2d(x, filter, padding=1, groups=x.size(1))

class MappingNetwork(torch.nn.Module):
    def __init__(self,
        z_dim = 64,                      # Input latent (Z) dimensionality, 0 = no latent.
        c_dim = 0,                      # Conditioning label (C) dimensionality, 0 = no label.
        w_dim = 64,                      # Intermediate latent (W) dimensionality.
        num_layers      = 8,        # Number of mapping layers.
        embed_features  = 128,     # Label embedding dimensionality, None = same as w_dim.
        layer_features  = 256,     # Number of intermediate features in the mapping layers, )
        **kwagrs):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_layers = num_layers

        if embed_features is None:
            embed_features = w_dim
        if c_dim == 0:
            embed_features = 0
        if layer_features is None:
            layer_features = w_dim
        features_list = [z_dim + embed_features] + [layer_features] * (num_layers - 1) + [w_dim]

        if c_dim > 0:
            self.embed = nn.Linear(c_dim, embed_features)
        blocks = []
        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            blocks += [nn.Linear(in_features, out_features),
            nn.ReLU()
            ]
        self.main = nn.Sequential(*blocks)


    def forward(self, z, c = None):
        # Embed, normalize, and concat inputs.
        # z_debug = z.detach().cpu().numpy()
        # c_debug = c.detach().cpu().numpy()
        #x = normalize_2nd_moment(
        # x = normalize_2nd_moment(self.embedz(z.to(torch.float32)))
        # z = F.normalize(z, dim=-1)
        x = None
        if self.z_dim > 0:
            x = normalize_2nd_moment(z.to(torch.float32))
        if self.c_dim > 0:
            y = normalize_2nd_moment(self.embed(c.to(torch.float32)))
            x = torch.cat([x, y], dim=-1) if x is not None else y

        # Main layers.
        if self.num_layers > 0:
            x = self.main(x)
        # x_debug = x.detach().cpu().numpy()
        return x

    
class SynthesisNetwork(torch.nn.Module):
    def __init__(self,
        img_size,             # Output image resolution.
        w_dim = 64,                      # Intermediate latent (W) dimensionality.
        img_channels = 3,               # Number of color channels.
        channel_max     = 512,      # Maximum number of channels in any layer.
        use_final_sigmoid = False,
        **block_kwargs,             # Arguments for SynthesisBlock.
    ):
        # assert img_size >= 4 and img_size & (img_size - 1) == 0
        super(SynthesisNetwork,self).__init__()
        self.w_dim = w_dim
        self.use_final_sigmoid = use_final_sigmoid

        blocks = []

        self.num_ws = 0
        self.h, self.w = img_size[0],img_size[1]
        repeat_num = int(np.log2(self.w)) - 3
        self.ws, self.hs = self.compute_inital_image_size(h = self.h, w = self.w, num_up_layers = repeat_num)

        dim_in = 2**14 // 256

        self.out = nn.Sequential(
            nn.InstanceNorm2d(dim_in, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_in, img_channels, 1, 1, 0))
        
        #self.const = nn.Embedding(1, dim_in * img_size//(2**repeat_num) * img_size //(2**repeat_num))
        
        for i in range(repeat_num):
            dim_out = min(2 * dim_in,  channel_max)
            blocks.insert(0,AdainResBlk(dim_out, dim_in, w_dim,
                               w_hpf=0, upsample=True))  # stack-like
            dim_in = dim_out

        for i in range(1):
            blocks.insert(0, AdainResBlk(dim_out, dim_out, w_dim,
                               w_hpf=0)) # stack-like

        self.decode = nn.Sequential(*blocks)
        self.const = torch.nn.Parameter(torch.randn(
            [1, dim_out, self.hs, self.ws]))

        
    def compute_inital_image_size(self, h, w, num_up_layers = 4):
        sw = w // (2**num_up_layers)
        sh = round(sw / (w / h))
        return sw, sh

    def forward(self, ws, **block_kwargs):
        #x = torch.ones([ws.shape[0], 1])
        x = self.const
        for block in self.decode:
            x = block(x, ws)
        x = F.interpolate(x, size=(self.h, self.w))
        x = self.out(x)

        # if self.use_final_sigmoid:
        #     x = torch.sigmoid(x)
        # a = x.detach().cpu().numpy()
        return x

    

# class MinibatchStdLayer(torch.nn.Module):
#     def __init__(self, group_size, num_channels=1):
#         super().__init__()
#         self.group_size = group_size
#         self.num_channels = num_channels

#     def forward(self, x):
#         N, C, H, W = x.shape
#         with misc.suppress_tracer_warnings(): # as_tensor results are registered as constants
#             G = torch.min(torch.as_tensor(self.group_size), torch.as_tensor(N)) if self.group_size is not None else N
#         F = self.num_channels
#         c = C // F

#         y = x.reshape(G, -1, F, c, H, W)    # [GnFcHW] Split minibatch N into n groups of size G, and channels C into F groups of size c.
#         y = y - y.mean(dim=0)               # [GnFcHW] Subtract mean over group.
#         y = y.square().mean(dim=0)          # [nFcHW]  Calc variance over group.
#         y = (y + 1e-8).sqrt()               # [nFcHW]  Calc stddev over group.
#         y = y.mean(dim=[2,3,4])             # [nF]     Take average over channels and pixels.
#         y = y.reshape(-1, F, 1, 1)          # [nF11]   Add missing dimensions.
#         y = y.repeat(G, 1, H, W)            # [NFHW]   Replicate over group and pixels.
#         x = torch.cat([x, y], dim=1)        # [NCHW]   Append to input as new channels.
#         return x

#     def extra_repr(self):
#         return f'group_size={self.group_size}, num_channels={self.num_channels:d}'


# class DiscriminatorEpilogue(torch.nn.Module):
#     def __init__(self,
#         in_channels,                    # Number of input channels.
#         cmap_dim,                       # Dimensionality of mapped conditioning label, 0 = no label.
#         resolution,                     # Resolution of this block.
#         img_channels,                   # Number of input color channels.
#         architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
#         mbstd_group_size    = 4,        # Group size for the minibatch standard deviation layer, None = entire minibatch.
#         mbstd_num_channels  = 1,        # Number of features for the minibatch standard deviation layer, 0 = disable.
#         activation          = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
#         conv_clamp          = None,     # Clamp the output of convolution layers to +-X, None = disable clamping.
#     ):
#         # assert architecture in ['orig', 'skip', 'resnet']
#         super().__init__()
#         self.in_channels = in_channels
#         self.cmap_dim = cmap_dim
#         self.resolution = resolution
#         self.img_channels = img_channels
#         self.architecture = architecture

#         # if architecture == 'skip':
#         #     self.fromrgb = Conv2dLayer(img_channels, in_channels, kernel_size=1, activation=activation)
#         self.mbstd = MinibatchStdLayer(group_size=mbstd_group_size, num_channels=mbstd_num_channels) if mbstd_num_channels > 0 else None
#         self.conv = Conv2dLayer(in_channels + mbstd_num_channels, in_channels, kernel_size=3, activation=activation, conv_clamp=conv_clamp)
#         self.fc = FullyConnectedLayer(in_channels * (resolution ** 2), in_channels, activation=activation)
#         self.out = FullyConnectedLayer(in_channels, 1 if cmap_dim == 0 else cmap_dim)

#     def forward(self, x, img, cmap, force_fp32=False):
#         # misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution]) # [NCHW]
#         _ = force_fp32 # unused
#         dtype = torch.float32
#         memory_format = torch.contiguous_format

#         # FromRGB.
#         x = x.to(dtype=dtype, memory_format=memory_format)
#         # if self.architecture == 'skip':
#         #     misc.assert_shape(img, [None, self.img_channels, self.resolution, self.resolution])
#         #     img = img.to(dtype=dtype, memory_format=memory_format)
#         #     x = x + self.fromrgb(img)

#         # Main layers.
#         if self.mbstd is not None:
#             x = self.mbstd(x)
#         x = self.conv(x)
#         x = self.fc(x.flatten(1))
#         x = self.out(x)

#         # Conditioning.
#         if self.cmap_dim > 0:
#             # misc.assert_shape(cmap, [None, self.cmap_dim])
#             x = (x * cmap).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))

#         assert x.dtype == dtype
#         return x


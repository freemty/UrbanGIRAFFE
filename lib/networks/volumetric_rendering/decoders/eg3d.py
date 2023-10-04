import torch
import torch.nn as nn
import torch.nn.functional as F

from numpy import pi
import numpy as np
import math
from tools.kitti360Scripts.helpers.labels import name2label

def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()


def generate_planes():
    """
    Defines planes by the three vectors that form the "axes" of the
    plane. Should work with arbitrary number of planes and planes of
    arbitrary orientation.
    """
    return torch.tensor([[[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]],
                            [[1, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0]],
                            [[0, 0, 1],
                            [1, 0, 0],
                            [0, 1, 0]]], dtype=torch.float32)

def project_onto_planes(planes, coordinates):
    """
    Does a projection of a 3D point onto a batch of 2D planes,
    returning 2D plane coordinates.

    Takes plane axes of shape n_planes, 3, 3
    # Takes coordinates of shape N, M, 3
    # returns projections of shape N*n_planes, M, 2
    """
    N, M, C = coordinates.shape
    n_planes, _, _ = planes.shape
    coordinates = coordinates.unsqueeze(1).expand(-1, n_planes, -1, -1).reshape(N*n_planes, M, 3)
    inv_planes = torch.linalg.inv(planes).unsqueeze(0).expand(N, -1, -1, -1).reshape(N*n_planes, 3, 3)
    projections = torch.bmm(coordinates, inv_planes)
    # a = projections.detach().cpu().numpy()
    return projections[..., :2]


def sample_from_planes(plane_axes, plane_features, coordinates, mode='bilinear', padding_mode='zeros', box_warp=1.1):
    assert padding_mode == 'zeros'
    B, n_planes, C, H, W = plane_features.shape
    _, M, _ = coordinates.shape
    plane_features = plane_features.reshape(B*n_planes, C, H, W)
    # plane_features = plane_features.view(B*n_planes, C, H, W)
    # a = plane_features.detach().cpu().numpy()

    coordinates = (2/box_warp) * coordinates # TODO: add specific box bounds

    projected_coordinates = project_onto_planes(plane_axes, coordinates).unsqueeze(1)
    # projected_coordinates = projected_coordinates.detach().cpu().numpy()
    output_features = torch.nn.functional.grid_sample(plane_features, projected_coordinates.float(), mode=mode, padding_mode=padding_mode, align_corners=False).permute(0, 3, 2, 1).reshape(B, n_planes, M, C)
    return output_features
class EG3DDecoder(torch.nn.Module):
    def __init__(self, 
    n_features = 48, 
    rgb_out_dim = 3, 
    feature_out_dim = 64,
    triplane_kwargs = {}, **kwargs):
        super().__init__()
        self.hidden_dim = 128

        # self.triplane_generator = TriplaneGenerator(**triplane_kwargs)
        self.plane_axes = generate_planes()

        self.net = torch.nn.Sequential(
            nn.Linear(n_features, self.hidden_dim),
            torch.nn.Softplus(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.Softplus(),
        )
        self.out_c = nn.Linear(self.hidden_dim, rgb_out_dim)
        self.out_s=  nn.Linear(self.hidden_dim, 1)
        self.out_f= nn.Linear(self.hidden_dim, feature_out_dim)
        
    def forward(self, z, ray_d = None, pts = None):
        # Aggregate features
        z = z.reshape((z.shape[0], -1))
        x = z

        # N, M, C = x.shape
        # x = x.view(N*M, C)

        x = self.net(x)
        # x = x.view(N, M, -1)
        rgb = torch.sigmoid(self.out_c(x)) # Uses sigmoid clamping from MipNeRF
        feat = self.out_f(x)
        feat = torch.cat((feat, rgb), dim = -1)
        sigma =self.out_s(x)
        return feat, sigma

    # def get_features(self, z, c, p):
    #     planes = self.triplane_generator(z, c)
    #     planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
    #     sampled_features = sample_from_planes(plane_axes = self.plane_axes.to(p.device), plane_features = planes, coordinates = p, box_warp = 1.1)

    #     return sampled_features


class TriplaneGenerator(torch.nn.Module):
    def __init__(self,
        z_dim = 512,                      # Input latent (Z) dimensionality.
        c_dim = 16,                      # Conditioning label (C) dimensionality.
        w_dim = 512,                      # Intermediate latent (W) dimensionality.
        img_resolution = 64,             # Output resolution.
        img_channels = 48,               # Number of output color channels.
        mapping_network_kwargs = {},         # Arguments for SynthesisNetwork.
        synthesis_network_kwargs = {}
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.plane_axse = generate_planes()
        self.synthesis = SynthesisNetwork(w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels, **synthesis_network_kwargs)
        self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim,** mapping_network_kwargs)


    def forward(self, z, c, p = None, semantic = None,  **synthesis_kwargs):
        ws = self.mapping(z, c)
        planes = self.synthesis(ws, bbox_semantic = semantic , **synthesis_kwargs)
        if p == None:
            return planes
        else:
            planes = planes.view(len(planes), 3, -1, planes.shape[-2], planes.shape[-1])
            sampled_features = sample_from_planes(plane_axes = self.plane_axse.to(p.device), plane_features = planes, coordinates = p, box_warp = 1.1)

            return sampled_features
        



#---------------
class MappingNetwork(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality, 0 = no latent.
        c_dim,                      # Conditioning label (C) dimensionality, 0 = no label.
        w_dim,                      # Intermediate latent (W) dimensionality.
        num_layers      = 8,        # Number of mapping layers.
        embed_features  = 512,     # Label embedding dimensionality, None = same as w_dim.
        layer_features  = 512,     # Number of intermediate features in the mapping layers, )
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

        if c_dim > 0:
            self.embedc = nn.Linear(c_dim, embed_features // 2)
        self.embedz = nn.Linear(z_dim, embed_features // 2)
        features_list = [embed_features] + [layer_features] * (num_layers - 1) + [w_dim]

        blocks = []
        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            blocks += [nn.Linear(in_features, out_features),
            nn.ReLU()
            ]
        self.main = nn.Sequential(*blocks)


    def forward(self, z, c):
        # Embed, normalize, and concat inputs.
        z_debug = z.detach().cpu().numpy()
        # c_debug = c.detach().cpu().numpy()
        #x = normalize_2nd_moment(
        x = normalize_2nd_moment(self.embedz(z.to(torch.float32)))
        if self.c_dim > 0:
            y = normalize_2nd_moment(self.embedc(c.to(torch.float32)))
            # y = self.embedc(c.to(torch.float32))
            x = torch.cat([x, y], dim=1) if x is not None else y
        # Main layers.
        x = self.main(x)
        # x_debug = x.detach().cpu().numpy()
        return x


class SynthesisNetwork(torch.nn.Module):
    def __init__(self,
        w_dim = 64,                      # Intermediate latent (W) dimensionality.
        img_size = 64,             # Output image resolution.
        img_channels = 96,               # Number of color channels.
        channel_max     = 512,      # Maximum number of channels in any layer.
        use_semantic_aware_output = True,
        semantic_list = ['car', 'building'],
        **kwargs            # Arguments for SynthesisBlock.
    ):
        assert img_size >= 4 and img_size & (img_size - 1) == 0
        super().__init__()
        self.w_dim = w_dim
        self.img_size = img_size
        self.triplane_channels = img_channels
        blocks = []
        self.use_semantic_aware_output = use_semantic_aware_output

        self.num_ws = 0

        repeat_num = int(np.log2(img_size)) - 2
        dim_in = min(img_channels * 2**repeat_num, channel_max)
        self.const = torch.nn.Parameter(torch.randn(
            [1, dim_in, img_size//(2**repeat_num), img_size //(2**repeat_num)]))
        #self.const = nn.Embedding(1, dim_in * img_size//(2**repeat_num) * img_size //(2**repeat_num))
        
        for i in range(repeat_num):
            dim_out = min(img_channels * 2**(repeat_num - i),  channel_max)
            blocks.append(AdainResBlk(dim_in, dim_out, w_dim,
                               w_hpf=0, upsample=True))  # stack-like
            dim_in = dim_out

        self.decode = nn.Sequential(*blocks)

        if use_semantic_aware_output:
            self.semantic_list = semantic_list
            self.out = nn.Sequential(
                nn.InstanceNorm2d(dim_in, affine=True),
                nn.LeakyReLU(0.2),
                nn.Conv2d(dim_in, self.triplane_channels * len(self.semantic_list), 1, 1, 0))
        else:
            self.out = nn.Sequential(
                nn.InstanceNorm2d(dim_in, affine=True),
                nn.LeakyReLU(0.2),
                nn.Conv2d(dim_in, self.triplane_channels, 1, 1, 0))

    def forward(self, ws, bbox_semantic = None, **block_kwargs):
        block_ws = []
        #x = torch.ones([ws.shape[0], 1])
        x = self.const
        for block in self.decode:
            x = block(x, ws)

        if self.use_semantic_aware_output:
            x = self.out(x)
            out = torch.zeros_like(x)[:,:self.triplane_channels]
            for i, sname in enumerate(self.semantic_list): 
                # a = x[bbox_category == i,i*self.triplane_channels:(i+1)*self.triplane_channels]
                out[bbox_semantic == name2label[sname].id] = x[bbox_semantic == name2label[sname].id,i*self.triplane_channels:(i+1)*self.triplane_channels]
                # b = out.detach().cpu().numpy()
        else: 
            out = self.out(x)
        
        return out


class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample=False):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)


    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance


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



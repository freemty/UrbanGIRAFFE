from tkinter import N
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.config import cfg
from numpy import pi

from lib.networks.volumetric_rendering.decoders.layers import ModulationLinear, PositionalEncoding


class RoadDecoder(nn.Module):
    """NeRF MLP with style modulation.

    This module maps input latent codes, coordinates, and viewing directions to alpha values and an output
    feature vector. Conventionally the output is a 3 dimensional RGB colour vector, but more general features can
    be output to be used for downstream upsampling and refinement.

    Note that skip connections are important for training any model with more than 4 layers, as shown by DeepSDF.

    Args:
    ----
    n_layers: int
        Number of layers in the MLP (excluding those for predicting alpha and the feature output.)
    channels: int
        Channels per layer.
    out_channel: int
        Output channels.
    z_dim: int
        Dimension of latent code.
    omega_coord: int
        Number of frequency bands to use for coordinate positional encoding.
    omega_dir: int
        Number of frequency bands to use for view direction positional encoding.
    skips: list
        Layers at which to apply skip connections. Coordinates will be concatenated to feature inputs at these
        layers.

    """

    def __init__(self, 
    n_layers=8, 
    channels=256, 
    out_channel=3, 
    z_dim=128, 
    use_viewdirs=True, 
    n_freq_posenc=10,
    n_freq_posenc_views=4,
    skips=[4],
    use_final_sigmoid = True,
    **kwargs):
        super().__init__()

        self.use_final_sigmoid = use_final_sigmoid
        self.skips = skips

        self.from_coords = PositionalEncoding(in_dim=3, frequency_bands=n_freq_posenc)
        self.from_dirs = PositionalEncoding(in_dim=3, frequency_bands=n_freq_posenc_views)
        self.n_layers = n_layers

        self.layers = nn.ModuleList(
            [ModulationLinear(in_channel=self.from_coords.out_dim, out_channel=channels, z_dim=z_dim)]
        )

        for i in range(1, n_layers):
            if i in skips:
                in_channels = channels + self.from_coords.out_dim
            else:
                in_channels = channels
            self.layers.append(ModulationLinear(in_channel=in_channels, out_channel=channels, z_dim=z_dim))

        self.fc_alpha = ModulationLinear(
            in_channel=channels, out_channel=1, z_dim=z_dim, demodulate=False, activate=False
        )
        self.fc_feat = ModulationLinear(in_channel=channels, out_channel=channels, z_dim=z_dim)
        self.fc_viewdir = ModulationLinear(
            in_channel=channels + self.from_dirs.out_dim, out_channel=channels, z_dim=z_dim
        )
        self.fc_out = ModulationLinear(
            in_channel=channels, out_channel=out_channel, z_dim=z_dim, demodulate=False, activate=False
        )

    def process_latents(self, z):
        # output should be list with separate latent code for each conditional layer in the model
        # should be a list

        if isinstance(z, list):  # latents already in proper format
            pass
        elif z.ndim == 2:  # standard training, shape [B, n_p ,ch]
            z = [z] * (self.n_layers + 4)
        elif z.ndim == 3:  # latent optimization, shape [B, n_latent_layers, ch]
            n_latents = z.shape[1]
            z = [z[:, i] for i in range(n_latents)]
        return z

    def forward(self, z, coords, viewdirs=None):
        """Forward pass.

        Input:
        -----
        z: torch.Tensor
            Latent codes of shape [B, z_dim].
        coords: torch.Tensor
            Spatial coordinates of shape [B, 3].
        viewdirs: torch.Tensor
            View directions of shape [B, 3].

        Return:
        ------
        out: torch.Tensor
            RGB pixels or feature vectors of shape [B, out_channel].
        alpha: torch.Tensor
            Occupancy values of shape [B, 1].

        """
        coords = self.from_coords(coords)
        z = self.process_latents(z)

        h = coords
        for i, layer in enumerate(self.layers):
            if i in self.skips:
                h = torch.cat([h, coords], dim=-1)

            h = layer(h, z[i])

        alpha = self.fc_alpha(h, z[i + 1])
        #alpha = torch.sigmoid(alpha)

        if viewdirs is None:
            return None, alpha

        h = self.fc_feat(h, z[i + 2])

        viewdirs = self.from_dirs(viewdirs)
        h = torch.cat([h, viewdirs], dim=-1)

        h = self.fc_viewdir(h, z[i + 3])
        out = self.fc_out(h, z[i + 4])
        
        
        if self.use_final_sigmoid:
            out = torch.sigmoid(out)    
        return out, alpha





class stuffDecoder(nn.Module):
    r""" MLP"""

    def __init__(self, 
                in_channel = 64,
                # out_channels_s=1, 
                # out_channels_c=3, 
                out_channel= 64,
                style_channel = 64,
                n_block_num = 2,
                hidden_channels=256,
                semantic_dim=16,
                use_pts= True,
                use_viewdirs=False,
                use_seg = False,
                use_final_sigmoid=True,
                use_positonal_encoding=False,
                use_density_modulate= False,
                n_freq_posenc=4,
                n_freq_posenc_pts=10,
                n_freq_posenc_views=4,
                ):
        super( stuffDecoder, self).__init__()
        assert n_block_num >= 2
        self.in_channels = in_channel
        self.use_final_sigmoid = use_final_sigmoid
        self.use_seg= use_seg
        self.use_pts = use_pts
        self.use_viewdirs = use_viewdirs
        self.use_posotonal_encoding = use_positonal_encoding
        self.use_density_modulate = use_density_modulate
        self.n_block_num = n_block_num
        if use_seg:
            self.fc_m_a = nn.Linear(semantic_dim, hidden_channels)


        self.pe_pts = PositionalEncoding(in_dim=3, frequency_bands=n_freq_posenc_pts, include_input= False)
        self.pe_z = PositionalEncoding(in_dim=self.in_channels//2, frequency_bands=n_freq_posenc, include_input= False)
        self.pe_dirs = PositionalEncoding(in_dim=3, frequency_bands=n_freq_posenc_views, include_input= False)

        #self.fc_viewdir = None
        self.fc_viewdir = nn.Linear(self.pe_dirs.out_dim, hidden_channels)

        # self.fc_in = nn.Linear(self.pe_x.out_dim + self.pe_pts.out_dim * use_pts + (in_channels // 2) + use_seg * semantic_dim, hidden_channels)
        # self.fc_in = nn.Linear(self.pe_x.out_dim + self.in_channels//2 + use_seg * semantic_dim, hidden_channels)


        
        if use_positonal_encoding:
            z_in_dim = self.in_channels//2 + self.pe_z.out_dim + semantic_dim * use_seg
            self.fc_z = nn.Linear(z_in_dim, hidden_channels // 2)
            self.fc_pts = nn.Linear(self.pe_pts.out_dim, hidden_channels // 2)
        else:
            z_in_dim = self.in_channels
            self.fc_z = nn.Linear(z_in_dim, hidden_channels // 2)
            self.fc_pts = nn.Linear(3, hidden_channels // 2)
        
        self.fc_main = nn.ModuleList([])
       
        for i in range(n_block_num):
            self.fc_main.append(nn.Linear(hidden_channels, hidden_channels))

        self.fc_sigma = nn.Linear(hidden_channels, 1)

        # Feat part
        if self.use_viewdirs:
            in_dim_dir = self.pe_dirs.out_dim if use_viewdirs else 3
            self.fc_dir = nn.Linear(in_dim_dir, hidden_channels)

        self.fc_feat = nn.Linear(hidden_channels, out_channel)
        self.fc_rgb = nn.Linear(out_channel, 3)

        self.act = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, w, pts = None, raydir = None, seg = None):
        r""" Forward network

        Args:
            z (N x H x W x M x in_channels tensor): Projected features.
            raydir (N x H x W x 1 x viewdir_dim tensor): Ray directions.
            seg (N x H x W x M x mask_dim tensor): One-hot segmentation maps.
        """
        z = w
        p_num, c_in = z.size()

        # partial position endoding
        if self.use_posotonal_encoding:
            z = torch.cat((self.pe_z(z[...,:self.in_channels//2]), z[...,self.in_channels//2:]), dim = -1)
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

        # rgb = self.fc_rgb(feat)
        # if self.use_final_sigmoid:
        #     rgb= torch.sigmoid(rgb)
        # feat = torch.cat((feat, rgb), dim = -1)
        return feat ,sigma


class StyleDecoder2D(nn.Module):
    """NeRF MLP with style modulation.

    This module maps input latent codes, coordinates, and viewing directions to alpha values and an output
    feature vector. Conventionally the output is a 3 dimensional RGB colour vector, but more general features can
    be output to be used for downstream upsampling and refinement.

    Note that skip connections are important for training any model with more than 4 layers, as shown by DeepSDF.

    Args:
    ----
    n_layers: int
        Number of layers in the MLP (excluding those for predicting alpha and the feature output.)
    channels: int
        Channels per layer.
    out_channel: int
        Output channels.
    z_dim: int
        Dimension of latent code.
    omega_coord: int
        Number of frequency bands to use for coordinate positional encoding.
    omega_dir: int
        Number of frequency bands to use for view direction positional encoding.
    skips: list
        Layers at which to apply skip connections. Coordinates will be concatenated to feature inputs at these
        layers.

    """

    def __init__(self, 
    n_layers=8, 
    channels=256, 
    out_channel=3, 
    z_dim=128, 
    n_freq_posenc=10,
    skips=[4],
    **kwargs):
        super().__init__()

        # self.use_final_sigmoid = use_final_sigmoid
        self.skips = skips

        self.from_coords = PositionalEncoding(in_dim=2, frequency_bands=n_freq_posenc)
        # self.from_dirs = PositionalEncoding(in_dim=3, frequency_bands=n_freq_posenc_views)
        self.n_layers = n_layers

        self.layers = nn.ModuleList(
            [ModulationLinear(in_channel=self.from_coords.out_dim, out_channel=channels, z_dim=z_dim)]
        )

        for i in range(1, n_layers):
            if i in skips:
                in_channels = channels + self.from_coords.out_dim
            else:
                in_channels = channels
            self.layers.append(ModulationLinear(in_channel=in_channels, out_channel=channels, z_dim=z_dim))

        # self.fc_alpha = ModulationLinear(
        #     in_channel=channels, out_channel=1, z_dim=z_dim, demodulate=False, activate=False
        # )
        self.fc_feat = ModulationLinear(in_channel=channels, out_channel=channels, z_dim=z_dim)
        # self.fc_viewdir = ModulationLinear(
        #     in_channel=channels + self.from_dirs.out_dim, out_channel=channels, z_dim=z_dim
    
        self.fc_out = ModulationLinear(
            in_channel=channels, out_channel=out_channel, z_dim=z_dim, demodulate=False, activate=False
        )

    def process_latents(self, z):
        # output should be list with separate latent code for each conditional layer in the model
        # should be a list

        if isinstance(z, list):  # latents already in proper format
            pass
        elif z.ndim == 2:  # standard training, shape [B, n_p ,ch]
            z = [z] * (self.n_layers + 4)
        elif z.ndim == 3:  # latent optimization, shape [B, n_latent_layers, ch]
            n_latents = z.shape[1]
            z = [z[:, i] for i in range(n_latents)]
        return z

    def forward(self, z, pts, viewdirs=None):
        """Forward pass.

        Input:
        -----
        z: torch.Tensor
            Latent codes of shape [B, z_dim].
        coords: torch.Tensor
            Spatial coordinates of shape [B, 2].
        viewdirs: torch.Tensor
            View directions of shape [B, 3].

        Return:
        ------
        out: torch.Tensor
            RGB pixels or feature vectors of shape [B, out_channel].
        alpha: torch.Tensor
            Occupancy values of shape [B, 1].

        """
        pts = self.from_coords(pts)
        z = self.process_latents(z)

        h = pts
        for i, layer in enumerate(self.layers):
            if i in self.skips:
                h = torch.cat([h, pts], dim=-1)

            h = layer(h, z[i])

        h = self.fc_feat(h, z[i + 1])
        rgb = self.fc_out(h, z[i + 2])

        if True: 
            rgb = torch.sigmoid(rgb)
        
        return rgb




# class LocalFeatureDecoder(nn.Module):
#     r""" MLP"""

#     def __init__(self, in_channels = 64,
#                 out_channels_s=1, 
#                 out_channels_c=3, 
#                 out_channels_feat = 64,
#                 hidden_channels=256,
#                 semantic_dim=16,
#                 use_viewdirs=False,
#                 use_seg = False,
#                 use_final_sigmoid=True,
#                 n_freq_posenc=10,
#                 n_freq_posenc_views=4,
#                 ):
#         super(LocalFeatureDecoder, self).__init__()

#         self.use_final_sigmoid = use_final_sigmoid
#         self.use_seg= use_seg
#         self.use_viewdirs = use_viewdirs
#         if use_seg:
#             self.fc_m_a = nn.Linear(semantic_dim, hidden_channels)

#         self.pe_x = PositionalEncoding(in_dim=24, frequency_bands=n_freq_posenc, include_input= False)
#         self.pe_dirs = PositionalEncoding(in_dim=3, frequency_bands=n_freq_posenc_views, include_input= False)

#         #self.fc_viewdir = None
#         self.fc_viewdir = nn.Linear(self.pe_dirs.out_dim, hidden_channels)

#         self.fc_1 = nn.Linear(self.pe_x.out_dim + (in_channels - 24) + use_seg * semantic_dim, hidden_channels)

#         self.fc_2 = nn.Linear(hidden_channels, hidden_channels)

#         self.fc_3 = nn.Linear(hidden_channels, hidden_channels)

#         self.fc_4 = nn.Linear(hidden_channels, hidden_channels)

#         self.fc_sigma = nn.Linear(hidden_channels, out_channels_s)

#         # if viewdir_dim > 0:
#         #     self.fc_5 = nn.Linear(hidden_channels, hidden_channels, bias=False)
#         #     self.mod_5 = AffineMod(hidden_channels, style_dim, mod_bias=True)
#         # else:
#         self.fc_5 =nn.Linear(hidden_channels, hidden_channels)
#         self.fc_6 =  nn.Linear(hidden_channels, hidden_channels)
#         self.fc_out_c = nn.Linear(hidden_channels, out_channels_c)
#         self.fc_out_feat = nn.Linear(hidden_channels, out_channels_feat)

#         self.act = nn.LeakyReLU(negative_slope=0.2)

#     def forward(self, x, raydir = None, seg = None):
#         r""" Forward network

#         Args:
#             x (N x H x W x M x in_channels tensor): Projected features.
#             raydir (N x H x W x 1 x viewdir_dim tensor): Ray directions.
#             seg (N x H x W x M x mask_dim tensor): One-hot segmentation maps.
#         """
#         p_num, c_in = x.size()
        
#         # partial position endoding
#         f = torch.cat((self.pe_x(x[...,:24]), x[...,24:]), dim = -1)
#         if self.use_seg:
#             f =  torch.cat((f, seg), dim = -1)
#         f = self.fc_1(f)
#         # Common MLP
#         f = self.act(f)
#         f = self.act(self.fc_2(f))
#         f = self.act(self.fc_3(f))
#         f = self.act(self.fc_4(f))

#         # Sigma MLP
#         sigma = self.fc_sigma(f)

#         # Color MLP
#         if self.use_viewdirs:
#             r =self.pe_dirs(raydir)
#             f = self.fc_5(f)
#             f = f + self.fc_viewdir(r)
#         else:
#             f = self.act(self.fc_5(f))
#         f = self.act(self.fc_6(f))
#         rgb = self.fc_out_c(f)
#         feat = self.fc_out_feat(f)

#         #sigma = torch.sigmoid(sigma)
#         if self.use_final_sigmoid:
#             rgb= torch.sigmoid(rgb)
#         feat = torch.cat((feat, rgb), dim = -1)
#         return sigma, feat
    

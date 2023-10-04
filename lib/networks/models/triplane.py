import torch
import torch.nn as nn
import torch.nn.functional as F
from .stylegan2_generator import StyleGAN2Generator

class TriPlaneGenerator(torch.nn.Module):
    def __init__(self,
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim = 0,                  # Conditioning label (C) dimensionality.
        w_dim = 512,                # Intermediate latent (W) dimensionality.
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        **synthesis_kwargs,         # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim=z_dim
        self.c_dim=c_dim
        self.w_dim=w_dim
        self.img_resolution=img_resolution
        self.img_channels=img_channels
        self.generator = StyleGAN2Generator(
            z_dim = z_dim, 
            # c_dim = c_dim, 
            w_dim =w_dim, 
            resolution=img_resolution,
            image_channels=img_channels * 3)    
        self._last_planes = None


    def forward(self, z, c, truncation_psi=1, **synthesis_kwargs):
        # Render a batch of generated images.
        triplane = self.generator(z, c)
        return triplane
from email.mime import image
from random import sample
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
import numpy as np

from lib.config import cfg

from lib.utils.img_utils import save_tensor_img
from lib.networks.reference.stylegan2 import MappingNetwork, SynthesisNetwork



class Generator(nn.Module):
    '''
    2D StyleGAN Generator
    '''
    def __init__(self,
        # image_size = (94,352),
        z_dim_global =  64,
        **kwargs):        
        super(Generator, self).__init__()
        #self.ray_sampler = RaySampler()
        image_size = (int(cfg.ratio * cfg.img_size_raw[0]), int(cfg.ratio * cfg.img_size_raw[1]))
        if cfg.use_cuda: 
            if cfg.distributed:
                self.device = torch.device('cuda:{}'.format(cfg.local_rank))
            else:
                self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.z_dim = z_dim_global
        # self.mapping = MappingNetwork(**kwargs['mapping_network_kwargs'])
        # self.synthesis = SynthesisNetwork(img_size = image_size, **kwargs['synthesis_network_kwargs'])

    def sample_z(self, size, to_device=True, tmp=1.):
        z = torch.randn(*size) * tmp
        if to_device:
            z = z.to(self.device)
        return z

    def forward(self, raw_batch, data_type  = 'gen',mode = 'train'):
        output = {}
        batch_size =  raw_batch['rgb'].shape[0]

        if 'z_global' in raw_batch.keys():
            z = raw_batch['z_global']
        else:
            z = self.sample_z((batch_size, self.z_dim))
        ws = self.mapping(z)
        # a = ws.detach().cpu().numpy()
        image = self.synthesis(ws)
        
        output['masked_rgb'] = output['rgb'] = image
        
        return output

from email.mime import image
from random import sample
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from lib.config import cfg

from lib.utils.img_utils import save_tensor_img
from lib.networks.volumetric_rendering.decoders import SPADEGenerator2D, StyleDecoder2D
from lib.networks.volumetric_rendering.sampling import sample_from_3dgrid



class Generator(nn.Module):
    '''
    2D StyleGAN Generator
    '''
    def __init__(self,
        image_size = (94,352),
        z_global_dim =  64,
        feature_dim = 16,
        condition_type = 'adaIN',
        feature_type = 'SPADE',
        pts_type = 'global',
        **kwargs):        
        super(Generator, self).__init__()
        #self.ray_sampler = RaySampler()
        self.feature_type = feature_type
        self.pts_type = pts_type


        if cfg.use_cuda: 
            if cfg.distributed:
                self.device = torch.device('cuda:{}'.format(cfg.local_rank))
            else:
                self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.z_dim = z_global_dim
        self.feature_dim = feature_dim
        self.feature_map_gernerator = SPADEGenerator2D(opt = kwargs['feature_map_generator_kwargs'])
        self.decoder = StyleDecoder2D(**kwargs['2d_decoder_kwargs'])

        self.n_semantic = 46
        self.semantic_embedding = nn.Embedding(self.n_semantic, int(self.feature_dim))

    def sample_z(self, size, to_device=True, tmp=1.):
        z = torch.randn(*size) * tmp
        if to_device:
            z = z.to(self.device)
        return z
    

    def get_feature_map(self, batch):
        '''
        use_seg 
        use_z_vec
        use_feature_map
        use_z_spade
        '''
        batch_size =  batch['idx'].shape[0]

        seg = batch['semantic'].to(torch.int64)

        seg_one_hot = F.one_hot(seg , num_classes = self.n_semantic).permute(0,3,1,2).to(torch.float32)
        # plt.imsave('tmp/aaaa.jpg',  seg[0].detach().cpu().numpy())
        

        if 'z_global' in batch.keys():
            z = batch['z_global']
        else:
            z = self.sample_z((batch_size, 1, self.z_dim))

        if self.feature_type == 'z':
            feature_map = z.reshape((batch_size,1,1,-1)).repeat((1,94,352,1))[...,:self.feature_dim]
        elif self.feature_type == 'seg':
            feature_map = self.semantic_embedding(seg)
            feature_map = feature_map.to(torch.float32).permute((0,3,1,2))
        elif self.feature_type == 'SPADE':
            feature_map = self.feature_map_gernerator(seg_one_hot, z, False)
        elif self.feature_type == 'SPADE_free':
            seg_one_hot = torch.zeros_like(seg_one_hot)
            feature_map = self.feature_map_gernerator(seg_one_hot, z, False)
        elif self.feature_type  == 'SPADE_hybrid':
            feature_map = self.feature_map_gernerator(seg_one_hot, z,  True)
        else: 
            raise KeyboardInterrupt
        

        # assert feature_map.shape == (batch_size, 1, 1)
        return feature_map
    
    def render_2d(self, feature_map):
        output = {}

        H, W = 94, 352

        batch_size =  feature_map.shape[0]
        X, Y = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy')
        pixels_loc = torch.cat((X[:, :, None], Y[:, :, None]), dim = -1).to(torch.float32).to(feature_map.device)
        pixels_loc = pixels_loc.unsqueeze(0).repeat((batch_size,1,1,1))
        # Render pixels and grid sample get crrsponding feature
        if True:
            pixels_loc += torch.rand_like(pixels_loc)
        pixels_loc =  2 * (pixels_loc / torch.tensor((352, 94), device=feature_map.device) - 0.5)
        # a = pixels_loc.detach().cpu().numpy()

        if self.feature_type == 'z':
            pixels_feature = feature_map
        elif self.feature_type == 'seg':
            pixels_feature = F.grid_sample(feature_map, grid = pixels_loc, mode= 'nearest', padding_mode= 'border')
        else:
            pixels_feature = F.grid_sample(feature_map, grid = pixels_loc, mode= 'bilinear', padding_mode= 'border')
        # plt.imsave('tmp/aaaa.jpg',  pixels_feature[0,0].detach().cpu().numpy())
        # assert pixels_loc.shape[:2] == (batch_size, 94, 352)
        a = pixels_feature.detach().cpu().numpy()
        a = feature_map.detach().cpu().numpy()

        pixels_feature = pixels_feature.reshape(batch_size * H * W, -1)
        if self.pts_type == 'global':
            pixels_pts =  pixels_loc.reshape(batch_size * H * W, 2)
        elif self.pts_type == 'local':
            pixels_pts = pixels_loc.reshape(batch_size * H * W, 2)
        elif self.pts_type == 'no':
            pixels_pts = torch.ones_like(pixels_loc)
            pixels_pts = pixels_pts.reshape(batch_size * H * W, 2)
        else: 
            raise KeyboardInterrupt

            

        pixels_rgb = self.decoder(pts = pixels_pts, z = pixels_feature)

        # Reformat output
        rgb = pixels_rgb.reshape(batch_size, H, W, 3).permute((0,3,1,2))
        output['masked_rgb'] = output['rgb'] = rgb
        return output
    

    def select_date(self, raw_batch, data_type):

        batch = {}

        batch['camera_mat'] = raw_batch['camera_mat'],
        if data_type == 'gen':
            batch['idx'] =  raw_batch['idx']
            batch['semantic'] = raw_batch['semantic']
        elif data_type == 'dis':
            batch['idx'] =  raw_batch['idx_fake']
            batch['semantic'] = raw_batch['semantic_fake']
        else:
            raise KeyError


        return batch

    def forward(self, batch, data_type  = 'gen',mode = 'train'):
        output = {}


        batch_valid = self.select_date(raw_batch = batch, data_type = data_type)


        feature_map = self.get_feature_map(batch_valid)
        output = self.render_2d(feature_map)
        
        return output

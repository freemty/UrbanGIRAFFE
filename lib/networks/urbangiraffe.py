from dis import dis
from cv2 import repeat
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from lib.config import cfg
#from lib.networks.decoder import Decoder, NerfStyleGenerator, DecoderEG3D
from lib.networks.GAN.generators import urbangiraffe, giraffe, ngp2d, styleGAN
from lib.networks.GAN.discriminators import conv, stylegan, stylegan2_discriminator
from lib.networks.GAN.augmentations import AdaAug

generator_dict = {
    'urbangiraffe': urbangiraffe.Generator,
    # 'urbangiraffe': urbangiraffe_new.Generator,
    'bbox2d': ngp2d.Generator,
    'stylegan': styleGAN.Generator,
    'giraffe': giraffe.Generator,
    

}
discriminator_dict = {
    # 'dc': conv.DCDiscriminator,
    # 'resnet': conv.DiscriminatorResnet,

    'StyleGAN2': stylegan2_discriminator.Discriminator,
    'stylegan_obj': stylegan.Discriminator_obj,
    'stylegan': stylegan.Discriminator,
    # 'urbangiraffe': urbangiraffe.Discrimiantor
}


class Network(nn.Module):
    '''GAN model class.

    Args:
        device (device): torch device
        discriminator (nn.Module): discriminator network
        generator (nn.Module): generator network
        generator_test (nn.Module): generator_test network
    '''
    def __init__(self):
        super(Network, self).__init__()
        kwargs = cfg.network_kwargs

        self.z_dim_global = cfg.network_kwargs.generator_kwargs.z_dim_global
        self.z_global = torch.nn.Parameter(self.sample_z((1, self.z_dim_global), tmp=.65).clamp(-1, 1))
        
        self.generator = generator_dict[kwargs.generator_type](**kwargs.generator_kwargs)
        self.generator_test = generator_dict[kwargs.generator_type](**kwargs.generator_kwargs)

        # Discriminator
        kwargs.discriminator_kwargs.image_size =(int(cfg.ratio * cfg.img_size_raw[0]), int(cfg.ratio * cfg.img_size_raw[1]))
        self.discriminator = discriminator_dict[kwargs.discriminator_type](**kwargs.discriminator_kwargs)
        if cfg.use_patch_discriminator:
            kwargs.discriminator_obj_kwargs.image_size =(cfg.patch_size,cfg.patch_size)
            kwargs.discriminator_obj_kwargs.num_domains = len(cfg.valid_object)
            self.discriminator_obj = discriminator_dict[kwargs.discriminator_obj_type](
                
                **kwargs.discriminator_obj_kwargs)

        assert cfg.aug_type in ['noaug', 'ada', 'fixed']
        if cfg.aug_type != 'noaug':
            self.add_module('aug', AdaAug(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1))
            self.aug.p = torch.tensor(cfg.ada_p_init, device=self.aug.p.device)
            if hasattr(self, 'discriminator_obj'):
                self.add_module('aug_obj', AdaAug(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1))
                self.aug_obj.p = torch.tensor(cfg.ada_p_init, device=self.aug.p.device)

            

        if cfg.local_rank == 0:
            print('Training total batch_size:%d'%(cfg.train.batch_size * cfg.world_size))
            print('G_type:%s ;D_tpye:%s'%(kwargs.generator_type,kwargs.discriminator_type))
            nparameters_d = sum(p.numel() for p in self.discriminator.parameters())
            nparameters_g = sum(p.numel() for p in self.generator.parameters())
            print('Total number of scene discriminator parameters: %d' % nparameters_d)
            print('Total number of generator parameters: %d' % nparameters_g)
            if cfg.use_patch_discriminator:
                # print(self.discriminator_obj)
                nparameters_d_obj = sum(p.numel() for p in self.discriminator_obj.parameters())
                print('Total number of object discriminator parameters: %d' % nparameters_d_obj)
                
    
    def sample_z(self, size, tmp=1.):
        z = torch.randn(*size) * tmp
        return z
    
            
    def forward(self, input_data, type = 'G', data_type = 'gen', mode = 'train',  **kwargs):
        if type == 'G':
            return self.generator(raw_batch=input_data, data_type=data_type, mode=mode )
        elif type == 'D':
            return self.discriminator(input_data)
        elif type == 'D_obj':
            return self.discriminator_obj(input_data, **kwargs)
        elif type == 'G_test':
             return self.generator_test(raw_batch=input_data, data_type=data_type, mode=mode)
        else:
            raise RuntimeError


        gen = self.generator_test
        if gen is None:
            gen = self.generator
        

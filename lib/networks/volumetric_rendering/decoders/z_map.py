import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.networks.reference.spade_network import BaseNetwork, SPADEResnetBlock

class simpleGenerator(nn.Module):
    def __init__(self, opt, **kwarg):
        
        nf = opt.ngf
        z_dim = opt.z_dim
        super(z_generator, self).__init__()
        self.nf = nf
        self.z_dim= z_dim
        self.sw, self.sh = 24,6 

        self.fc = nn.Linear(z_dim, 16 * nf * self.sw * self.sh)
        self.conv = nn.Conv2d(16 * nf, 16 * nf, kernel_size=3, padding=1)      
        self.conv_0 = nn.Conv2d(16 * nf, 8 * nf, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(8 * nf, 4 * nf, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv2d(4 * nf, 2 * nf, kernel_size=3, padding=1)
        self.conv_3 = nn.Conv2d(2 * nf, 1 * nf, kernel_size=3, padding=1)
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, z, input = None):
        batch_size = z.shape[0]

        feature_map = self.fc(z).reshape(batch_size, 16 * self.nf, self.sh, self.sw)

        feature_map = self.actvn(self.conv(feature_map)) 
        feature_map = self.actvn(self.conv_0(feature_map))
        feature_map = self.up(feature_map)
        feature_map = self.actvn(self.conv_1(feature_map))
        feature_map = self.up(feature_map)
        feature_map = self.actvn(self.conv_2(feature_map))
        feature_map = self.up(feature_map)
        feature_map = self.actvn(self.conv_3(feature_map))
        feature_map = self.up(feature_map)

        raw_H, raw_W = input.shape[2], input.shape[3]
        z_map = F.leaky_relu(feature_map, 2e-1)
        z_map = F.interpolate(z_map, size=(raw_H, raw_W))

        return z_map

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)




class SPADEGenerator(BaseNetwork):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ngf

        max_nc = self.opt.max_nc
        self.max_nc = max_nc

        self.sw, self.sh = self.compute_latent_vector_size(opt)

        self.fc = nn.Linear(opt.z_dim, min(16 * nf, max_nc) * self.sw * self.sh)

        self.head_0 = SPADEResnetBlock(min(16 * nf, max_nc), min(16 * nf, max_nc), opt)

        self.G_middle_0 = SPADEResnetBlock(min(16 * nf, max_nc), min(16 * nf, max_nc), opt)
        # self.G_middle_1 = SPADEResnetBlock(min(16 * nf, max_nc), min(16 * nf, max_nc), opt)

        self.up_0 = SPADEResnetBlock(min(16 * nf, max_nc), min(8 * nf, max_nc), opt)
        self.up_1 = SPADEResnetBlock(min(8 * nf, max_nc), min(4 * nf, max_nc), opt)
        self.up_2 = SPADEResnetBlock(min(4 * nf, max_nc), min(2 * nf, max_nc), opt)
        self.up_3 = SPADEResnetBlock(min(2 * nf, max_nc), min(1 * nf, max_nc), opt)

        final_nc = nf

        if opt.num_upsampling_layers == 'most':
            self.up_4 = SPADEResnetBlock(1 * nf, nf // 2, opt)
            final_nc = nf // 2

        #self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)

    def compute_latent_vector_size(self, opt):
        if opt.num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif opt.num_upsampling_layers == 'more':
            num_up_layers = 6
        elif opt.num_upsampling_layers == 'most':
            num_up_layers = 7
        else:
            raise ValueError('opt.num_upsampling_layers [%s] not recognized' %
                             opt.num_upsampling_layers)

        sw = opt.crop_size // (2**num_up_layers)
        sh = round(sw / opt.aspect_ratio)

        return sw, sh

    def forward(self, input, z=None):
        seg = input

        if z is None:
            z = torch.randn(input.size(0), self.opt.z_dim,
                            dtype=torch.float32, device=input.get_device())
        x = self.fc(z)
        x = x.view(-1, min(16 * self.opt.ngf, self.max_nc), self.sh, self.sw)


        x = self.head_0(x, seg)

        x = self.up(x)
        x = self.G_middle_0(x, seg)

        if self.opt.num_upsampling_layers == 'more' or \
           self.opt.num_upsampling_layers == 'most':
            x = self.up(x)

        #x = self.G_middle_1(x, seg)
        x = self.up(x)
        x = self.up_0(x, seg)
        x = self.up(x)
        x = self.up_1(x, seg)
        x = self.up(x)
        x = self.up_2(x, seg)
        x = self.up(x)
        x = self.up_3(x, seg)

        if self.opt.num_upsampling_layers == 'most':
            x = self.up(x)
            x = self.up_4(x, seg)

        if True:
            raw_H, raw_W = input.shape[2], input.shape[3]
            x = F.leaky_relu(x, 2e-1)
            x = F.interpolate(x, size=(raw_H, raw_W))
        else:
            x = self.conv_img(F.leaky_relu(x, 2e-1))
            x = F.tanh(x)
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import math
from lib.networks.reference.spade_network.base_network import BaseNetwork
from lib.networks.reference.spade_network.architecture import SPADEResnetBlock as SPADEResnetBlock
import torch.nn.utils.spectral_norm as spectral_norm


class SPADE3D(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc, nhidden = 128):
        super().__init__()

        assert config_text.startswith('spade')
        parsed = re.search('spade(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm3d(norm_nc, affine=False)
        # elif param_free_norm_type == 'syncbatch':
        #     self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm3d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = nhidden

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv3d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv3d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv3d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)
    
        # Part 2. produce scaling and bias conditioned on semantic map

        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        # segmap_debug = segmap.detach().cpu().numpy()
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out

    

class SPADE2D(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc, nhidden = 128):
        super().__init__()

        assert config_text.startswith('spade')
        parsed = re.search('spade(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        # elif param_free_norm_type == 'syncbatch':
        #     self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = nhidden

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)
    
        # Part 2. produce scaling and bias conditioned on semantic map

        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        # segmap_debug = segmap.detach().cpu().numpy()
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out





class SPADE3DResnetBlock(nn.Module):
    def __init__(self, fin, fout, opt):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv3d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv3d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv3d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        if 'spectral' in opt.norm_G:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        spade_config_str = opt.norm_G.replace('spectral', '')
        self.norm_0 = SPADE3D(spade_config_str, fin, opt.spade_nc, opt.nhidden)
        self.norm_1 = SPADE3D(spade_config_str, fmiddle, opt.spade_nc,opt.nhidden)
        if self.learned_shortcut:
            self.norm_s = SPADE3D(spade_config_str, fin, opt.spade_nc, opt.nhidden)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))

        out = (x_s + dx) / math.sqrt(2)  # unit variance

        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


    

class SPADE2DResnetBlock(nn.Module):
    def __init__(self, fin, fout, opt):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        if 'spectral' in opt.norm_G:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        spade_config_str = opt.norm_G.replace('spectral', '')
        self.norm_0 = SPADE2D(spade_config_str, fin, opt.spade_nc, opt.nhidden)
        self.norm_1 = SPADE2D(spade_config_str, fmiddle, opt.spade_nc,opt.nhidden)
        if self.learned_shortcut:
            self.norm_s = SPADE2D(spade_config_str, fin, opt.spade_nc, opt.nhidden)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))

        out = (x_s + dx) / math.sqrt(2)  # unit variance

        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)

class Norm3DResnetBlock(nn.Module):
    def __init__(self, fin, fout):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv3d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv3d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv3d(fin, fout, kernel_size=1, bias=False)

        # define normalization layers
        self.norm_0 = nn.InstanceNorm3d(fin, affine=True)
        self.norm_1 = nn.InstanceNorm3d(fmiddle, affine=True)
        if self.learned_shortcut:
            self.norm_s =  nn.InstanceNorm3d(fin, affine=True)

    def forward(self, x):
        x_s = self.shortcut(x)

        dx = self.conv_0(self.actvn(self.norm_0(x)))
        dx = self.conv_1(self.actvn(self.norm_1(dx)))

        out = (x_s + dx) / math.sqrt(2)  # unit variance
        return out

    def shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)

    

class Norm2DResnetBlock(nn.Module):
    def __init__(self, fin, fout):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # define normalization layers
        self.norm_0 = nn.InstanceNorm3d(fin, affine=True)
        self.norm_1 = nn.InstanceNorm3d(fmiddle, affine=True)
        if self.learned_shortcut:
            self.norm_s =  nn.InstanceNorm3d(fin, affine=True)

    def forward(self, x):
        x_s = self.shortcut(x)

        dx = self.conv_0(self.actvn(self.norm_0(x)))
        dx = self.conv_1(self.actvn(self.norm_1(dx)))

        out = (x_s + dx) / math.sqrt(2)  # unit variance
        return out

    def shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)





class SPADEGenerator3D(BaseNetwork):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.out_channel

        max_nc = self.opt.max_nc
        self.max_nc = max_nc
        opt.semantic_nc = opt.semantic_channel

        self.sw, self.sh, self.sl = self.compute_latent_vector_size3D(opt)
        if opt.use_oasis:
            opt.z_dim = opt.z_dim_oasis
            opt.spade_nc = opt.semantic_nc + self.opt.z_dim
            self.fc = nn.Conv3d(opt.semantic_nc + self.opt.z_dim, min(16 * nf, max_nc), 3, padding=1)
        else:
            opt.spade_nc = opt.semantic_nc
            self.fc = nn.Linear(opt.z_dim, min(16 * nf, max_nc) * self.sw * self.sh * self.sl)

       
        self.head_0 = SPADE3DResnetBlock(min(16 * nf, max_nc), min(16 * nf, max_nc), opt)
        self.G_middle_0 = SPADE3DResnetBlock(min(16 * nf, max_nc), min(8 * nf, max_nc), opt)

        self.up = nn.Upsample(scale_factor=2)
        self.up_0 = SPADE3DResnetBlock(min(8 * nf, max_nc), min(4 * nf, max_nc), opt)
        self.up_1 = SPADE3DResnetBlock(min(4 * nf, max_nc), min(2 * nf, max_nc), opt)
        self.up_2 = SPADE3DResnetBlock(min(2 * nf, max_nc), min(2 * nf, max_nc), opt)
        if opt.use_uncondition_layer:
            self.nospade = Norm3DResnetBlock(min(2 * nf, max_nc), min(1 * nf, max_nc))
        else:
            self.spade = SPADE3DResnetBlock(min(2 * nf, max_nc), min(1 * nf, max_nc), opt)

        
        self.out =  nn.Sequential\
        (   nn.InstanceNorm3d(nf, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv3d(nf, nf, 1, 1, 0))



    def compute_latent_vector_size3D(self, opt):
        if opt.num_upsampling_layers == 'normal':
            num_up_layers = 4
        elif opt.num_upsampling_layers == 'more':
            num_up_layers = 5
        elif opt.num_upsampling_layers == 'most':
            num_up_layers = 6
        else:
            raise ValueError('opt.num_upsampling_layers [%s] not recognized' %
                             opt.num_upsampling_layers)

        sw = opt.w // (2**num_up_layers)
        sh = round(sw / (opt.w / opt.h))
        sl = round(sw / (opt.w / opt.l))

        return sw, sh, sl

    def forward(self, input, z=None):
        seg = input

        if z is None:
             z = torch.randn(input.size(0), self.opt.z_dim,
                                dtype=torch.float32, device=input.get_device())
        elif z.shape[-1] > self.opt.z_dim:
            z = z[...,:self.opt.z_dim]
        
        if not self.opt.use_oasis:
            # we sample z from unit normal and reshape the tensor
            x = self.fc(z)
            x = x.view(-1, min(16 * self.opt.ngf, self.max_nc), self.sh, self.sw, self.sl)
        else:
            z = z.view(z.size(0), self.opt.z_dim, 1, 1,1)
            z = z.expand(z.size(0), self.opt.z_dim, seg.size(2), seg.size(3),seg.size(4))
            seg = torch.cat((z, seg), dim = 1)
            # we downsample segmap and run convolution
            x = F.interpolate(seg, size=(self.sh, self.sw,self.sl))
            #a = x.detach().cpu().numpy()
            x = self.fc(x)

        x = self.head_0(x, seg)
        x = self.up(x)
        x = self.G_middle_0(x, seg)
        #x = self.G_middle_1(x, seg)
        x = self.up(x)
        x = self.up_0(x, seg)
        x = self.up(x)
        x = self.up_1(x, seg)
        x = self.up(x)
        x = self.up_2(x, seg)

        if self.opt.use_uncondition_layer:
            x = self.nospade(x)

        if self.opt.use_out_normalize:
            x = self.out(x)
            #x = F.interpolate(x, size=(raw_H, raw_W))
        else:
            x = F.leaky_relu(x, 2e-1)
        return x




class SPADEGenerator2D(BaseNetwork):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ngf

        max_nc = self.opt.max_nc
        self.max_nc = max_nc

        self.sw, self.sh = self.compute_latent_vector_size2D(opt)

        if opt.use_oasis:
            opt.z_dim = opt.oasis_zdim
            opt.spade_nc = opt.semantic_nc + self.opt.z_dim
            self.fc = nn.Conv2d(opt.semantic_nc + self.opt.z_dim, min(16 * nf, max_nc), 3, padding=1)
        else:
            opt.spade_nc = opt.semantic_nc
            self.fc = nn.Linear(opt.z_dim, min(16 * nf, max_nc) * self.sw * self.sh)

       
        self.head_0 = SPADE2DResnetBlock(min(16 * nf, max_nc), min(16 * nf, max_nc), opt)

        self.G_middle_0 = SPADE2DResnetBlock(min(16 * nf, max_nc), min(8 * nf, max_nc), opt)
        # self.G_middle_1 = SPADEResnetBlock(min(16 * nf, max_nc), min(16 * nf, max_nc), opt)

        self.up_0 = SPADE2DResnetBlock(min(8 * nf, max_nc), min(4 * nf, max_nc), opt)
        self.up_1 = SPADE2DResnetBlock(min(4 * nf, max_nc), min(2 * nf, max_nc), opt)
        self.up_2 = SPADE2DResnetBlock(min(2 * nf, max_nc), min(2 * nf, max_nc), opt)

        
        self.nospade = Norm2DResnetBlock(min(2 * nf, max_nc), min(1 * nf, max_nc))
        self.spade = SPADE2DResnetBlock(min(2 * nf, max_nc), min(1 * nf, max_nc), opt)

        # if opt.num_upsampling_layers == 'most':
        #     self.up_4 = SPADEResnetBlock(1 * nf, nf // 2, opt)
        #     final_nc = nf // 2

        self.up = nn.Upsample(scale_factor=2)

        self.out =  nn.Sequential\
        (   nn.InstanceNorm2d(nf, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(nf, nf, 1, 1, 0))



    def compute_latent_vector_size2D(self, opt):

        if opt.num_upsampling_layers == 'small':
            num_up_layers = 3
        elif opt.num_upsampling_layers == 'normal':
            num_up_layers = 4
        elif opt.num_upsampling_layers == 'more':
            num_up_layers = 5
        elif opt.num_upsampling_layers == 'most':
            num_up_layers = 6
        else:
            raise ValueError('opt.num_upsampling_layers [%s] not recognized' %
                             opt.num_upsampling_layers)

        sw = opt.w // (2**num_up_layers)
        sh = round(sw / (opt.w / opt.h))

        return sw, sh

    def forward(self, input, z=None, use_free_layer = False):
        seg = input

        if z is None:
             z = torch.randn(input.size(0), self.opt.z_dim,
                                dtype=torch.float32, device=input.get_device())
        elif z.shape[-1] > self.opt.z_dim:
            z = z[...,:self.opt.z_dim]
        
        if not self.opt.use_oasis:
            # we sample z from unit normal and reshape the tensor
            x = self.fc(z)
            x = x.view(-1, min(16 * self.opt.ngf, self.max_nc), self.sh, self.sw)
        else:
            z = z.view(z.size(0), self.opt.z_dim, 1, 1)
            z = z.expand(z.size(0), self.opt.z_dim, seg.size(2), seg.size(3))
            seg = torch.cat((z, seg), dim = 1)
            # we downsample segmap and run convolution
            x = F.interpolate(seg, size=(self.sh, self.sw))
            #a = x.detach().cpu().numpy()
            x = self.fc(x)

        x = self.head_0(x, seg)
        x = self.up(x)
        x = self.G_middle_0(x, seg)
        #x = self.G_middle_1(x, seg)
        if self.opt.num_upsampling_layers == 'normal':
            x = self.up(x)
        x = self.up_0(x, seg)
        x = self.up(x)
        x = self.up_1(x, seg)
        x = self.up(x)
        x = self.up_2(x, seg)

        if use_free_layer:
            x = self.nospade(x)
        else:
            x = self.spade(x, seg)

        if self.opt.use_out_normalize:
            x = self.out(x)
            
        else:
             x = F.leaky_relu(x, 2e-1)
        x = F.interpolate(x, size=(self.opt.h, self.opt.w))
        return x

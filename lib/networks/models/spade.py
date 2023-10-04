import torch
import torch.nn as nn
import torch.nn.functional as F
import math


from .layers import SPADE3DResnetBlock, ConvResBlock3d, EqualLinear, EqualConv3d, ConvResBlock2d

class FeatureVolumeGenerator(nn.Module):
    def __init__(self, 
        init_res = 4,
        volume_res = (64, 64, 64),
        max_channel= 512,
        out_channel = 16,
        semantic_channel = 42,
        spade_hidden_channel = 128,
        data_normalize_type = 'instance',
        weight_normalize_type = 'equal_lr',
        noise_type = 'oasis',
        z_dim = 256,
        z_dim_oasis = 64,
        sparse_conv = False,
        kernel_size = 3,
        final_unconditional_layer = True,
        final_tanh = False,
        **kwargs):
        super().__init__()
        self.h, self.w, self.l = volume_res
        out_res = min(volume_res)
        self.out_nc = out_channel
        self.max_nc = max_channel
        self.noise_type = noise_type
        self.z_dim = z_dim
        self.final_tanh = final_tanh
        self.final_unconditional_layer = final_unconditional_layer

        res_log2 =  int(math.log2(out_res) - math.log2(init_res))
        self.block_num = res_log2
        nc_list = [min(self.out_nc * (2 ** (res_log2 - i)),self.max_nc) for i in range(res_log2)]
        nc_list += [nc_list[-1], out_channel]
        self.nc_list = nc_list
        self.sw, self.sh, self.sl = self.compute_latent_vector_size3D(num_up_layers=self.block_num)
        if noise_type == 'oasis':
            self.z_dim_oasis = z_dim_oasis 
            z_nc = z_dim_oasis
            spade_nc = semantic_channel + z_nc
            self.fc = EqualConv3d(spade_nc ,nc_list[0], 3, padding=1)
        else:
            spade_nc = semantic_channel
            self.fc = EqualLinear(z_dim, nc_list[0] * self.sw * self.sh * self.sl)

        for i in range(self.block_num):
            block_name = f'spade_block{i}'
            self.add_module(block_name, 
                            SPADE3DResnetBlock(in_channel=nc_list[i],out_channel=nc_list[i+1],spade_nc=spade_nc, hidden_nc=spade_hidden_channel))

        self.up2x= nn.Upsample(scale_factor=2)

        if final_unconditional_layer:
            self.nospade = ConvResBlock3d(nc_list[-2], nc_list[-2])
        # else:
        #     self.spade = SPADE3DResnetBlock(min(2 * nf, max_nc), min(1 * nf, max_nc), opt)
        self.out =  nn.Sequential\
        (  nn.LeakyReLU(0.2),
        EqualConv3d(nc_list[-2], nc_list[-1], kernel_size=3 ,padding=1))

    def compute_latent_vector_size3D(self, num_up_layers):
         
        sw = self.w // (2**num_up_layers)
        sh = round(sw / (self.w / self.h))
        sl = round(sw / (self.w / self.l))

        return sw, sh, sl

    def forward(self, input, z=None):
        seg = input
        bs= input.shape[0]
        if z is None:
             z = torch.randn(bs, self.z_dim,
                                dtype=torch.float32, device=seg.get_device())
        # elif z.shape[-1] > self.opt.z_dim:
        #     z = z[...,:self.opt.z_dim]
        
        if self.noise_type == 'oasis':
            # z = z[:, :self.z_dim_oasis] # only use z_oasis part
            # z = z.view(bs, self.z_dim_oasis, 1, 1, 1)
            # z = z.expand(bs, self.z_dim_oasis, seg.size(2), seg.size(3),seg.size(4))
            # seg = torch.cat((z, seg), dim = 1)
            # # we downsample segmap and run convolution
            # x = F.interpolate(seg, size=(self.sh, self.sw, self.sl))
            # x = self.fc(x)
            
            # when doing inversion, we optimize each latent 
            
        else:
            # we sample z from unit normal and reshape the tensor
            x = self.fc(z)
            x = x.view(-1, self.nc_list[0], self.sh, self.sw, self.sl)

        for i in range(self.block_num):
            block = getattr(self, f'spade_block{i}')
            x = block(x, seg)
            x = self.up2x(x)

        if self.final_unconditional_layer:
            x = self.nospade(x)
        x = self.out(x)
        if self.final_tanh:
            x = torch.tanh(x)
        return x


class FeatureVolumeEncoder(nn.Module):
    def __init__(self,
                volume_size = (64,64,64),
                input_channel = 64,
                output_channel = 128,
                channel_max = 256):
        super.__init__()
        h, w, l = volume_size
        
        init_res = min(w, l)
        wl_ratio = round(max((w, l)) / init_res)

        block_num = torch.log2(init_res)
        for i in range(block_num):
            self.add_module('block_%d'%i,ConvResBlock2d(in_channel=1,
                                                        out_channel=1,
                                                        downsample=True))
            
        self.fc_out = EqualLinear(in_channel= 1,
                                  out_channel=out_channel,
                                  activate=True)
        

    def forward(self, V):

        B, C, H, W, L= V.shape

        x = V.reshape(B, C * H, W, L)

        for i in range(self.block_num):
            
            x = (x)

        return x



class FeatureMapGenerator(nn.Module):
    def __init__(self):
        

        # self.encoder = V
        pass


    def forward(self, V):
        B, C = V.shape

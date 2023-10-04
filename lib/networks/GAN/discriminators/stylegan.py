import torch.nn as nn
import numpy as np 
import torch
from torchvision import transforms
import torchvision
import torch.nn.functional as F
import math
from lib.utils.img_utils import save_tensor_img
from lib.config import cfg
from lib.networks.reference.stylegan2 import MappingNetwork


sobel_x = torch.tensor([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]], dtype=torch.float, requires_grad=False).view(1, 1, 3, 3)
sobel_y = torch.tensor([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=torch.float, requires_grad=False).view(1, 1, 3, 3)
laplace = torch.tensor([[0, 1, 0],
                        [1, -4, 1],
                        [0, 1, 0]], dtype=torch.float, requires_grad=False).view(1, 1, 3, 3)
avgpool = torch.tensor([[1/9, 1/9, 1/9],
                         [1/9, 1/9, 1/9],
                         [1/9, 1/9, 1/9]], dtype=torch.float, requires_grad=False).view(1, 1, 3, 3)

egde_kernel_dict = {
    'sobel_x':sobel_x,
    'sobel_y':sobel_y,
    'laplace':laplace,
    'avgpool':avgpool
}

class Discriminator(nn.Module):
    def __init__(self, image_size=256, num_domains=1, n_feat=512,in_channels = 3, is_kitti_img =True):
        super().__init__()
        self.num_domains = num_domains
        self.in_channels = in_channels
        max_conv_dim = n_feat
        dim_in = 2**14 // image_size
        blocks = []
        blocks += [nn.Conv2d(in_channels, dim_in, 3, 1, 1)]


        repeat_num = int(np.log2(image_size)) - 2

        downsample_list = [True,True,True,True,True,True,True,True]

        for i in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=downsample_list[i])]
            dim_in = dim_out



        # blocks += [nn.LeakyReLU(0.2)]
        # blocks += [nn.Conv2d(dim_out, dim_out, 2, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        if is_kitti_img:
            blocks += [nn.Conv2d(dim_out, dim_out, (1,5), 1, 0)]
            blocks += [nn.LeakyReLU(0.2)]
        else:
            blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
            blocks += [nn.LeakyReLU(0.2)]
            
        # blocks += [nn.Conv2d(dim_out, 1, 1, 1, 0)]
        self.main = nn.Sequential(*blocks)
        self.fc = nn.Linear(dim_out, num_domains)

        # if True:
        #     self.depth_pred = DepthPredModule()
    def forward(self, x):
        out = {}
        x = x[:,None]
        if cfg.is_debug:
            save_tensor_img(torchvision.utils.make_grid(x[:,0,:3].detach().cpu()), 'tmp', 'rgb_Din.jpg')
            save_tensor_img(torchvision.utils.make_grid(x[:,0,-1:]), 'tmp', 'depth_Din.jpg', 'depth')
  
        depth_pred_fearures = []
        # x = x[:,:,:self.in_channels]
        B, D, C, H, W = x.shape
        assert self.num_domains == D
        x = x.reshape(B * D, C, H, W)[:,:self.in_channels]
        # save_tensor_img(torchvision.utils.make_grid(x[:,:3]), 'tmp', 'aa.jpg')
        for l in self.main:
            x = l(x)
            if l in [2,3,4]:
                depth_pred_fearures.append()
        score = x.reshape(x.size(0), -1)
        score = self.fc(score)
        score = score.reshape(B, D, -1) 
        # out = out.view(B, D, 2)  # (batch, num_domains)
        idx = torch.LongTensor(range(score.size(1))).to(score.device)
        score = score[:, idx, idx]  # (batch)
        out['score'] = score
        # if True:
        #     self.dep
        #     out['depth_pred'] = depth_pred
        return out

class Discriminator_obj(nn.Module):
    def __init__(self, image_size=64, num_domains=1, n_feat=256,in_channels = 3, use_pose_condition = True, mapping_layer_num = 2, c_dim = 15, c_map_dim = 256):
        super().__init__()
        self.num_domains = num_domains
        self.in_channels = in_channels
        self.use_pose_condition = use_pose_condition
        self.mapping_layer_num = mapping_layer_num 
        self.c_dim = c_dim
        self.c_map_dim = c_map_dim
        max_conv_dim = n_feat
        dim_in = 2**14 // image_size
        blocks = []
        blocks += [nn.Conv2d(in_channels, dim_in, 3, 1, 1)]


        repeat_num = int(np.log2(image_size)) - 2

        # downsample_list = [True,True,True,True,True,True]

        for i in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        self.main = nn.Sequential(*blocks)

        # self.c_normalize = nn.BatchNorm1d(self.c_dim, affine=True, track_running_stats=False)
        self.mapping = MappingNetwork(z_dim=0,c_dim=self.c_dim,w_dim=self.c_map_dim,num_layers= mapping_layer_num)
        self.pose_condition = DiscriminatorEpilogue(in_channels=n_feat,cmap_dim=self.c_map_dim,resolution=4)
        # blocks += [nn.Conv2d(dim_out, 1, 1, 1, 0)]

        no_pose = []
        no_pose += [nn.LeakyReLU(0.2)]
        no_pose += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        no_pose += [nn.LeakyReLU(0.2)]
        no_pose += [nn.Conv2d(dim_out, num_domains, 1, 1, 0)]
        self.no_pose = nn.Sequential(*no_pose)
    def forward(self, x, c = None, **kwargs):

        if cfg.is_debug:         
             save_tensor_img(torchvision.utils.make_grid(x.detach().cpu()[:,0:3]), 'tmp', 'rgb_obj_Din.jpg')
            # if self.in_channels == 6:
            #     save_tensor_img(torchvision.utils.make_grid(x.detach().cpu()[:,3:6]), 'tmp', 'pose_obj_Din.jpg')

        x = x[:,:self.in_channels]
        # save_tensor_img(torchvision.utils.make_grid(x.detach().cpu()), 'tmp', 'z.jpg')
        B, C, H, W = x.shape

        # save_tensor_img(torchvision.utils.make_grid(x[:,:3]), 'tmp', 'aa.jpg')
        for l in self.main:
            x = l(x)
        # out = x.reshape(x.size(0), -1)
        # out = out.view(B, D, 2)  # (batch, num_domains)
        out = self.no_pose(x).reshape(x.size(0), -1)
        if self.use_pose_condition:
            cmap = self.mapping(None, c[:,:self.c_dim])
            out += self.pose_condition(x, cmap)
            
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

class DiscriminatorEpilogue(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        cmap_dim,                       # Dimensionality of mapped conditioning label, 0 = no label.
        resolution,                     # Resolution of this block.
        # img_channels,                   # Number of input color channels.
        # architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        mbstd_group_size    = 4,        # Group size for the minibatch standard deviation layer, None = entire minibatch.
        mbstd_num_channels  = 1,        # Number of features for the minibatch standard deviation layer, 0 = disable.
    ):
        # assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.cmap_dim = cmap_dim
        self.resolution = resolution
        # self.img_channels = img_channels
        # self.architecture = architecture
        # if architecture == 'skip':
        #     self.fromrgb = Conv2dLayer(img_channels, in_channels, kernel_size=1, activation=activation)
        self.mbstd = MinibatchStdLayer(group_size=mbstd_group_size, num_channels=mbstd_num_channels) if mbstd_num_channels > 0 else None
        self.conv = nn.Conv2d(in_channels + mbstd_num_channels, in_channels, 3, 1, 1)
        self.fc = nn.Linear(in_channels * (resolution ** 2), in_channels)
        self.out = nn.Linear(in_channels, 1 if cmap_dim == 0 else cmap_dim)
        self.act = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, cmap):

        # Main layers.
        if self.mbstd is not None:
            x = self.mbstd(x)
        x = self.act(self.conv(x))
        x = self.act(self.fc(x.flatten(1)))
        x = self.out(x)
        
        # Conditioning.
        if self.cmap_dim > 0:
            # misc.assert_shape(cmap, [None, self.cmap_dim])
            x = (x * cmap).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))

        return x

class MinibatchStdLayer(torch.nn.Module):
    def __init__(self, group_size, num_channels=1):
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, x):
        N, C, H, W = x.shape
        G = N
        F = self.num_channels
        c = C // F

        y = x.reshape(G, -1, F, c, H, W)    # [GnFcHW] Split minibatch N into n groups of size G, and channels C into F groups of size c.
        y = y - y.mean(dim=0)               # [GnFcHW] Subtract mean over group.
        y = y.square().mean(dim=0)          # [nFcHW]  Calc variance over group.
        y = (y + 1e-8).sqrt()               # [nFcHW]  Calc stddev over group.
        y = y.mean(dim=[2,3,4])             # [nF]     Take average over channels and pixels.
        y = y.reshape(-1, F, 1, 1)          # [nF11]   Add missing dimensions.
        y = y.repeat(G, 1, H, W)            # [NFHW]   Replicate over group and pixels.
        x = torch.cat([x, y], dim=1)        # [NCHW]   Append to input as new channels.
        return x


class DepthPredModule(torch.nn.Module):
    def __init__(self, in_channels_64, in_channels_32, in_channels_16, out_channels=10, scale_factor=4):
        super().__init__()
        self.in_channels_64 = in_channels_64
        self.in_channels_32 = in_channels_32
        self.in_channels_16 = in_channels_16
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.conv1 = nn.Conv2d(in_channels=self.in_channels_16,
                               out_channels=self.in_channels_16//2,
                               kernel_size=3)
        self.block1 = ResBlk(self.in_channels_16//2)
        self.conv2 = nn.Conv2d(in_channels=self.in_channels_16//2+self.in_channels_32,
                               out_channels=self.in_channels_32//2,
                               kernel_size=3)
        self.block2 = ResBlk(self.in_channels_32//2)
        self.conv3 = nn.Conv2d(in_channels=self.in_channels_32//2+self.in_channels_64,
                               out_channels=self.in_channels_64//2,
                               kernel_size=3)
        self.block3 = ResBlk(self.in_channels_64//2)
        self.convf = nn.Conv2d(in_channels=self.in_channels_64//2,
                               out_channels=out_channels,
                               kernel_size=1)
        
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x64, x32, x16):
        x = self.lrelu(self.conv1(x16.to(torch.float32)))
        x = self.block1(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat([x,  x32.to(torch.float32)], dim=1)

        x = self.lrelu(self.conv2(x))
        x = self.block2(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat([x,  x64.to(torch.float32)], dim=1)

        x = self.lrelu(self.conv3(x))
        x = self.block3(x)
        x = self.convf(x)
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        return x



# class DepthLossLayer(torch.nn.Module):
#     def __init__(self, in_channels_64, in_channels_32, in_channels_16, out_channels=10, scale_factor=4):
#         super().__init__()
#         self.in_channels_64 = in_channels_64
#         self.in_channels_32 = in_channels_32
#         self.in_channels_16 = in_channels_16
#         self.out_channels = out_channels
#         self.scale_factor = scale_factor
#         self.conv1 = ConvLayer(in_channels=self.in_channels_16,
#                                out_channels=self.in_channels_16//2,
#                                kernel_size=3,
#                                add_bias=True,
#                                scale_factor=1,
#                                filter_kernel=None,
#                                use_wscale=True,
#                                wscale_gain=1.0,
#                                lr_mul=1.0,
#                                activation_type='lrelu',
#                                conv_clamp=None)
#         self.block1 = ResBlock(self.in_channels_16//2)
#         self.conv2 = ConvLayer(in_channels=self.in_channels_16//2+self.in_channels_32,
#                                out_channels=self.in_channels_32//2,
#                                kernel_size=3,
#                                add_bias=True,
#                                scale_factor=1,
#                                filter_kernel=None,
#                                use_wscale=True,
#                                wscale_gain=1.0,
#                                lr_mul=1.0,
#                                activation_type='lrelu',
#                                conv_clamp=None)
#         self.block2 = ResBlock(self.in_channels_32//2)
#         self.conv3 = ConvLayer(in_channels=self.in_channels_32//2+self.in_channels_64,
#                                out_channels=self.in_channels_64//2,
#                                kernel_size=3,
#                                add_bias=True,
#                                scale_factor=1,
#                                filter_kernel=None,
#                                use_wscale=True,
#                                wscale_gain=1.0,
#                                lr_mul=1.0,
#                                activation_type='lrelu',
#                                conv_clamp=None)
#         self.block3 = ResBlock(self.in_channels_64//2)
#         self.convf = ConvLayer(in_channels=self.in_channels_64//2,
#                                out_channels=out_channels,
#                                kernel_size=1,
#                                add_bias=True,
#                                scale_factor=1,
#                                filter_kernel=None,
#                                use_wscale=True,
#                                wscale_gain=1.0,
#                                lr_mul=1.0,
#                                activation_type='lrelu',
#                                conv_clamp=None)

#     def forward(self, x64, x32, x16):
#         x = self.conv1(x16.to(torch.float32))
#         x = self.block1(x)
#         x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
#         x = torch.cat([x,  x32.to(torch.float32)], dim=1)

#         x = self.conv2(x)
#         x = self.block2(x)
#         x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
#         x = torch.cat([x,  x64.to(torch.float32)], dim=1)

#         x = self.conv3(x)
#         x = self.block3(x)
#         x = self.convf(x)
#         x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
#         return x


# class ConvLayer(nn.Module):
#     """Implements the convolutional layer.

#     If downsampling is needed (i.e., `scale_factor = 2`), the feature map will
#     be filtered with `filter_kernel` first.
#     """

#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  kernel_size,
#                  add_bias,
#                  scale_factor,
#                  filter_kernel,
#                  use_wscale,
#                  wscale_gain,
#                  lr_mul,
#                  activation_type,
#                  conv_clamp):
#         """Initializes with layer settings.

#         Args:
#             in_channels: Number of channels of the input tensor.
#             out_channels: Number of channels of the output tensor.
#             kernel_size: Size of the convolutional kernels.
#             add_bias: Whether to add bias onto the convolutional result.
#             scale_factor: Scale factor for downsampling. `1` means skip
#                 downsampling.
#             filter_kernel: Kernel used for filtering.
#             use_wscale: Whether to use weight scaling.
#             wscale_gain: Gain factor for weight scaling.
#             lr_mul: Learning multiplier for both weight and bias.
#             activation_type: Type of activation.
#             conv_clamp: A threshold to clamp the output of convolution layers to
#                 avoid overflow under FP16 training.
#         """
#         super().__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.add_bias = add_bias
#         self.scale_factor = scale_factor
#         self.filter_kernel = filter_kernel
#         self.use_wscale = use_wscale
#         self.wscale_gain = wscale_gain
#         self.lr_mul = lr_mul
#         self.activation_type = activation_type
#         self.conv_clamp = conv_clamp

#         weight_shape = (out_channels, in_channels, kernel_size, kernel_size)
#         fan_in = kernel_size * kernel_size * in_channels
#         wscale = wscale_gain / np.sqrt(fan_in)
#         if use_wscale:
#             self.weight = nn.Parameter(torch.randn(*weight_shape) / lr_mul)
#             self.wscale = wscale * lr_mul
#         else:
#             self.weight = nn.Parameter(
#                 torch.randn(*weight_shape) * wscale / lr_mul)
#             self.wscale = lr_mul

#         if add_bias:
#             self.bias = nn.Parameter(torch.zeros(out_channels))
#             self.bscale = lr_mul
#         else:
#             self.bias = None
#         self.act_gain = bias_act.activation_funcs[activation_type].def_gain

#         if scale_factor > 1:
#             assert filter_kernel is not None
#             self.register_buffer(
#                 'filter', upfirdn2d.setup_filter(filter_kernel))
#             fh, fw = self.filter.shape
#             self.filter_padding = (
#                 kernel_size // 2 + (fw - scale_factor + 1) // 2,
#                 kernel_size // 2 + (fw - scale_factor) // 2,
#                 kernel_size // 2 + (fh - scale_factor + 1) // 2,
#                 kernel_size // 2 + (fh - scale_factor) // 2)

#     def extra_repr(self):
#         return (f'in_ch={self.in_channels}, '
#                 f'out_ch={self.out_channels}, '
#                 f'ksize={self.kernel_size}, '
#                 f'wscale_gain={self.wscale_gain:.3f}, '
#                 f'bias={self.add_bias}, '
#                 f'lr_mul={self.lr_mul:.3f}, '
#                 f'downsample={self.scale_factor}, '
#                 f'downsample_filter={self.filter_kernel}, '
#                 f'act={self.activation_type}, '
#                 f'clamp={self.conv_clamp}')

#     def forward(self, x, runtime_gain=1.0, impl='cuda'):
#         dtype = x.dtype

#         weight = self.weight
#         if self.wscale != 1.0:
#             weight = weight * self.wscale
#         bias = None
#         if self.bias is not None:
#             bias = self.bias.to(dtype)
#             if self.bscale != 1.0:
#                 bias = bias * self.bscale

#         if self.scale_factor == 1:  # Native convolution without downsampling.
#             padding = self.kernel_size // 2
#             x = conv2d_gradfix.conv2d(
#                 x, weight.to(dtype), stride=1, padding=padding, impl=impl)
#         else:  # Convolution with downsampling.
#             down = self.scale_factor
#             f = self.filter
#             padding = self.filter_padding
#             # When kernel size = 1, use filtering function for downsampling.
#             if self.kernel_size == 1:
#                 x = upfirdn2d.upfirdn2d(
#                     x, f, down=down, padding=padding, impl=impl)
#                 x = conv2d_gradfix.conv2d(
#                     x, weight.to(dtype), stride=1, padding=0, impl=impl)
#             # When kernel size != 1, use stride convolution for downsampling.
#             else:
#                 x = upfirdn2d.upfirdn2d(
#                     x, f, down=1, padding=padding, impl=impl)
#                 x = conv2d_gradfix.conv2d(
#                     x, weight.to(dtype), stride=down, padding=0, impl=impl)

#         act_gain = self.act_gain * runtime_gain
#         act_clamp = None
#         if self.conv_clamp is not None:
#             act_clamp = self.conv_clamp * runtime_gain
#         x = bias_act.bias_act(x, bias,
#                               act=self.activation_type,
#                               gain=act_gain,
#                               clamp=act_clamp,
#                               impl=impl)

#         assert x.dtype == dtype
#         return x

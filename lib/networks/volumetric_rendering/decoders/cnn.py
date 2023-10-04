import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class ResnetBlock(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # norm_layer = nn.BatchNorm2d(num_features=dim, affine=True ,track_running_stats = False)

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channel = dim, out_channel = dim, kernel_size=kernel_size,stride=1, padding=1),
            nn.BatchNorm2d(num_features=dim, affine=True ,track_running_stats = False),
            self.activation,
            nn.Conv2d(in_channel = dim, out_channel = dim, kernel_size=kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(num_features=dim, affine=True ,track_running_stats = False)
        )

    def forward(self, x):
        y = self.conv_block(x)
        out = self.activation(x + y)
        return out


class CNNRender(nn.Module):
    r"""CNN converting intermediate feature map to final image."""

    def __init__(self, in_channel = 64, hidden_channel=128, style_channel = 64,
                 leaky_relu=True, super_res = 1, **kwagrs):
        super(CNNRender, self).__init__()
        #self.fc_z_cond = nn.Linear(style_dim, 2 * 2 * hidden_channel)
        self.conv_in = nn.Conv2d(in_channel, hidden_channel, 1, stride=1, padding=0)
        AdainResBlk(hidden_channel,hidden_channel,style_channel)
        self.res0 = AdainResBlk(hidden_channel,hidden_channel,style_channel, upsample = (super_res >= 4))
        self.res1 = AdainResBlk(hidden_channel, hidden_channel,style_channel, upsample =(super_res >= 2))

        self.toRGB = nn.Conv2d(hidden_channel, 3, 1, stride=1, padding=0)

        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)



    def forward(self, x, rgb, w = None):
        r"""Forward network.

        Args:
            x (N x in_channel x H x W tensor): Intermediate feature map
            z (N x style_dim tensor): Style codes.
        """
        # z = self.fc_z_cond(z)
        # adapt = torch.chunk(z, 2 * 2, dim=-1)

        x = self.act(self.conv_in(x))
        x = self.res0(x, w)
        x = self.res1(x, w)
        rgb = self.toRGB(x)
        # if True:
        #     rgb = torch.tanh(rgb)

        return rgb



def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()


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

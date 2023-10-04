
import torch.nn as nn
from math import log2
from torchvision import transforms
import torchvision
import torch.nn.functional as F
from lib.utils.img_utils import save_tensor_img, set_grid


class DCDiscriminator(nn.Module):
    ''' DC Discriminator class.

    Args:
        in_dim (int): input dimension
        n_feat (int): features of final hidden layer
        img_size (int): input image size
    '''
    def __init__(self, 
    in_dim=3, 
    n_feat=512, 
    img_size=256,
    **kwargs):
        super(DCDiscriminator, self).__init__()

        self.in_dim = in_dim
        n_layers = int(log2(img_size) - 2)
        self.blocks = nn.ModuleList(
            [nn.Conv2d(in_dim,int(n_feat / (2 ** (n_layers - 1))),4, 2, 1, bias=False)] + 
            [nn.Conv2d(int(n_feat / (2 ** (n_layers - i))),int(n_feat / (2 ** (n_layers - 1 - i))),4, 2, 1, bias=False) for i in range(1, n_layers)])

        self.conv_out = nn.Conv2d(n_feat, 1, 4, 1, 0, bias=False)
        self.actvn = nn.LeakyReLU(0.2, inplace=True)

        self.resize = transforms.Resize((img_size, img_size))
        self.img_size = img_size

    def forward(self, x, **kwargs):
        # a = x[0].detach().numpy()
        if (x.shape[2] != self.img_size or x.shape[3] != self.img_size):
            x = self.resize(x)
            # save_tensor_img(x[0], save_dir='tmp/', name = 'aaaaaa.jpg')
        batch_size = x.shape[0]
        if x.shape[1] != self.in_dim:
            x = x[:, :self.in_dim]
        for layer in self.blocks:
            x= self.actvn(layer(x))

        out = self.conv_out(x)
        out = out.reshape(batch_size, 1)
        return out


class DiscriminatorResnet(nn.Module):
    ''' ResNet Discriminator class.

    Adopted from: https://github.com/LMescheder/GAN_stability

    Args:
        img_size (int): input image size
        nfilter (int): first hidden features
        nfilter_max (int): maximum hidden features
    '''
    def __init__(self,
        in_channels=3, 
        n_feat=512, 
        img_size=256,):
        super().__init__()
        s0 = self.s0 = 4
        self.sw = 2
        self.sh = 6
        nfilter = 16
        nf = self.nf = nfilter
        nfilter_max = n_feat
        nf_max = self.nf_max = nfilter_max

        size = img_size

        self.resize = transforms.Resize((size, size))
        self.img_size = size

        # Submodules
        self.conv_img = nn.Conv2d(in_channels, 1*nf, 3, padding=1)
        nlayers = int(log2(size / s0))
        self.nf0 = min(nf_max, nf * 2**nlayers)

        blocks = [
            ResnetBlock(nf, nf)
        ]
        for i in range(nlayers):
            nf0 = min(nf * 2**i, nf_max)
            nf1 = min(nf * 2**(i+1), nf_max)
            blocks += [
                nn.AvgPool2d(3, stride=2, padding=1),
                ResnetBlock(nf0, nf1),
            ]
        self.resnet = nn.Sequential(*blocks)
        self.fc = nn.Linear(self.nf0*2*6, 1)
        self.actvn = nn.LeakyReLU(0.2)

    def forward(self, x, **kwargs):
        batch_size = x.size(0)
        # if (x.shape[2] != self.img_size or x.shape[3] != self.img_size):
        #     #x = self.resize(x)
        #     x = F.interpolate(x, (self.img_size, self.img_size))
        save_tensor_img(torchvision.utils.make_grid(x[:,0:3]), 'tmp', 'real.jpg')

        out = self.conv_img(x)
        out = self.resnet(out)
        #a = self.nf0*self.s0*self.s0
        out = out.reshape(batch_size, self.nf0*self.sw*self.sh)
        out = self.fc(self.actvn(out))
        return out


class ResnetBlock(nn.Module):
    def __init__(self, fin, fout, fhidden=None, is_bias=True):
        super().__init__()
        # Attributes
        self.is_bias = is_bias
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden

        # Submodules
        self.conv_0 = nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(self.fhidden, self.fout,
                                3, stride=1, padding=1, bias=is_bias)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(
                self.fin, self.fout, 1, stride=1, padding=0, bias=False)

    def actvn(self, x):
        out = F.leaky_relu(x, 2e-1)
        return out

    def forward(self, x):
        x_s = self._shortcut(x)
        dx = self.conv_0(self.actvn(x))
        dx = self.conv_1(self.actvn(dx))
        out = x_s + 0.1*dx

        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s


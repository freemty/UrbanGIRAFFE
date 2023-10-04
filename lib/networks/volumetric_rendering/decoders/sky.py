import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.config import cfg
from numpy import pi

from lib.networks.volumetric_rendering.decoders.layers import PositionalEncoding

class skyDecoder(nn.Module):
    r"""MLP converting ray directions to sky features."""

    def __init__(self, style_dim = 256,in_channels = 3,frequency_bands = 6, 
    # out_channels_c=3,
    out_channel = 64,
    hidden_channel=256, leaky_relu=True, use_final_sigmoid = True, **kwargs):
        super(skyDecoder, self).__init__()

        out_channels_c = 3
        out_channels_feat = out_channel
        self.use_final_sigmoid = use_final_sigmoid

        self.positional_encode = PositionalEncoding(in_dim= in_channels, frequency_bands=frequency_bands ) 

        self.fc_z_a = nn.Linear(style_dim, hidden_channel, bias=False)

        self.fc1 = nn.Linear(in_channels + frequency_bands * 6 +hidden_channel , hidden_channel)
        self.fc2 = nn.Linear(hidden_channel, hidden_channel)
        self.fc3 = nn.Linear(hidden_channel, hidden_channel)
        self.fc4 = nn.Linear(hidden_channel, hidden_channel)
        self.fc5 = nn.Linear(hidden_channel, hidden_channel)

        self.fc_out_c = nn.Linear(out_channels_feat, out_channels_c)
        self.fc_out_feat = nn.Linear(hidden_channel, out_channels_feat)

        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x, z):
        r"""Forward network

        Args:
            x (... x in_channels tensor): Ray direction embeddings.
            z (... x style_dim tensor): Style codes.
        """

        x  = self.positional_encode(x)
        z = self.fc_z_a(z)
        while z.dim() < x.dim():
            z = z.unsqueeze(1).repeat(1,x.shape[1],1)
        # z = z.repeat(1,x.shape[1],1)
        y = self.fc1(torch.cat((x, z), dim = -1))
        y = self.act(y)
        y = self.act(self.fc2(y))
        y = self.act(self.fc3(y))
        y = self.act(self.fc4(y))
        y = self.act(self.fc5(y))



        feat = self.fc_out_feat(y)
        if True: 
            feat = torch.tanh(feat)
        rgb = self.fc_out_c(feat)
        # if self.use_final_sigmoid:
        #     rgb = torch.sigmoid(rgb)
        # feat = torch.cat((feat, rgb), dim = -1)
        return feat


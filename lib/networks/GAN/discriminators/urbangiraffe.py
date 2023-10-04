import os
from imageio import save
from pyrsistent import s
import torch.nn as nn
# from math import log2
# from torchvision import transforms
# import torchvision
# import torch.nn.functional as F
# from lib.utils.img_utils import save_tensor_img, set_grid


class Discriminator(nn.Module):
    ''' urbangiraffe Discriminator class.

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
        super(Discriminator, self).__init__()

        self.discriminator_main = 0

        if 'car' in ['car']:
            self.discriminator_car = 0
        if 'builiding' in ['building']:
            self.discriminator_building = 0

    def forward(self, x,data_type = 'object'):
        if data_type == 'main':
            d = self.discriminator_main(x)
        elif data_type == 'object':
            d = self.discriminator_car(x)


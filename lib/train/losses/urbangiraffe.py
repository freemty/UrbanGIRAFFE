from operator import imod
from matplotlib import image
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from lib.utils import net_utils
from lib.utils.img_utils import save_tensor_img
from lib.config import cfg
from torch.nn import functional as F
import pickle as pkl

import math
class NetworkWrapper(nn.Module):
    def __init__(self, net):
        super(NetworkWrapper, self).__init__()
        self.net = net
        self.color_crit = nn.MSELoss(reduction='sum')
        self.occupancy_crit = nn.MSELoss(reduction='sum')
        self.decay_speed = 0.00005
    
    def forward(self, batch, mode = 'train'):

        output = self.net(batch, mode = mode)
        batch_size =  batch['image'].shape[0]
        scalar_stats = {}
        loss = 0

        occupancy_gt = batch['occupancy_mask'] # bg -> 0, fg -> 1
        occupancy_pixel_num = torch.sum(occupancy_gt.reshape(batch_size,-1), dim = -1, keepdim=True)
        scope, alpha = output['scope_all'], output['alpha_all']

        if 'alpha_obj' in output.keys():
            obj_occupancy_gt = batch['obj_occupancy_mask']
            scope_obj, alpha_obj = output['scope_obj'], output['alpha_obj']
            bbx_fg_pixel_num = torch.sum(occupancy_gt.reshape(batch_size,-1), dim = -1, keepdim=True)
            bbx_pixel_num = torch.sum(scope_obj[0].reshape(batch_size,-1), dim = -1, keepdim=True)
            bbx_bg_pixel_num = bbx_pixel_num - bbx_fg_pixel_num
        if  'alpha_road' in output.keys():
            road_occupancy_gt = batch['road_occupancy_mask']
            scope_road, alpha_road = output['scope_road'], output['alpha_road']
            road_pixel_num = torch.sum(scope_road[0].reshape(batch_size,-1), dim = -1, keepdim=True)

        masked_rgb_pred = output['rgb'] * occupancy_gt.repeat(1, 3, 1, 1)
        masked_image = batch['image'] * occupancy_gt.repeat(1, 3, 1, 1)

        rgb_loss = cfg.train.weight_color *  self.color_crit(masked_image, masked_rgb_pred) / occupancy_pixel_num
        scalar_stats.update({'rgb_loss':rgb_loss})
        loss += rgb_loss
   
        scalar_stats.update({'loss': loss})
        image_stats = {}

        return output, loss, scalar_stats, image_stats




    def run_G(self):
        pass

    def run_D(self):
        pass

    def compute_D_loss(self):
        pass

    def compute_G_loss(self):
        pass

    def compute_recon_loss(self):
        pass

    def compute_point_loss(self):
        pass

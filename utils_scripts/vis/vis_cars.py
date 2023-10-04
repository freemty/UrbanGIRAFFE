
import json
import numpy as np
import os
import random
import imageio
from lib.config.yacs import CfgNode as CN

from tools.kitti360Scripts.helpers.labels import labels, name2label
import cv2
import pickle as pkl
from lib.utils.img_utils import save_tensor_img
from lib.utils.transform_utils import RT2tr, create_R
from lib.utils.camera_utils import  project_3dbbox
import torch
from lib.datasets.kitti360.urbangiraffe import Dataset


cfg = CN({
        'task': 'urbangiraffe',
        'min_visible_pixel': {'car':5000},
        'ratio': 0.5,
        'render_obj': True,
        'render_road': False,
        'render_stuff': True,
        'render_sky': True,
        'max_obj_num':4,
        'valid_object':['car'],
        'use_depth': False
})
dataset = Dataset(data_root = '/data/ybyang/KITTI-360', seq_list = [0], split = 'train')
dataloader = torch.utils.data.DataLoader(dataset)

patches = []
poses = []
i = 0
for batch in dataloader:
    s = batch['bbox_semantic_gt']
    a = batch['bbox_patches_gt'][s == 26]
    p = batch['bbox_pose_gt'][s == 26]
    poses += p
    patches += a
    poses += p
    if len(patches) >= 64:
        patches = torch.stack(patches)
        r = torch.stack(poses).detach().cpu().numpy()
        save_tensor_img(patches, 'tmp/good_data_', 'patches_gt%d.jpg'%i)
        i += 1
        patches, poses = [], []

#     if len(poses) >= 2000:
#         poses = torch.stack(poses).detach().cpu().numpy()
#         r, t, s = poses[:,:9], poses[:,9:12], poses[:,12:]

#         std_s, mean_s = np.std(s,axis=0), np.mean(s,axis=0)
#         std_t, mean_t = np.std(t,axis=0), np.mean(t,axis=0)
#         print(std_s, mean_s)
#         print(std_t, mean_t)

#         std_pose, mean_pose = np.std(poses,axis=0), np.mean(poses,axis=0)
#         print(std_pose, mean_pose)
#         print('a')

    
# [0.65541524 0.47280714 0.32196763] [4.5622563 2.1030486 1.6521143]
# [1.7725044  2.438227   0.14865872] [-0.40005124  1.1419871   0.4524286 ]
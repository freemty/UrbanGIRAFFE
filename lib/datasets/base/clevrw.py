
import numpy as np
import os
import random
from yaml import load
from lib.config import cfg, args
import imageio
import re

from tools.kitti360Scripts.helpers.labels import labels, name2label
import cv2
import pickle as pkl
from lib.utils.transform_utils import parse_R
from lib.utils.camera_utils import build_rays

class Dataset:
    def __init__(self, data_root, seq_list, split, use_full = False):
        super(Dataset, self).__init__()
        # path and initialization

        self.data_root = data_root

        self.seq_list = seq_list
        # self.render_road = cfg.render_road
        self.render_obj = cfg.render_obj
        self.render_stuff = cfg.render_stuff
        self.render_sky = cfg.render_sky
        # self.render_bg = cfg.render_bg

        self.use_depth = cfg.use_depth
        self.use_occupancy_mask = cfg.use_occupancy_mask

        # Split and val
        self.split = split
        self.spilt_rate = cfg.split_rate
        self.spilt_chunk_size = cfg.split_chunk_size
        self.use_full_dataset = use_full



        if split == 'train':
            self.stuff_semantic_name = cfg.stuff_semantic_list
            self.use_trajectory = cfg.train.use_trajectory
            # self.given_frame = cfg.train.train_given_frame 
        elif split == 'test':
            self.stuff_semantic_name = cfg.stuff_semantic_list
            self.use_trajectory = cfg.test.use_trajectory
            # self.given_frame = cfg.test.test_given_frame
        elif split == 'render':
            self.stuff_semantic_name = cfg.stuff_semantic_list_render
            self.use_trajectory = cfg.render.render_given_frame
            self.given_frame = cfg.render.render_given_frame
        else: 
            self.use_trajectory = False

        self.sr_multiple = cfg.super_resolution
        assert self.sr_multiple in [1,2,4]
        print('init base dataset done!')

    

    @staticmethod
    def load_layout(path, tr_mat = np.eye(4,4)):
        with open(path, 'rb+') as f:
            layout = pkl.load(f)

        for bbox in layout:
            bbox['bbox_tr'] = tr_mat @ bbox['bbox_tr']
        return layout

    @staticmethod
    def load_img(path, W = 352, H = 94, type = 'rgb', sr_multiple = 1):
        image = imageio.imread(path).astype(np.float32)
        # /data/ybyang/KITTI-360/data_2d_depth/2013_05_28_drive_0000_sync/image_00/depth_rect/0000000320.png
        if type == 'rgb':
            mean=127.5
            std=127.5
            rgb_image = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)
            rgb_image = (rgb_image - mean) / std
            return rgb_image[...,:3]
        elif type == 'depth':
            z_near, z_far = 0.001, cfg.z_far
            depth_image = cv2.resize(image, (W // sr_multiple, H // sr_multiple), interpolation=cv2.INTER_AREA).astype(np.float32)
            depth_image = cv2.resize(depth_image, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)
            # print(np.max(depth_image))
            depth_image = (depth_image - z_near) / (z_far - z_near)
            depth_image = np.clip(depth_image, 0., 1.)
            return np.expand_dims(depth_image, -1)
        else:
            raise KeyError


    @staticmethod
    def load_occupancy_mask(path, W = 352, H = 94, semantic = -1, to_bin = False,erode = False):
        occupancy_mask = imageio.imread(path)
        occupancy_mask = cv2.resize(occupancy_mask, (W, H),interpolation=cv2.INTER_NEAREST)
        if semantic != -1:
            occupancy_mask =  np.where(occupancy_mask == semantic, occupancy_mask,np.zeros_like(occupancy_mask) )

        if to_bin:
            occupancy_mask =  np.where(occupancy_mask != 0, np.ones_like(occupancy_mask),np.zeros_like(occupancy_mask) )

        if erode:
            kernel = np.ones((3,3),np.uint8)  
            occupancy_mask = cv2.erode(occupancy_mask,kernel,iterations = 1)

        return np.tile(occupancy_mask, (1,1,1)).transpose(1,2,0)

    @staticmethod
    def load_stuff_semantic(path, semantic_list):
        H, W, L = 64, 64, 64
        with open(path, 'rb+') as fp: 
            stuff_semantic_idx = pkl.load(fp)
        stuff_semantic = np.zeros(H * W * L)
        for s in semantic_list:
            if stuff_semantic_idx[s].shape[0] == 0:
                continue
            stuff_semantic[stuff_semantic_idx[s]] = name2label[s].id

        # stuff_semantic = np.ones_like(t['road']) * -1
        # for s in t.keys():
        #      stuff_semantic[t[s] != 0] = name2label[s].id
         
        return stuff_semantic.reshape(H, W, L)


    @staticmethod
    def load_bbox_pathces(patches_dir,patch_size, use_mask = True):

        patches_img, poses_img = [], []

        for patch_path in patches_dir:
            if patch_path == '/':
                patches_img.append(np.zeros((patch_size,patch_size,3)))
                poses_img.append(np.zeros((patch_size,patch_size,3)))
            else:
                patch = imageio.imread(patch_path)
                patch = (patch - 127.5) / 127.5
                pose_path = patch_path.replace('rgb', 'pose')
                pose = imageio.imread(pose_path) / 255.
                if use_mask:
                    occupancy_path = patch_path.replace('rgb', 'occupancy')
                    occupancy = imageio.imread(occupancy_path).clip(0,1)
                    # pose = imageio.imread(pose_path).clip(0,1)
                    kernel = np.ones((3,3),np.uint8)  
                    occupancy = cv2.erode(occupancy,kernel,iterations = 1)
                    occupancy = np.expand_dims(occupancy, axis=-1)
                    patch = patch * occupancy
                    # pose = pose * occupancy

                patches_img.append(patch)
                poses_img.append(pose)

        patches_img = np.stack(patches_img).astype(np.float32)
        poses_img = np.stack(poses_img).astype(np.float32)

        return patches_img, poses_img
    
    @staticmethod
    def parse_bbox_pose(bbox_trs, camera_pose):
        bbox_scale , bbox_rotate, _ = parse_R(bbox_trs[:,:3,:3],True)
        bbox_translate = bbox_trs[:,:3,3]
        _, camera_rotate, _ = parse_R(camera_pose[:3,:3], True)
        camera_translate = camera_pose[None,:3,3]
        

        camera_translate_bbox = (bbox_rotate.transpose(0,2,1) @ (camera_translate - bbox_translate)[:,:,None])[:,:,0] / bbox_scale
        camera_rotate_bbox = bbox_rotate.transpose(0,2,1) @ camera_rotate
        camera_rotate_bbox = camera_rotate_bbox.reshape((camera_rotate_bbox.shape[0],-1))
        
        #! camera_rotate_bbox = np.linalg.inv(bbox_rotate) @ camera_rotate
        bbox_poses = np.concatenate((camera_rotate_bbox , camera_translate_bbox, bbox_scale), axis= -1)

        poses_std = np.array([[0.4940787, 0.09209801, 0.8466173,  0.8509623,  0.05761873, 0.49184027,
                        0.03320127, 0.00357069, 0.02494918, 1.6393136,  2.3910007,  0.18180372,
                        0.60924697, 0.25510105, 0.32655525]] )
        
        poses_mean = np.array(
            [[ 6.5321065e-03, -1.7236471e-02,  1.7407486e-01, -1.7488994e-01,
            -2.9561704e-03,  6.3222572e-03, -3.9829183e-03, -9.9392325e-01,
            -1.0182006e-01, -4.0938866e-01,  9.4472373e-01,  4.4090703e-01,
            4.6299782e+00,  2.0885360e+00,  1.6754940e+00]])
        bbox_poses = (bbox_poses - poses_mean) / poses_std
        return  bbox_poses


    def __getitem__(self, index):

        camera_mat = self.intrinsic
        
        # Load scene for G training
        frame_id = self.frame_id_list[index]
        idx = self.frame2idx[frame_id]
        if self.use_trajectory:
            
            while True:
                frame_in_trajectory = random.choice(list(self.trajectory_dict[frame_id]))
                if frame_in_trajectory in self.frame2idx:
                    break
            frame_id_traj = frame_in_trajectory
            # frame_in_trajectory = list(self.trajectory_dict[frame_id])[15]
            rgb = self.rgb_image_dict[frame_in_trajectory]
            depth =  self.depth_image_dict[frame_in_trajectory]
            if self.use_occupancy_mask:
                occupancy_mask = self.occupancy_mask_dict[frame_in_trajectory]
            c2w = self.trajectory_dict[frame_id][frame_in_trajectory]
            # rays =  build_rays(H = self.H, W = self.W, K = camera_mat, c2w = c2w)
            rays =  build_rays(H = self.H_ray, W = self.W_ray, K = camera_mat, c2w = c2w)

            if self.render_obj:
                tr_mat = c2w @ np.linalg.inv(self.c2w_default)
                layout = self.load_layout(self.layout_dict[frame_in_trajectory], tr_mat)

        else:
            c2w = self.c2w_default
            rays = self.rays
            rgb = self.rgb_image_dict[frame_id]
            depth =  self.depth_image_dict[frame_id]
            if self.use_occupancy_mask:
                occupancy_mask = self.occupancy_mask_dict[frame_id]
            frame_id_traj = frame_id

            if self.render_obj:
                layout = self.load_layout(self.layout_dict[frame_id])

        # Load fake scene for D training
        frame_id_fake = self.frame_id_list_fake[index]
        idx_fake = self.frame2idx[frame_id_fake]

        if self.use_trajectory:
            # A = self.trajectory_dict[8078]
            while True:
                frame_in_trajectory = random.choice(list(self.trajectory_dict[frame_id_fake]))
                if frame_in_trajectory in self.frame2idx:
                    break
            frame_id_traj_fake = frame_in_trajectory
            # frame_in_trajectory = list(self.trajectory_dict[frame_id_fake])[-1]
            rgb_fake = self.rgb_image_dict[frame_in_trajectory]
            depth_fake = self.depth_image_dict[frame_in_trajectory]
            if self.use_occupancy_mask:
                occupancy_mask_fake = self.occupancy_mask_dict[frame_in_trajectory]
            c2w_fake = self.trajectory_dict[frame_id_fake][frame_in_trajectory]
            rays_fake =  build_rays(H = self.H_ray, W = self.W_ray, K = camera_mat, c2w = c2w_fake)

            if self.render_obj:
                tr_mat_fake = c2w_fake @ np.linalg.inv(self.c2w_default)
                layout_fake = self.load_layout(self.layout_dict[frame_in_trajectory], tr_mat_fake)

        else:
            c2w_fake = self.c2w_default
            rays_fake = self.rays
            rgb_fake = self.rgb_image_dict[frame_id_fake]
            depth_fake = self.depth_image_dict[frame_id_fake]
            if self.use_occupancy_mask:
                occupancy_mask_fake = self.occupancy_mask_dict[frame_id_fake]
            frame_id_traj_fake = frame_id_fake

            if self.render_obj:
                layout_fake = self.load_layout(self.layout_dict[frame_id_fake])

        # Load gt image for D training
        frame_id_gt = self.frame_id_list_gt[index]
        idx_gt = self.frame2idx[frame_id_gt]
        rgb_gt = self.rgb_image_dict[frame_id_gt]
        depth_gt= self.depth_image_dict[frame_id_gt]

        ret = {
            'camera_mat': camera_mat.astype(np.float32),
            'idx': idx,
            'frame_id': frame_id ,
            'rgb': self.load_img(rgb,W = self.W, H = self.H,type = 'rgb').transpose(2,0,1),
            'world_mat': c2w.astype(np.float32),
            'rays': rays.reshape((self.H_ray, self.W_ray, 9)).astype(np.float32),

            'idx_fake': idx_fake,
            'frame_id_fake': frame_id_fake , 

            'rgb_fake': self.load_img(rgb_fake,W = self.W, H = self.H, type = 'rgb').transpose(2,0,1),
            'world_mat_fake': c2w_fake.astype(np.float32),
            'rays_fake': rays_fake.reshape((self.H_ray, self.W_ray, 9)).astype(np.float32),

            'idx_gt': idx_gt,
            'frame_id_gt': frame_id_gt,   
            'rgb_gt': self.load_img(rgb_gt,W = self.W, H = self.H, type ='rgb').transpose(2,0,1),

        }
        if self.use_occupancy_mask:
            occupancy_mask_gt = self.occupancy_mask_dict[frame_id_gt]
            ret['occupancy_mask_fake'] =  self.load_occupancy_mask(occupancy_mask_fake,W = self.W, H = self.H).transpose(2,0,1)
            ret['occupancy_mask'] = self.load_occupancy_mask(occupancy_mask,W = self.W, H = self.H).transpose(2,0,1) 
            ret['occupancy_mask_gt'] = self.load_occupancy_mask(occupancy_mask_gt,W = self.W, H = self.H).transpose(2,0,1) 
        if self.use_depth:
            ret['depth'] = self.load_img(depth,W = self.W, H = self.H, type = 'depth',sr_multiple = self.sr_multiple).transpose(2,0,1)
            ret['depth_fake'] = self.load_img(depth_fake,W = self.W, H = self.H, type = 'depth', sr_multiple = self.sr_multiple).transpose(2,0,1)
            ret['depth_gt'] =  self.load_img(depth_gt,W = self.W, H = self.H, type = 'depth',sr_multiple = self.sr_multiple).transpose(2,0,1)
            # ret['occupancy_mask_sky' ] = self.load_occupancy_mask(occupancy_mask, semantic = 1,to_bin = True,erode = True).transpose(2,0,1) 
            # ret['occupancy_mask_sky_gt'] =  self.load_occupancy_mask(occupancy_mask_gt, semantic = 1,to_bin = True,erode = True).transpose(2,0,1) 

        if self.render_stuff:
            ret['stuff_semantic_grid'] = self.load_stuff_semantic(self.stuff_dict[frame_id], self.stuff_semantic_name)
            ret['stuff_loc_grid'] = self.stuff_loc_gird

            ret['stuff_semantic_grid_fake']= self.load_stuff_semantic(self.stuff_dict[frame_id_fake], self.stuff_semantic_name)
            ret['stuff_loc_grid_fake'] = self.stuff_loc_gird

        if self.render_obj:
            layout_gt = self.load_layout(self.layout_dict[frame_id_gt])
            ret["bbox_patches_gt"] = self.load_bbox_pathces(self.layout_patehcs_dict[frame_id_gt],self.patch_size, self.use_pacth_mask)[0].transpose(0,3,1,2)
            ret["bbox_patches_fake"] = self.load_bbox_pathces(self.layout_patehcs_dict[frame_id], self.patch_size, self.use_pacth_mask)[0].transpose(0,3,1,2)
            ret["bbox_patches"] = self.load_bbox_pathces(self.layout_patehcs_dict[3400], self.patch_size, self.use_pacth_mask)[0].transpose(0,3,1,2)
            ret["bbox_pose_patches"] = self.load_bbox_pathces(self.layout_patehcs_dict[frame_id_traj],self.patch_size)[1].transpose(0,3,1,2)
            
            ret["bbox_pose_patches_gt"] = self.load_bbox_pathces(self.layout_patehcs_dict[frame_id_gt],self.patch_size, self.use_pacth_mask)[1].transpose(0,3,1,2)
            ret["bbox_pose_patches_fake"] = self.load_bbox_pathces(self.layout_patehcs_dict[frame_id_traj_fake], self.patch_size)[1].transpose(0,3,1,2)


            ret['bbox_id_gt'] = np.array([bbox['globalId'] for bbox in layout_gt])
            ret['bbox_semantic_gt'] = np.array([bbox['semanticId'] for bbox in layout_gt])
            ret['bbox_tr_gt'] = np.array([bbox['bbox_tr'] for bbox in layout_gt]).astype(np.float32)
            ret['bbox_pose_gt'] = self.parse_bbox_pose(ret['bbox_tr_gt'], self.c2w_default).astype(np.float32)

            # layout = self.load_layout(self.layout_dict[frame_id])
            ret['bbox_id'] = np.array([bbox['globalId'] for bbox in layout])
            ret['bbox_semantic'] = np.array([bbox['semanticId'] for bbox in layout])
            ret['bbox_tr'] = np.array([bbox['bbox_tr'] for bbox in layout]).astype(np.float32)
            ret['bbox_pose'] = self.parse_bbox_pose(ret['bbox_tr'], ret['world_mat']).astype(np.float32)


            # layout_fake = self.load_layout(self.layout_dict[frame_id_fake])
            ret['bbox_id_fake'] = np.array([bbox['globalId'] for bbox in layout_fake])
            ret['bbox_semantic_fake'] = np.array([bbox['semanticId'] for bbox in layout_fake])
            ret['bbox_tr_fake'] = np.array([bbox['bbox_tr'] for bbox in layout_fake]).astype(np.float32)
            ret['bbox_pose_fake'] = self.parse_bbox_pose(ret['bbox_tr_fake'], ret['world_mat_fake']).astype(np.float32)





        return ret

    def __len__(self):
        return len(self.frame_id_list)


import json
from locale import normalize
import numpy as np
import os
import random
from yaml import load
from lib.config import cfg, args
import imageio
import re

import cv2
import pickle as pkl
import matplotlib.pyplot as plt
import trimesh
from math import sin, cos, tan
from lib.utils.transform_utils import RT2tr, create_R, rotate_mat2Euler, parse_R
from lib.utils.camera_utils import build_rays, project_3dbbox

name2id = {'ground' : 1,
              'wall' : 2,
              'object' : 26 }
name2color = {'ground' : 1,
              'wall' : 1,
              'object' : 2 }

idx2rgb = {
    1:[255, 0, 0],
    2:[0, 128, 0],
    3:[0, 0, 255],
    4:[255, 255, 0],
    5:[141, 211, 199],
    6:[255, 255, 179],
    7:[190, 186, 218],
    8:[251, 128, 114],
    9:[128, 177, 211]}

    
class Dataset:
    def __init__(self, data_root, seq_list, split):
        super(Dataset, self).__init__()
        # path and initialization
        self.data_root = data_root
        self.split = split
        self.render_obj = cfg.render_obj
        self.render_stuff = cfg.render_stuff
        self.render_sky = cfg.render_sky
        self.max_obj_num = cfg.max_obj_num
        self.valid_object = cfg.valid_object
        self.use_depth = cfg.use_depth

        self.stuff_semantic_name = cfg.stuff_semantic_list
        self.use_patch_discriminator = cfg.use_patch_discriminator
        self.patch_size = cfg.patch_size
        self.use_patch_occupancy_mask = cfg.use_patch_occupancy_mask

        obj_semantic_name = ['object']

        # self.road_semanticId_list = np.array([name2label[n].id for n in road_semantic_name ])
        self.obj_semanticId_list = np.array(sorted([name2id[n] for n in obj_semantic_name]))
        self.stuff_semanticId_list = np.array(sorted([name2id[n] for n in self.stuff_semantic_name]))

        # for GIRAFFE baseline 
        self.render_bg = True if cfg.task == 'GIRAFFE_clevr' else False

        # load intrinsics
        self.sr_multiple = cfg.super_resolution
        assert self.sr_multiple in [1,2,4]

        self.ratio = cfg.ratio
        self.height, self.width = 256, 256
        self.H = int(self.height * cfg.ratio)
        self.W = int(self.width  * cfg.ratio)
        self.H_ray = int(self.height * cfg.ratio / self.sr_multiple)
        self.W_ray = int(self.width  * cfg.ratio / self.sr_multiple)
        # parameters of intrinsic calibration matrix K
        camera_width, camera_height = 320, 240
        focal = 35.
        sensor_width, sensor_height = 32 ,18.0
        s_u = camera_width / sensor_width
        s_v = camera_height / sensor_height
        alpha_u = focal * s_u
        alpha_v = focal * s_v
        u_0 = camera_width / 2
        v_0 = camera_height / 2
        skew = 0 # only use rectangular pixels
        self.raw_intrinsic = np.array([
            [alpha_u,    skew, u_0],
            [      0, alpha_v, v_0],
            [      0,       0,   1]
        ])
        #! Why I need to exchange fx and fy here?    
        self.raw_intrinsic = np.array([
            [280.0,0.0,128.0],
            [0.0,280,128.0],
            [0.0,0.0,1.0]])
        # t = self.raw_intrinsic[0,0]
        # self.raw_intrinsic[0,0] = self.raw_intrinsic[1,1]
        # self.raw_intrinsic[1,1] = t
        self.intrinsic = self.raw_intrinsic.copy()
        self.intrinsic[:2] = self.intrinsic[:2] * (cfg.ratio / self.sr_multiple)

        self.max_obj_num = 6
        self.c2w_dict = {}
        self.rect_zy = np.array(
                    [[1,0,0,0],
                     [0,0,1,0],
                     [0,-1,0,0],
                     [0,0,0,1]]
                )
        # self.rays = build_rays(H = self.H_ray, W = self.W_ray, K = self.intrinsic, c2w= rect_mat)


        if self.render_stuff:
            x_max, x_min, y_max, y_min, z_max, z_min = 8, -8, 4, 0, 8, -8
            point_num_H, point_num_W, point_num_L, = 64, 64, 64
            vertices_gridx,  vertices_gridy, vertices_gridz= np.meshgrid(np.linspace(x_min, x_max, point_num_W), np.linspace(y_min, y_max, point_num_H), np.linspace(z_min, z_max, point_num_L), indexing='xy') 
            '''
            the indexing mode, either “xy” or “ij”, defaults to “ij”. See warning for future changes.

            If “xy” is selected, the first dimension corresponds to the cardinality of the second input and the second dimension corresponds to the cardinality of the first input.

            If “ij” is selected, the dimensions are in the same order as the cardinality of the inputs.
            '''
            self.stuff_loc_gird = np.concatenate((vertices_gridx[:,:,:,None], vertices_gridy[:,:,:,None], vertices_gridz[:,:,:,None]), axis = -1)#! shape [H, W, L, 3]
        # Load object laouot, road seamntic plane and 
        idx2frame = []
        self.layout_dict = {}
        self.bbox_patehcs_dict = {}
        self.frame_id_list = []
        self.rgb_image_dict, self.depth_image_dict = {}, {}
        self.seg_image_dict = {}
        self.stuff_dict = {}
        self.c2w_dict = {}
        N = 1000
        for seq_num in seq_list:
            metadata_dir = os.path.join(data_root, '%d'%seq_num, 'Metadata')
            # /data/ybyang/clevrtex_train/1/Metadata
            # /data/ybyang/clevrtex_train/1/Metadata
            self.metadata_dict = {int(s[16:21]) : os.path.join(metadata_dir, s) for s in os.listdir(metadata_dir)}

            for idx in self.metadata_dict:
                rgb_path = os.path.join(data_root, '%d'%seq_num, 'RGB', 'CLEVRTEX_train_%06d.png'%(idx))
                seg_path = os.path.join(data_root, '%d'%seq_num, 'Mask', 'CLEVRTEX_train_%06d_mask.png'%(idx))
                depth_path = os.path.join(data_root, '%d'%seq_num, 'Depth', 'CLEVRTEX_train_%06d_depth_0001.png'%(idx))
                if (not os.path.exists(rgb_path)) or (not os.path.exists(seg_path)):
                    continue
                self.rgb_image_dict[idx] = rgb_path
                self.depth_image_dict[idx] = depth_path
                self.seg_image_dict[(idx)] = seg_path


                with open(self.metadata_dict[(idx)],'rb+') as f:
                    metadata = json.load(f)
                if self.use_patch_discriminator:
                    obj_patches_root = os.path.join(data_root, '%d'%seq_num, 'Patches', '%d'%idx)

                # self.c2w_dict[idx] = T @ self.rect_zy @ R 
                # a = np.array(metadata['extrinsic'])
                K = np.array(metadata['intrinsic'])
                c2w_blender = np.array(metadata['extrinsic'])
                # c2w_blender[:3,0] = -c2w_blender[:3,0]
                c2w_blender[:3,1] = -c2w_blender[:3,1]
                c2w_blender[:3,2] = -c2w_blender[:3,2]
                self.c2w_dict[idx] = self.rect_zy @ c2w_blender

                # if self.render_obj:
                objects = metadata['objects']
                layout = []
                bbox_id = 0
                for o in objects:
                    if o['shape'] == 'wall':
                        continue
                    tr = np.zeros((4,4))
                    tr[3,3] = 1
                    tr[0:3,3] = o['3d_coords']
                    R = create_R(rotate=(0, 0., o['rotation'] * 3.14 / 180.), scale = np.array((2.,2.,2.)) * o['size'])
                    tr[0:3,0:3] = R
                    tr = self.rect_zy @ tr
                    layout += [{'bbox_id': o['index'], 'bbox_tr' : tr, 'bbox_semantic': name2id['object'], 'bbox_shape':o['shape'], 'color':o['color']}]
                    bbox_id += 1

                layout += [{'bbox_id': -1, 'bbox_tr' : np.eye(4), 'bbox_semantic': -1, 'bbox_shape':'None','color':o['color']}] * (self.max_obj_num - len(layout))

                self.layout_dict[idx] = layout

                if self.render_stuff:
                    stuff_voxel_path = os.path.join(data_root, '%d'%seq_num, 'Voxel', 'CLEVRTEX_train_%06d.pkl'%(idx))
                    if not os.path.exists(stuff_voxel_path):
                        continue
                    self.stuff_dict[idx] = stuff_voxel_path

                if self.use_patch_discriminator:
                    self.bbox_patehcs_dict[idx] = self.load_patches_path(seq_num=seq_num, idx = idx, layout= layout, obj_patches_root=obj_patches_root)

                self.frame_id_list += [idx]
                    

            print('Load clevr Stuff seq%d Done!!'%seq_num)

        if split == 'render' and cfg.render.render_given_frame:
            self.frame_id_list = [i for i in self.frame_id_list if i in cfg.render.render_frame_list]
        if split == 'test' and cfg.test.test_given_frame:
            self.frame_id_list = [i for i in self.frame_id_list if i in cfg.test.test_frame_list]

    
        frame_id_list_gt = self.frame_id_list.copy()
        frame_id_list_fake = self.frame_id_list.copy()

        random.shuffle(frame_id_list_gt)
        random.shuffle(frame_id_list_fake)

        self.frame_id_list_gt =frame_id_list_gt
        self.frame_id_list_fake = frame_id_list_fake

        self.idx2frame = {idx : frame  for idx,frame in enumerate(self.frame_id_list)}
        self.frame2idx = {frame : idx  for idx,frame in enumerate(self.frame_id_list)}


        print('Load dataset done!')

    def load_patches_path(self, seq_num, idx, layout, obj_patches_root):
        # Empty layout
        patches_path = []
        if layout[0]['bbox_id'] == -1:
            pass
        else:
            obj_patches_dir = os.path.join(obj_patches_root, '%010d'%idx)
            for o in layout:
                if o['bbox_id'] == -1:
                    break
                rgb_patch_path = os.path.join(obj_patches_dir, '%05d_rgb.jpg'%o['bbox_id'])
                if not os.path.exists(rgb_patch_path):
                    seg_img = imageio.imread(self.seg_image_dict[idx])[...,:3]
                    rgb_img = imageio.imread(self.rgb_image_dict[idx])[...,:3]

                    a = np.unique(seg_img.reshape(-1,3), axis=0)
                    self.make_obj_patch(o, 
                                        rgb_img= rgb_img,
                                        seg_img= seg_img,
                                        patch_dir= obj_patches_dir,
                                        patch_size= self.patch_size,
                                        c2w = self.c2w_dict[idx])

                patches_path += [rgb_patch_path]
            # Patches already exist
        # Padding
        patches_path += ['/'] * (len(layout) - len(patches_path))
        return patches_path

    def load_stuff_path(self, seq_num, frame_ids,grid_param = (16,64,64,64,64,64) ,  N_seq = 100000):
        stuff_dict =  {}
        #assert frame_ids ==  list(layout_dict.keys())
        H, W, L, H_num, W_num, L_num = grid_param

        sequence = os.path.join('2013_05_28_drive_' + '%04d'%seq_num + '_sync')
        stuff_root = os.path.join(self.data_root, 'semantic_voxel', sequence) 

        for idx in frame_ids:
            #rgb_image_path = imageio.imread(os.path.join(rgb_image_root, '%010d.png'%idx))
            stuff_file_path = os.path.join(stuff_root,  '(H:%d:%d,W%d:%d,L%d:%d)'%(H,H_num, W, W_num, L,L_num), '{:010d}.pkl'.format(idx))
            if not os.path.exists(stuff_file_path):
                self.stuff_dict[idx + seq_num * N_seq] = ''
                #raise RuntimeError('%s does not exist!' % stuff_file_path)
            else:
                self.stuff_dict[idx + seq_num * N_seq] = stuff_file_path


    def make_obj_patch(self, obj, rgb_img, seg_img, patch_dir, patch_size = 64, vertex_threshold = 5, occupancy_ratio_threshold = 0.25, c2w = None):
        obj_globalId,  obj_tr, obj_color = obj['bbox_id'], obj['bbox_tr'], np.array(idx2rgb[obj['bbox_id']])
        # valid_vertex_num, _ = project_3dbbox(obj_tr, self.c2w_default, self.raw_intrinsic, render = False)
        # if valid_vertex_num < vertex_threshold:
        #     return False, '/'
        
        rgb_patch = os.path.join(patch_dir, '%05d_rgb.jpg'%(obj_globalId))
        pose_patch = os.path.join(patch_dir, '%05d_pose.jpg'%obj_globalId)
        occupancy_patch =  os.path.join(patch_dir, '%05d_occupancy.jpg'%obj_globalId)

        
        if os.path.exists(rgb_patch):
            pose_img  = imageio.imread(pose_patch)
            obj_rgb = imageio.imread(rgb_patch)
            obj_occupancy = imageio.imread(occupancy_patch)
        else:
            _, (u_min , u_max, v_min , v_max) = project_3dbbox(obj_tr, c2w, self.raw_intrinsic, H = 256, W = 256, render = False, is_kitti=False)

            occupancy_img = np.all(seg_img == obj_color, axis=-1)[...,None]

            rgb_img = cv2.resize(rgb_img, (self.W, self.H), interpolation=cv2.INTER_LANCZOS4)
            occupancy_img = cv2.resize(occupancy_img * 50., (self.W, self.H), interpolation=cv2.INTER_NEAREST)
            # pose_img = cv2.resize(pose_img, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
            # imageio.imsave('tmp/aa.jpg',rgb_img_)
            v_min , v_max, u_min, u_max = \
                int(v_min * self.ratio)  , int(v_max * self.ratio) , int(u_min * self.ratio),int(u_max * self.ratio)
            
            obj_rgb = rgb_img[v_min :v_max ,u_min :u_max ]
            obj_occupancy = occupancy_img[v_min :v_max ,u_min :u_max ]
            # obj_pose = pose_img[v_min :v_max ,u_min :u_max ]


            h_patch, w_patch=  obj_rgb.shape[0], obj_rgb.shape[1]
            if h_patch <= w_patch:
                pad_num = int((w_patch - h_patch)/2)
                obj_rgb = cv2.copyMakeBorder(obj_rgb,pad_num,pad_num,0,0, cv2.BORDER_CONSTANT, value=(0,0,0))
                obj_occupancy = cv2.copyMakeBorder(obj_occupancy,pad_num,pad_num,0,0, cv2.BORDER_CONSTANT, value=(0))
                # obj_pose = cv2.copyMakeBorder(obj_pose,pad_num,pad_num,0,0, cv2.BORDER_CONSTANT, value=(1,1,1))
            else: 
                pad_num = int((h_patch - w_patch)/2)
                obj_rgb = cv2.copyMakeBorder(obj_rgb,0,0,pad_num,pad_num, cv2.BORDER_CONSTANT, value=(0,0,0))
                obj_occupancy = cv2.copyMakeBorder(obj_occupancy,0,0,pad_num,pad_num, cv2.BORDER_CONSTANT, value=(0))
                # obj_pose = cv2.copyMakeBorder(obj_pose,pad_num,pad_num,0,0, cv2.BORDER_CONSTANT, value=(255,255,255))

            
            imageio.imsave('tmp/bb.jpg',obj_rgb)
            obj_rgb = cv2.resize(obj_rgb, (patch_size,patch_size), interpolation=cv2.INTER_AREA)
            obj_occupancy = cv2.resize(obj_occupancy, (patch_size,patch_size), interpolation=cv2.INTER_NEAREST)
            # obj_pose = cv2.resize(obj_pose, (patch_size,patch_size), interpolation=cv2.INTER_NEAREST)

        occupancy_ratio = np.sum(obj_occupancy / 50) / (patch_size * patch_size)

        if True :
            if not os.path.exists(patch_dir):
                os.makedirs(patch_dir)
            if not os.path.exists(rgb_patch):
                # imageio.imsave(pose_patch,obj_pose)
                imageio.imsave(rgb_patch,obj_rgb)
                imageio.imsave(occupancy_patch,obj_occupancy)
            return True, rgb_patch
        else:
            return False, '/'



    @staticmethod
    def load_layout(path, tr_mat = np.eye(4,4)):
        with open(path, 'rb+') as f:
            layout = pkl.load(f)

        for bbox in layout:
            bbox['bbox_tr'] = tr_mat @ bbox['bbox_tr']
        return layout

    @staticmethod
    def load_img(path, W = 128, H = 128, type = 'rgb'):
        image = imageio.imread(path).astype(np.float32)
        # /data/ybyang/KITTI-360/data_2d_depth/2013_05_28_drive_0000_sync/image_00/depth_rect/0000000320.png
        if type == 'rgb':
            rgb_image = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.
            return rgb_image[:,:,:3]
        elif type == 'depth':
            z_near, z_far = 0.001, cfg.z_far
            depth_image = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)
            # print(np.max(depth_image))
            depth_image = (depth_image - z_near) / (z_far - z_near)
            depth_image = np.clip(depth_image, 0., 1.)
            return np.expand_dims(depth_image, -1)
        else:
            raise KeyError


    @staticmethod
    def load_occupancy_mask(path, W = 128, H = 128, semantic = -1, to_bin = False,erode = False):
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
        
        with open(path, 'rb+') as fp: 
            voxel = pkl.load(fp)
        H, W, L = voxel['(H,W,L)']
        stuff_semantic_idx = voxel['semantic']
        stuff_semantic = np.zeros(H * W * L)
        for s in semantic_list:
            if stuff_semantic_idx[s].shape[0] == 0:
                continue
            stuff_semantic[stuff_semantic_idx[s]] = name2id[s]
         
        return stuff_semantic.reshape(H, W, L)


    @staticmethod
    def load_bbox_pathces(patches_dir,patch_size, use_mask = True, cat_mask = False,):

        patches_img, poses_img = [], []

        for patch_path in patches_dir:
            if patch_path == '/':
                patches_img.append(np.zeros((patch_size,patch_size,3)))
                poses_img.append(np.zeros((patch_size,patch_size,3)))
            else:
                patch = imageio.imread(patch_path) / 255
                # pose_path = patch_path.replace('rgb', 'pose')
                # pose = imageio.imread(pose_path) / 255.
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
                # poses_img.append(pose)

        patches_img = np.stack(patches_img).astype(np.float32)
        # poses_img = np.stack(poses_img).astype(np.float32)

        return patches_img, 0 
    
    # @staticmethod
    def parse_bbox_pose(self, bbox_trs, camera_pose):

        bbox_scale , bbox_rotate, _ = parse_R(bbox_trs[:,:3,:3],True)
        bbox_translate = bbox_trs[:,:3,3]
        _, camera_rotate, _ = parse_R(camera_pose[:3,:3], True)
        camera_translate = camera_pose[None,:3,3]
        

        camera_translate_bbox = (bbox_rotate.transpose(0,2,1) @ (camera_translate - bbox_translate)[:,:,None])[:,:,0] / bbox_scale
        camera_rotate_bbox = bbox_rotate.transpose(0,2,1) @ camera_rotate
        camera_rotate_bbox = camera_rotate_bbox.reshape((camera_rotate_bbox.shape[0],-1))
        

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

        c2w = self.c2w_dict[frame_id]
        rays = build_rays(self.H_ray, self.W_ray, self.intrinsic, self.c2w_dict[frame_id],z_reverse = False)
        rgb = self.rgb_image_dict[frame_id]
        if self.use_depth:
            depth =  self.depth_image_dict[frame_id]
            depth_fake =  self.depth_image_dict[frame_id_fake]

        # if self.render_obj:
        #     layout = self.load_layout(self.layout_dict[frame_id])
        #     layout_fake = self.load_layout(self.layout_dict[frame_id_fake])

        # Load fake scene for D training
        frame_id_fake = self.frame_id_list_fake[index]
        idx_fake = self.frame2idx[frame_id_fake]

        c2w_fake = self.c2w_dict[frame_id_fake]
        rays_fake = build_rays(self.H_ray, self.W_ray, self.intrinsic, self.c2w_dict[frame_id_fake],z_reverse = False)
        rgb_fake = self.rgb_image_dict[frame_id_fake]
        depth_fake = self.depth_image_dict[frame_id_fake]

            

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
        if self.use_depth:
            ret['depth'] = self.load_img(depth,W = self.W, H = self.H, type = 'depth').transpose(2,0,1)
            ret['depth_fake'] = self.load_img(depth_fake,W = self.W, H = self.H, type = 'depth').transpose(2,0,1)
            ret['depth_gt'] =  self.load_img(depth_gt,W = self.W, H = self.H, type = 'depth').transpose(2,0,1)

        if self.render_stuff:
            ret['stuff_semantic_grid'] = self.load_stuff_semantic(self.stuff_dict[frame_id], self.stuff_semantic_name)
            ret['stuff_loc_grid'] = self.stuff_loc_gird
            ret['stuff_semantic_grid_fake']= self.load_stuff_semantic(self.stuff_dict[frame_id_fake], self.stuff_semantic_name)
            ret['stuff_loc_grid_fake'] = self.stuff_loc_gird

        if self.render_obj:
            # layout_gt = self.load_layout(self.layout_dict[frame_id_gt])
            if self.use_patch_discriminator:
                ret["bbox_patches_gt"] = self.load_bbox_pathces(self.bbox_patehcs_dict[frame_id_gt],self.patch_size, use_mask=self.use_patch_occupancy_mask)[0].transpose(0,3,1,2)
            # ret["bbox_pose_patches"] = self.load_bbox_pathces(self.layout_patehcs_dict[frame_id],self.patch_size)[1].transpose(0,3,1,2)
            # ret["bbox_pose_patches_gt"] = self.load_bbox_pathces(self.layout_patehcs_dict[frame_id_gt],self.patch_size)[1].transpose(0,3,1,2)
            # ret["bbox_pose_patches_fake"] = self.load_bbox_pathces(self.layout_patehcs_dict[frame_id_fake], self.patch_size)[1].transpose(0,3,1,2)

            # a = self.layout_dict[1367]
            ret['bbox_id_gt'] = np.array([bbox['bbox_id'] for bbox in self.layout_dict[frame_id_gt]])
            ret['bbox_semantic_gt'] = np.array([bbox['bbox_semantic'] for bbox in self.layout_dict[frame_id_gt]])
            ret['bbox_tr_gt'] = np.array([bbox['bbox_tr'] for bbox in self.layout_dict[frame_id_gt]]).astype(np.float32)
            ret['bbox_pose_gt'] = self.parse_bbox_pose(ret['bbox_tr_gt'], c2w).astype(np.float32)

            # layout = self.load_layout(self.layout_dict[frame_id])
            ret['bbox_id'] = np.array([bbox['bbox_id'] for bbox in self.layout_dict[frame_id]])
            ret['bbox_semantic'] = np.array([bbox['bbox_semantic'] for bbox in self.layout_dict[frame_id]])
            ret['bbox_tr'] = np.array([bbox['bbox_tr'] for bbox in self.layout_dict[frame_id]]).astype(np.float32)
            ret['bbox_pose'] = self.parse_bbox_pose(ret['bbox_tr'], ret['world_mat']).astype(np.float32)

            # layout_fake = self.load_layout(self.layout_dict[frame_id_fake])
            ret['bbox_id_fake'] = np.array([bbox['bbox_id'] for bbox in self.layout_dict[frame_id_fake]])
            ret['bbox_semantic_fake'] = np.array([bbox['bbox_semantic'] for bbox in self.layout_dict[frame_id_fake]])
            ret['bbox_tr_fake'] = np.array([bbox['bbox_tr'] for bbox in self.layout_dict[frame_id_fake]]).astype(np.float32)
            ret['bbox_pose_fake'] = self.parse_bbox_pose(ret['bbox_tr_fake'], ret['world_mat_fake']).astype(np.float32)

        return ret

    def __len__(self):
        return len(self.frame_id_list)

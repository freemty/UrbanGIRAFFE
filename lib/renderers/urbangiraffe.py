import torch
import numpy as np
# from im2scene.common import interpolate_sphere
from torchvision.utils import save_image, make_grid
import imageio
from math import sqrt
import os
from os import makedirs
from os.path import join
from lib.utils.camera_utils import build_rays
from lib.utils.img_utils import save_tensor_img, make_gif, make_video
from lib.utils.transform_utils import tr2RT, parse_R, create_R
from tools.kitti360Scripts.helpers.labels import name2label
from numpy import pi
import pickle as pkl
# valid_task = ['move_forward']
import copy
import shutil

class Renderer(object):
    '''  Render class for UrbanGIRAFFE.

    It provides functions to render the representation.

    Args:
        model (nn.Module): trained GIRAFFE model
        device (device): pytorch device
    '''

    def __init__(self,cfg):

        self.task_list = cfg.render.task_list
        self.z_dim_global = cfg.network_kwargs.generator_kwargs.z_dim_global
        self.z_dim_obj = cfg.network_kwargs.generator_kwargs.z_dim_obj
        self.step_num =  cfg.render.step_num

        self.data_root = cfg.render_dataset.data_root
        self.stuff_semantic_list = cfg.stuff_semantic_list
        self.exp_name = cfg.exp_name
        self.H, self.W = int(cfg.img_size_raw[0]) * cfg.ratio , int(cfg.img_size_raw[1] *cfg.ratio)
        self.H_ray, self.W_ray = int(self.H / cfg.super_resolution),  int(self.W / cfg.super_resolution)
        self.is_kitti = cfg.is_kitti360
        self.fps = cfg.render.render_fps


        # self.render_batch_size = cfg.render_batch_size
    def render_tasks(self, network, raw_batch, out_dir = 'tmp', using_G_test = True):
        '''
        '''
        step_num = self.step_num
        if using_G_test:
            generator = network.generator_test
        else: 
            generator = network.generator

        self.out_dir = out_dir
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        for task in self.task_list:
            # self.set_random_seed()
            # Render Demo4Paper
    #?----------------------------------------------------------------------------------
    #?---------------------------Paper-------------------------------------------------
            if task == 'render_teaser':
                self.render_teaser(generator, raw_batch, out_dir, step_num, fix_scene = True, fix_obj= True, task_name='render_teaser') # 3400
            if task == 'render_rebuttal':
                self.render_rebuttal(generator, raw_batch, out_dir, step_num, fix_scene = True, fix_obj= True, task_name='rebuttal') # 3400
            if task == 'render_method':
                self.render_method(generator, raw_batch, out_dir, step_num, fix_scene = True, fix_obj= True, task_name='render_method') # 3552
            if task == 'render_camera_figure_kitti360':
                self.render_camera_figure_kitti360(generator, raw_batch, out_dir, task_name='camera_figure_kitti360')
            if task == 'render_object_figure_kitti360':
                self.render_object_figure_kitti360(generator, raw_batch, out_dir, task_name='object_figure_kitti360')
            if task == 'render_stuff_figure_kitti360':
                self.render_building_to_tree(generator, raw_batch, out_dir, step_num, fix_scene = True, fix_obj= True, task_name='building_to_tree')
                self.render_road_to_grass(generator, raw_batch, out_dir, step_num, fix_scene = True, fix_obj= True, task_name='road_to_grass')
                self.render_building_lower(generator, raw_batch, out_dir, step_num, fix_scene = True, fix_obj= True, task_name='stuff_higher')
                self.render_stuff_translate(generator, raw_batch, out_dir, step_num, fix_scene = True, fix_obj= True, task_name='stuff_translate')
            if task == 'render_ablation_figure_kitti360':
                self.render_ablation_figure_kitti360(generator, raw_batch, out_dir, task_name='ablation_figure_kitti360')
            # if task == 'render_generalization_figure_kitti360':
            #     self.render_generalization_figure_kitti360(generator, raw_batch, out_dir, task_name='generalization_figure_kitti360')

        #?--------------------------Video----------------------------------------------
            if task == 'move_forward_video':
                self.render_move_forward_video(generator, raw_batch, out_dir, step_num = 40, distance = 40, task_name='move_forward_video', fix_scene=True)
            if task == 'move_forward_video_compare':
                self.render_move_forward_video(generator, raw_batch, out_dir, step_num = 40, distance = 40,for_compare = True, task_name='move_forward_video_compare')
            if task == 'limitaion':
                self.render_move_forward_video(generator, raw_batch, out_dir, step_num = 10, distance = 0,for_compare = True, task_name='limitaion', fix_scene=True)
            if task == 'elevate_camera_video':
                self.render_elevate_camera_video(generator, raw_batch, out_dir, step_num = 40, task_name='elevate_camera_video')
            if task == 'interpolate_camera_video':
                self.render_interpolate_camera_video(generator, raw_batch, out_dir, fix_scene = True, fix_obj = True, task_name = 'interpolate_camera_video')
            if task == 'building2tree_video':
                self.render_building2tree_video(generator, raw_batch, out_dir, fix_scene = True, fix_obj = True, task_name = 'building2tree_video')
            if task == 'road2grass_video':
                self.render_road2grass_video(generator, raw_batch, out_dir, fix_scene = True, fix_obj = True, task_name = 'road2grass_video')
            if task == 'building_lower_video':
                 self.render_building_lower_video(generator, raw_batch, out_dir, step_num = 20, fix_scene = True, fix_obj= True, task_name='building_lower_video')
            if task == 'move_tree_video':
                 self.render_move_tree_video(generator, raw_batch, out_dir, step_num = 20, fix_scene = True, fix_obj= True, task_name='move_tree_video')
            if task == 'object_editing_video':
                 self.render_object_editing_video(generator, raw_batch, out_dir, step_num = 20, fix_scene = True, fix_obj= True, task_name='object_editing_video')
            if task == 'clevr_video':
                 self.render_clevr_video(generator, raw_batch, out_dir, step_num = 20, fix_scene = True, fix_obj= True, task_name='clevr_video')
            if task == 'teaser_video1':
                self.render_teaser_video1(generator, raw_batch, out_dir, step_num, fix_scene = True, fix_obj= True, task_name='render_teaser_video1') #
            if task == 'teaser_video2':
                self.render_teaser_video2(generator, raw_batch, out_dir, step_num, fix_scene = True, fix_obj= True, task_name='render_teaser_video2') #

#?----------------------------------------------------------------------------------
#?--------------------------Figure-------------------------------------------
#?----------------------------------------------------------------------------------
    def render_method(self, generator,raw_data, out_dir, step_num, fix_scene = True, fix_obj = True, task_name = 'teaser'):

        new_tr1 = np.array([[ 0.1933, -2.0816,  0.0884, -1.618],
          [-0.0514, -0.1277, -1.4529, -0.8732],
          [ 4.9960,  0.0792, -0.0184, 10.7895],
          [ 0.0000,  0.0000,  0.0000,  1.0000]])


        raw_batch = {k : raw_data[k].clone() for k in  raw_data}
        for i in range(20):
            raw_batch =  self.assign_latentcodes(raw_batch, fix_scene, fix_obj)

        task_batch = []
        task_batch.append(raw_batch)

        out =  self.batchify_render(task_batch= task_batch, generator= generator, valid_keys=['frame_id', 'camera_intrinsic', 'camera_pose', 'rgb', 'rgb_gt', 'feat_raw', 'semantic_gird', 'feature_grid', 'bbox_tr', 'bbox_semantic', 'bbox_patches']) 
        

        valid_idx = torch.argwhere(out['bbox_semantic'] != -1)
        patches_fake = out['bbox_patches'][valid_idx[:,0],valid_idx[:,1]]
        for i in range(patches_fake.shape[1]):
            save_tensor_img(patches_fake[0,i], 'tmp', '%d.jpg'%i)
        for i in range(4):
             save_tensor_img(raw_batch['bbox_patches_gt'][0][i], 'tmp', '%dgt.jpg'%i)
             save_tensor_img(raw_batch['bbox_patches_fake'][0][i], 'tmp', '%dfake.jpg'%i)
             save_tensor_img(raw_batch['bbox_patches'][0][i], 'tmp', '%dfae.jpg'%i)
        # pose_patches_fake = fake_output['bbox_pose_patches'][valid_idx[:,0],valid_idx[:,1]]
        # patches_fake = torch.cat((patches_fake, pose_patches_fake), dim=1)

        self.save_img(out_dir, out, task_name, valid_keys=['rgb', 'rgb_gt'])
        
        self.save_scene(out_dir, out, task_name)

        return task_batch

    def render_teaser(self, generator,raw_data, out_dir, step_num, fix_scene = True, fix_obj = True, task_name = 'teaser'):
        new_tr1 = np.array([[ 0.1933, -2.0816,  0.0884, 2.4618],
          [-0.0514, -0.1277, -1.4529, -0.8732],
          [ 4.9960,  0.0792, -0.0184, 17.7895],
          [ 0.0000,  0.0000,  0.0000,  1.0000]])
        new_tr2 = np.array([[ 0.1933, -2.0816,  0.0884, -2.4618],
          [-0.0514, -0.1277, -1.4529, -0.8732],
          [ 4.9960,  0.0792, -0.0184, 17.7895],
          [ 0.0000,  0.0000,  0.0000,  1.0000]])
        new_tr3 = np.array([[ 0.1933, -2.0816,  0.0884, 0.4618],
          [-0.0514, -0.1277, -1.4529, -0.8732],
          [ 4.9960,  0.0792, -0.0184, 20.7895],
          [ 0.0000,  0.0000,  0.0000,  1.0000]])

        raw_batch = {k : raw_data[k].clone() for k in  raw_data}
        for i in range(20):
            raw_batch =  self.assign_latentcodes(raw_batch, fix_scene, fix_obj)

        task_batch = []

        # add_obj_batch = self.add_obj(add_obj_batch, new_tr2)
        task_batch.append(raw_batch)

        if True:
            for i in range(10):
                rate = (i + 1) / 10
                raw_batch = self.modify_camera(raw_batch, rotate_angle=(0,0,0), translate = (0,0,7 / 20))
                task_batch.append(raw_batch)
        else:
            raw_batch = self.modify_camera(raw_batch, rotate_angle=(0,0,0), translate = (0,0,7))
        # raw_batch = self.modify_camera(raw_batch, rotate_angle=(-20 * 3.14 / 180,0,0), translate = (0,-2.5,5))
            task_batch.append(raw_batch)
        

        raw_batch = self.add_obj(raw_batch, new_tr1, idx=-1)
        task_batch.append(raw_batch)
        raw_batch = self.add_obj(raw_batch, new_tr2, idx = -1)
        task_batch.append(raw_batch)
        raw_batch = self.add_obj(raw_batch, new_tr3, idx = -1)
        task_batch.append(raw_batch)
        raw_batch = self.add_obj(raw_batch, idx = 0, operation='delate')
        task_batch.append(raw_batch)
        for i in range(20):
            raw_batch =  self.assign_latentcodes(raw_batch, False, fix_obj)
        task_batch.append(raw_batch)

        # for i, scene in enumerate(task_batch):
        #     self.save_scene(scene, '%d'%i)
        # raw_batch = self.modify_camera(raw_batch, rotate_angle=(20 * 3.14 / 180,0,0), translate = (0,2.5,2))
        
        for i in range(4):
             raw_batch = self.edit_stuff(raw_batch, i = i, raw_stuff=11, target_stuff=21)
             task_batch.append(raw_batch)
        raw_batch = self.edit_stuff(raw_batch, i = -1, raw_stuff=11, target_stuff=21)
        # building2tree_batch = self.render_edit_stuff(task_batch[-1], i = 4, raw_stuff=11, target_stuff=21)
        task_batch.append(raw_batch) 

        out =  self.batchify_render(task_batch= task_batch, generator= generator, valid_keys=['camera_intrinsic','camera_pose', 'rgb', 'rgb_gt', 'feat_raw','semantic_gird','feature_grid','bbox_tr','bbox_semantic']) 
        self.save_img(out_dir, out, task_name)
        self.save_scene(out_dir, out, task_name)

        return task_batch

    def render_object_figure_kitti360(self,generator,raw_data, out_dir, task_name = 'object_figure_kitti360'):

        new_tr1, new_tr2 = np.eye(4), np.eye(4)
        new_tr2[:3,:3] = create_R((90 * 3.14 / 180,90 * 3.14 / 180,0),(6,2,1.85))
        new_tr2[:3,3] = np.array((2,-0.8,8))
        new_tr1[:3,:3] = create_R((90 * 3.14 / 180,90 * 3.14 / 180,0),(5,2,1.45))
        new_tr1[:3,3] = np.array((0,-0.8,10))

        raw_batch = {k : raw_data[k].clone() for k in  raw_data}
        # raw_batch =  self.assign_latentcodes(raw_batch, True,False)

        task_batch = []


        raw_batch =  self.assign_latentcodes(raw_batch, True,True)

        task_batch.append(raw_batch)
 
        raw_batch  = self.add_obj(raw_batch, new_tr1)
        add_batch = self.add_obj(raw_batch, new_tr2)
        task_batch.append(add_batch)

        task_batch.append(raw_batch)

        for i in range(3):
            raw_batch =  self.assign_latentcodes(raw_batch, True,True)
        task_batch.append(raw_batch)

        a_batch = self.rotate_obj(raw_batch, rotate_angle=(0,25 * 3.14 / 180,0), index=-1)
        task_batch.append(a_batch)

        b_batch = self.rotate_obj(a_batch,rotate_angle=(0,-25 * 3.14 / 180,0),translate=(1,0,0),  index=0)
        # task_batch.append(raw_batch) 
        b_batch = self.rotate_obj(b_batch,rotate_angle=(0,25 * 3.14 / 180,0),translate=(1,0,0),  index=1)
        task_batch.append(b_batch) 

        raw_batch = self.rotate_obj(raw_batch, rotate_angle=(0,0,0),translate=(0,0,-3), index=1)
        task_batch.append(raw_batch)
        raw_batch = self.rotate_obj(raw_batch,rotate_angle=(0,0,0), translate=(0,0,5), index=-1)
        task_batch.append(raw_batch) 

        out =  self.batchify_render(task_batch= task_batch, generator= generator) 

        self.save_img(out_dir, out, task_name,)
        self.save_scene(out_dir, out, task_name)

        return task_batch
  
    def render_camera_figure_kitti360(self,generator,raw_data, out_dir, task_name = 'camera_figure_kitti360'):
        raw_batch = {k : raw_data[k].clone() for k in  raw_data}
        # raw_batch = self.add_obj(raw_batch, name = 'car')
        raw_batch =  self.assign_latentcodes(raw_batch,True,True)
        task_batch = []

        # Rotate camera
        r_gamma = self.modify_camera(raw_batch ,rotate_angle=(0,-25 * 3.14/180,0), translate = (0,0,0))
        task_batch.append(r_gamma)
        r_alpha = self.modify_camera(raw_batch ,rotate_angle=(0,0,20 * 3.14/180), translate = (0,0,0))
        task_batch.append(r_alpha)

        # elevate
        elevate_batch = self.modify_camera(raw_batch ,rotate_angle=(-20 * 3.14 / 180,0,0), translate = (0,-2.5,0))
        task_batch.append(elevate_batch)
        elevate_batch = self.modify_camera(elevate_batch ,rotate_angle=(15 * 3.14 / 180,0,0), translate = (0,0,0))
        K = elevate_batch['camera_mat'].cpu().numpy()
        K[:,0,0] *= 0.6
        K[:,1,1] *= 0.8
        elevate_batch= self.modify_camera(elevate_batch, K = K)

        task_batch.append(elevate_batch)


        out =  self.batchify_render(task_batch= task_batch, generator= generator)
        self.save_img(out_dir, out, task_name)

    def render_ablation_figure_kitti360(self,generator, raw_data, out_dir, task_name = 'ablation_figure_kitti360'):
        raw_batch = {k : raw_data[k].clone() for k in  raw_data}
        # raw_batch = self.add_obj(raw_batch, name = 'car')
        raw_batch =  self.assign_latentcodes(raw_batch,True,True)
        task_batch = []

 
        task_batch.append(raw_batch)
        # raw_batch = self.modify_camera(raw_batch, translate = (0,0,0))
        # task_batch.append(raw_batch)
        out =  self.batchify_render(task_batch= task_batch, generator= generator, valid_keys=['rgb','rgb_gt'])
        self.save_img(out_dir, out, task_name)

    # def render_generalization_figure_kitti360(self,generator, raw_data, out_dir, task_name = 'ablation_generalization_kitti360'):
    #     raw_batch = {k : raw_data[k].clone() for k in  raw_data}
    #     # raw_batch = self.add_obj(raw_batch, name = 'car')
    #     for i in range(20): 
    #         raw_batch =  self.assign_latentcodes(raw_batch,True,True)
    #     task_batch = []

    #     kittivoxel_path_list = os.listdir('/data/ybyang/kittivoxel4render')
    #     kittivoxel_path_list.sort()
    #     kittivoxel_path_list = kittivoxel_path_list
    #     # K = raw_batch['camera_mat'].cpu().numpy()
    #     # K[:,0,0] = 707.0912 * 0.25
    #     # K[:,1,1] = 707.0912 * 0.25
    #     # K[:,0,2] = 601.8873 * 0.25
    #     # K[:,1,2] = 183.1104 * 0.25
    #     # path = kittivoxel_path_list[0]

    #     from voxel_convert import convert2badvox
    #     N2S_mat = np.array(
    #                 [[0,-1,0,0],
    #                 [0,0,-1,0],
    #                 [1,0,0,0],
    #                 [0,0,0,1]])
    #     for i, path in  enumerate(kittivoxel_path_list[:]):
    #         with open(os.path.join('/data/ybyang/kittivoxel4render', path), 'rb') as fp:
    #             k_data = pkl.load(fp)
    #             normalvoxel = k_data['stuff_semantic_voxel'].transpose((1,0,2)) # X, Y, Z == L, W, H
    #             # normal_loc = k_data['stuff_semantic_voxel']
    #             badvoxel = convert2badvox(normalvoxel[:,:,:,None])[...,0]

    #         batch_i = {k : raw_batch[k].clone() for k in raw_batch if k!= 'stuff_semantic_grid'}
    #         # self.H_ray, self.W_ray = int(k_data['HW'][0]/ 4) , int(k_data['HW'][1] / 4)
    #         # c2w =  np.array([[1,0,0,0], [0,0,1,0], [0,-1,0,0],[0,0,0,1]]) @np.linalg.inv(k_data['T_velo_2_cam'])
    #         # del batch_i['world_mat'], batch_i['rays'] 
    #         # batch_i['world_mat'] = torch.tensor(c2w, device=raw_batch['frame_id'].device)[None]
    #         # batch_i  = self.modify_camera(batch_i  ,K = K)
    #         # batch_i =  self.assign_latentcodes(batch_i,True,True)
    #         batch_i['stuff_semantic_grid'] =  torch.tensor(badvoxel[None], device=raw_batch['frame_id'].device)
    #         # batch_i = self.edit_stuff(batch_i, i = -1, raw_stuff=11, operation='higher', distance=20)
    #         # batch_i = self.edit_stuff(batch_i, i = -1, raw_stuff=21, operation='higher', 
    #         # distance=40)
    #         task_batch.append(batch_i)

    #     # raw_batch = self.modify_camera(raw_batch, translate = (0,0,0))
    #     # task_batch.append(raw_batch)
    #     generator.render_obj,generator.render_sky = False , False
    #     out =  self.batchify_render(task_batch= task_batch, generator= generator, valid_keys=['frame_id','rgb','rgb_gt'])
    #     self.save_img(out_dir, out, task_name)

    def render_camera_figure_clevr(self,generator,raw_data, out_dir, fix_scene = True, fix_obj = True, task_name = 'camera_figure_clevr'):
        raw_batch = {k : raw_data[k].clone() for k in  raw_data}
        # raw_batch = self.add_obj(raw_batch, name = 'car')
        step_num = 1
        
        task_batch = []

        for i in range(20):
            raw_batch =  self.assign_latentcodes(raw_batch,True,True)
        task_batch.append(raw_batch)
        # for i in range(3):
        #     raw_batch =  self.assign_latentcodes(raw_batch,True,True)

        raw_batch =  self.assign_latentcodes(raw_batch,True,True)
        # a_batch =  self.assign_latentcodes(raw_batch,False,True)
        # task_batch.append(a_batch)
        task_batch.append(raw_batch)

        raw_batch =  self.rotate_obj(raw_batch,translate=np.array((0,0,2)), index=0)
        raw_batch =  self.rotate_obj(raw_batch ,translate=np.array((-2,0,0)), index=2)
        task_batch.append(raw_batch)


        raw_batch =  self.add_obj(raw_batch, idx=0, operation='delate')
        raw_batch =  self.add_obj(raw_batch, idx=2, operation='delate')
        task_batch.append(raw_batch)  
        raw_batch =  self.add_obj(raw_batch, idx=1, operation='delate')
        task_batch.append(raw_batch)  
        # del_batch =  self.add_obj(raw_batch, idx=2, operation='delate')
        # del_batch =  self.add_obj(del_batch, idx=2, operation='delate')
        # task_batch.append(del_batch)  

        trs = raw_batch['bbox_tr'][0].cpu().numpy()
        new_trs = raw_batch['bbox_tr_fake'][0].cpu().numpy()
        new_trs_1,new_trs_2,new_trs_3,new_trs_4  = np.eye(4),  np.eye(4),  np.eye(4), np.eye(4)
        a = parse_R(trs[1])


        new_trs_1[:3,:3] =  create_R((-1.57,0.2,0),(0.8,0.8,0.8))
        new_trs_1[:3,3] = np.array((2,0.4,2))

        new_trs_2[:3,:3] = create_R((-1.57,0.2,0),(1.5,1.5,1.5))
        new_trs_2[:3,3] = np.array((-1,0.75,0.75))
        new_trs_3[:3,:3] =  create_R((-1.57,0.2,0),(1.2,1.2,1.2))
        new_trs_3[:3,3] = np.array((2,0.6,0.6))

        # new_trs_4[:3,:3] = create_R((-1.57,0.2,0),(1.6,1.6,1.6))
        # new_trs_4[:3,3] = np.array((2,1.6,2))


        add_batch =  self.add_obj(raw_batch, idx=-1, new_tr=new_trs_1, operation='add')
        # task_batch.append(add_batch)  
        add_batch =  self.add_obj(add_batch, idx=-1, new_tr=new_trs_2, operation='add')
        task_batch.append(add_batch)  
        add_batch =  self.add_obj(add_batch, idx=-1, new_tr=new_trs_3, operation='add')
        task_batch.append(add_batch) 
        raw_batch = add_batch
        # add_batch =  self.add_obj(add_batch, idx=-1, new_tr=new_trs_4, operation='add')
        # task_batch.append(add_batch)  

        if self.exp_name == 'giraffe_clevr':
             task_batch += [raw_batch] * (2)
        else:
            # for i in range(step_num):
            #     rate = (i + 1) / step_num
            # low_batch = self.edit_stuff_clevr(raw_batch,operation='lower',height=10)
            # task_batch.append(low_batch)
            high1_batch = self.edit_stuff(raw_batch,operation='higher', raw_stuff=2, i = 0 ,distance=20)
            high1_batch = self.edit_stuff(high1_batch,operation='lower', raw_stuff=2, i = 1 ,distance=-10)
            task_batch.append(high1_batch)
            high2_batch = self.edit_stuff(raw_batch,operation='higher', raw_stuff=2, i = 1 ,distance=20)
            # high2_batch = self.edit_stuff(high2_batch,operation='lower', raw_stuff=2, i = 0 ,distance=-10)
            task_batch.append(high2_batch)

        K = raw_batch['camera_mat'].cpu().numpy()
        K[:,0,0] *= 0.9
        K[:,1,1] *= 0.9
        elevate_batch = self.modify_camera(raw_batch ,K = K, rotate_angle=  np.array((0,0,0)), translate = np.array((0,0,0)))
        task_batch.append(elevate_batch)

        K = raw_batch['camera_mat'].cpu().numpy()
        K[:,0,0] *= 1.8
        K[:,1,1] *= 1.8
        down_batch = self.modify_camera(raw_batch, K = K ,rotate_angle= np.array((10 * 3.14/180,-21 * 3.14/180,-20 * 3.14/180)), translate = np.array((0,-1,3)))
        task_batch.append(down_batch)
        # raw_batch = self.modify_camera(raw_batch ,rotate_angle= np.array((0,0,0)), translate = np.array((0,-2,0)))
        # task_batch.append(raw_batch)

        out =  self.batchify_render(task_batch= task_batch, generator= generator, valid_keys=['frame_id', 'rgb'])
        self.save_img(out_dir, out, task_name)
        return task_batch


    def render_rebuttal(self, generator,raw_data, out_dir, step_num, fix_scene = True, fix_obj = True, task_name = 'teaser'):
        new_trs = [np.array([[ 0.1933, -2.0816,  0.0884, 2.4618],
          [-0.0514, -0.1277, -1.4529, -0.8732],
          [ 4.9960,  0.0792, -0.0184, 1e5],
          [ 0.0000,  0.0000,  0.0000,  1.0000]]) for i in range(5)]

        if raw_data['frame_id'][0] == 10922:
            a = raw_data['bbox_tr'].detach().cpu().numpy()[0]
            new_trs[1] = rightcar = a[0]
            new_trs[1][0,3] =  new_trs[1][0,3] - 5.5
            new_trs[1][:3,:3] = create_R((90 * 3.14 / 180, 180 * 3.14 / 180,0),(3.9,1.5, 1.5))

            new_trs[0][:3,:3] = create_R((90 * 3.14 / 180,90 * 3.14 / 180,0),(4.48,2.0,1.53))
            new_trs[0][:3,3] = np.array((0,-0.92,5))

        elif raw_data['frame_id'][0] == 7131:
            a = raw_data['bbox_tr'].detach().cpu().numpy()[0]
            new_trs[0] = near_car = a[0]
            new_trs[0][0,3] =  new_trs[0][0,3] - 5
            # new_trs[0][2,3] =  new_trs[0][2,3] - 3
            new_trs[1] = far_car = a[1]
            new_trs[1][0,3] =  new_trs[1][0,3] - 5
            new_trs[1][2,3] =  new_trs[1][2,3] + 1

            new_trs[2][:3,:3] = create_R((90 * 3.14 / 180,90 * 3.14 / 180,0),(4,1.5,1.45))
            new_trs[2][:3,3] = np.array((-5,-0.8,5))

            # new_trs[3][:3,:3] = create_R((90 * 3.14 / 180,0 / 180,0),(4,1.5,1.45))
            # new_trs[3][:3,3] = np.array((-3,-1,10))

        elif raw_data['frame_id'][0] == 701263:
            new_trs[0][:3,:3] = create_R((90 * 3.14 / 180,0 * 3.14 / 180,0),(4,1.6, 1.6))
            new_trs[0][:3,3] = np.array((7.5,-0.8,6))

            new_trs[1][:3,:3] = create_R((90 * 3.14 / 180,0 * 3.14 / 180,0),(3.8,1.5, 1.5))
            new_trs[1][:3,3] = np.array((8,-0.8,9))

            new_trs[2][:3,:3] = create_R((90 * 3.14 / 180,-180 * 3.14 / 180,0),(4,1.4, 1.45))
            new_trs[2][:3,3] = np.array((8,-0.8,12))

            new_trs[3][:3,:3] = create_R((90 * 3.14 / 180,-90 * 3.14 / 180,0),(4,1.6, 1.6))
            new_trs[3][:3,3] = np.array((9,-0.8,12))

            new_trs[4][:3,:3] = create_R((90 * 3.14 / 180,-90 * 3.14 / 180,0),(4,1.6, 1.6))
            new_trs[4][:3,3] = np.array((1,-0.8,10))

        elif raw_data['frame_id'][0] == 700084:
            new_trs[0][:3,:3] = create_R((90 * 3.14 / 180,90 * 3.14 / 180,0),(3.3,1.5,1.45))
            new_trs[0][:3,3] = np.array((-2,-1,12))

            new_trs[1][:3,:3] = create_R((90 * 3.14 / 180,-90 * 3.14 / 180,0),(4,1.5,1.45))
            new_trs[1][:3,3] = np.array((1,-1,10))
        elif raw_data['frame_id'][0] == 22605:
            new_trs[0][:3,:3] = create_R((90 * 3.14 / 180,90 * 3.14 / 180,0),(4,1.5,1.45))
            new_trs[0][:3,3] = np.array((-2,-1,15))

#-----------------------------------------------------------------------------
#--------------------------KITTI---------------------------------------------
#-----------------------------------------------------------------------------
        elif raw_data['frame_id'][0] == 22605:
            raw_data['bbox_tr'][0][0][1,3] += 1.4
            raw_data['bbox_tr'][0][1][1,3] += 1.4
            new_trs[0][:3,:3] = create_R((90 * 3.14 / 180,90 * 3.14 / 180,0),(4,1.5,1.45))
            new_trs[0][:3,3] = np.array((-2,0,15))    
        elif raw_data['frame_id'][0] == 20080:
            raw_data['bbox_tr'][0][0][1,3] += 1.4
            raw_data['bbox_tr'][0][1][1,3] += 1.4
            new_trs[0][:3,:3] = create_R((90 * 3.14 / 180,90 * 3.14 / 180,0),(4,1.5,1.45))
            new_trs[0][:3,3] = np.array((-2,0,15))     
        elif raw_data['frame_id'][0] == 10455:
            a = parse_R(raw_data['bbox_tr'][0][0][:3,:3].detach().cpu().numpy())
            raw_data['bbox_tr'][0][0][1,3] += 1.7
            raw_data['bbox_tr'][0][0][0,3] -= 1.4
            raw_data['bbox_tr'][0][0][2,3] += 0
            new_trs[0][:3,:3] = create_R((90 * 3.14 / 180,90 * 3.14 / 180,0),(4.48,2.07,1.59))
            new_trs[0][:3,3] = np.array((-2,0.8,19))
        else:
            # new_trs[0][:3,:3] = create_R((90 * 3.14 / 180,90 * 3.14 / 180,0),(4,1.5,1.45))
            # new_trs[0][:3,3] = np.array((-2,-1.8,15))
            a = parse_R(raw_data['bbox_tr'][0][0][:3,:3].detach().cpu().numpy())
            raw_data['bbox_tr'][0][0][1,3] += 1.6
            raw_data['bbox_tr'][0][0][0,3] -= 0
            raw_data['bbox_tr'][0][0][2,3] -= 6

            new_trs[0][:3,:3] = create_R((90 * 3.14 / 180,0 * 3.14 / 180,0),(4.48,2.07,1.59))
            new_trs[0][:3,3] = np.array((3,0.4,13))
            new_trs[1][:3,:3] = create_R((90 * 3.14 / 180,0 * 3.14 / 180,0),(4.48,2.0,1.53))
            new_trs[1][:3,3] = np.array((2.5,0.4,10))


        raw_batch = {k : raw_data[k].clone() for k in  raw_data}
        # for i in range(10):
        #     raw_batch =  self.assign_latentcodes(raw_batch, fix_scene, fix_obj)

        task_batch = []

        # add_obj_batch = self.add_obj(add_obj_batch, new_tr2)
        task_batch.append(raw_batch)

        for i in range(5):
            raw_batch = self.add_obj(raw_batch, new_trs[i], idx=-1)
            task_batch.append(raw_batch)


        for i in range(20):
            raw_batch =  self.assign_latentcodes(raw_batch, False, False)
            task_batch.append(raw_batch)

        # building2tree_batch = self.render_edit_stuff(task_batch[-1], i = 4, raw_stuff=11, target_stuff=21)

        out =  self.batchify_render(task_batch= task_batch, generator= generator, valid_keys=['camera_intrinsic','camera_pose', 'rgb', 'rgb_gt', 'feat_raw','semantic_gird','feature_grid','bbox_tr']) 
        self.save_img(out_dir, out, task_name)
        self.save_scene(out_dir, out, task_name)

        return task_batch

#?----------------------------------------------------------------------------------
#?-------------------------Video------------------------------------------
#?----------------------------------------------------------------------------------
    def render_move_forward_video(self, generator, raw_data, out_dir, step_num = 4, distance = 20., with_rotate =False, fix_scene = True, fix_obj = True, task_name = 'move_forward_video', for_compare = False):
        '''
        render camera move forward in urban scene
        '''
        raw_batch = {k : raw_data[k].clone() for k in raw_data}

        raw_batch = self.assign_latentcodes(raw_batch, fix_scene, fix_obj)

        step_num = 30 * int(self.fps / 5)
        if not for_compare:
            angle_list = []
            max_degree = 5 * 3.14 / 180
            for i in range(step_num):
                theta = 20 * i * 3.14 / 180
                angle_list += [(np.cos(theta) * max_degree, np.sin(theta) * max_degree, 0)]
        else:
            step_num = 40 
            distance = 20
            angle_list = 40 * [(0,0, 0)]
            max_degree = 5 * 3.14 / 180
        # render_batch = raw_batch
        raw_batch['world_mat']
        task_batch = []
        raw_camera_pose = raw_batch['world_mat']
        batch_size = raw_batch['world_mat'].shape[0]
        for i in range(step_num):
            # camera_pose = 
            translate_i = (0, 0, i * distance / step_num)
            rotate_i = angle_list[i]
            batch_i = self.modify_camera(raw_batch, rotate_angle=rotate_i, translate = translate_i)
            task_batch.append(batch_i)
            

        out =  self.batchify_render(task_batch= task_batch, generator= generator,valid_keys=['rgb', 'depth'])

        self.save_img(out_dir, out, task_name, valid_keys=['rgb', 'depth'])
        # a = task_batch['rays'].cpu().numpy()
        # b = task_batch['world_mat'].cpu().numpy()
        return task_batch

    def render_elevate_camera_video(self,generator, raw_data, out_dir, step_num = 20, elevate_height  = 1.5, distance = 32., fix_scene = True, fix_obj = True, task_name = 'elevate_camera_video'):
        raw_batch = {k : raw_data[k].clone() for k in  raw_data}
        # raw_batch = self.add_obj(raw_batch, name = 'car')
        raw_batch =  self.assign_latentcodes(raw_batch, fix_scene, fix_obj)

        task_batch = []
        step_num = 30 * int(self.fps / 5)
        step_num_mini = 5 * int(self.fps / 5)
        for i in range(step_num_mini):
            rate = (i+1) / step_num_mini
            translate_i = (0, rate * -elevate_height,0)
            rotate_i =  (rate * -15 * 3.14 / 180,0,0)
            batch_i = self.modify_camera(raw_batch, rotate_angle=rotate_i, translate = translate_i)
            task_batch.append(batch_i)
        raw_batch = {k : batch_i[k].clone() for k in  batch_i}
        zoom_in_times = 0.92
        for i in range(step_num_mini):
            rate = (i+1) / step_num_mini
            # translate_i = (0, i * -elevate_height / step_num,0)
            K = batch_i['camera_mat'].cpu().numpy()
            K[:,0,0] *= 1 +  rate * (zoom_in_times - 1)
            K[:,1,1] *= 1 +  rate * (zoom_in_times - 1)
            batch_i = self.modify_camera(raw_batch, K = K)
            task_batch.append(batch_i)
        raw_batch =  {k : batch_i[k].clone() for k in  batch_i}
        for i in range(step_num):
            rate = (i+1) / step_num
            translate_i = (0,0,rate * distance)
            # rotate_i = (0,0,0)   
            batch_i = self.modify_camera(raw_batch,  translate = translate_i)
            task_batch.append(batch_i)

        out =  self.batchify_render(task_batch= task_batch, generator= generator)
        self.save_img(out_dir, out, task_name)

    def render_interpolate_camera_video(self,generator, raw_data, out_dir, fix_scene = True, fix_obj = True, task_name = 'interpolate_camera_video'):
        raw_batch = {k : raw_data[k].clone() for k in  raw_data}
        # raw_batch = self.add_obj(raw_batch, name = 'car')
        raw_batch =  self.assign_latentcodes(raw_batch, fix_scene, fix_obj)


        target_pose_list = [[(0,0,0),(0,0,0)]]
        target_pose_list += [[(0,-3,10), (-15,0,0)]]
        target_pose_list += [[(0,-1,20), (0,0,0)]]
        target_pose_list += [[(0,-1,20), (0,-15,-15)]]
        target_pose_list += [[(0,-1,30), (0,15,15)]]
        target_pose_list += [[(0,-4,10), (-15,0,0)]]
        target_pose_list += [[(0,0,0), (0,0,0)]]

        target_pose_list = np.array(target_pose_list)
        elevate_height  = 4
        task_batch = []
        step_num = 5 * int(self.fps / 5)
        for i in range(target_pose_list.shape[0] - 1):
            for j in range(step_num):
                theta = (j + 1) * (3.14 /2) / step_num
                rotate_j = (np.cos(theta) * target_pose_list[i,1] + np.sin(theta) * target_pose_list[i+1,1]) * 3.14 / 180
                translate_j = np.cos(theta) *target_pose_list[i,0] + np.sin(theta) * target_pose_list[i+1,0]
                batch_j = self.modify_camera(raw_batch, rotate_angle=rotate_j, translate = translate_j)
                task_batch.append(batch_j)


        out =  self.batchify_render(task_batch= task_batch, generator= generator)
        self.save_img(out_dir, out, task_name)

    def render_building2tree_video(self,generator, raw_data, out_dir, fix_scene = True, fix_obj = True, task_name = 'building2tree_video'):
        '''
        render stand still urban scene
        '''
        raw_batch = {k : raw_data[k].clone() for k in  raw_data}
        raw_batch =  self.assign_latentcodes(raw_batch, fix_scene, fix_obj)
 
        task_batch = []
        elevate_height = 1.5
        task_batch+= [raw_batch] * 3 * int(self.fps / 5)
        for i in range(10):
            raw_batch = self.edit_stuff(raw_batch, i = i, raw_stuff=name2label['building'].id, target_stuff=21)
            task_batch+= [raw_batch] * 3 * int(self.fps / 5)
            
        raw_batch = self.edit_stuff(raw_batch, i = -1, raw_stuff=name2label['building'].id, target_stuff=21)
        task_batch+= [raw_batch] * 9 * int(self.fps / 5)
        out =  self.batchify_render(task_batch= task_batch, generator= generator)

        self.save_img(out_dir, out, task_name)

        return task_batch

    def render_road2grass_video(self,generator, raw_data, out_dir, fix_scene = True, fix_obj = True, task_name = 'building2tree_video'):
        '''
        render stand still urban scene
        '''
        task_batch = []
        raw_batch = {k : raw_data[k].clone() for k in  raw_data}
        raw_batch =  self.assign_latentcodes(raw_batch, fix_scene, fix_obj)
    
        # Road only
        task_batch+= [raw_batch] * int(self.fps / 5)
        # Sidewalk and others
        for i in range(20):
            raw_batch = self.edit_stuff(raw_batch, i = -4, raw_stuff=name2label['road'].id, target_stuff=name2label['terrain'].id, z_threshold=20 - i)
            raw_batch = self.edit_stuff(raw_batch, i = -4, raw_stuff=name2label['sidewalk'].id, target_stuff=name2label['terrain'].id, z_threshold=20 - i)
            raw_batch = self.edit_stuff(raw_batch, i = -4, raw_stuff=name2label['fence'].id, target_stuff=name2label['terrain'].id, z_threshold=20 - i)
            raw_batch = self.edit_stuff(raw_batch, i = -4, raw_stuff=name2label['guard rail'].id, target_stuff=name2label['terrain'].id, z_threshold=20 - i)
            task_batch+= [raw_batch] * int(self.fps / 5)

        task_batch+= [raw_batch] * 4 * int(self.fps / 5)
        # for i, scene in enumerate(task_batch):
        #     self.save_scene(scene, '%d'%i)
        out =  self.batchify_render(task_batch= task_batch, generator= generator) 
        self.save_img(out_dir, out, task_name)

        return task_batch

    def render_move_tree_video(self,generator, raw_data, out_dir, step_num, fix_scene = True, fix_obj = True, task_name = 'stuff_higher'):
        '''
        for frame 8062 only
        render stand still urban scene
        '''
        raw_batch = {k : raw_data[k].clone() for k in  raw_data}
        raw_batch =  self.assign_latentcodes(raw_batch, fix_scene, fix_obj)

        stuff_id = []
        batch_size = raw_batch['frame_id'].shape[0]
 
        task_batch = []
        elevate_height = 1.5
        translate_i = (0,-elevate_height,0) 
        raw_batch = self.modify_camera(raw_batch, translate = translate_i)
        task_batch.append(raw_batch)
        # for i in range(3):
        for i in range(5):
            raw_batch, tree_idx = self.edit_stuff(raw_batch, i = -2, raw_stuff = 21,operation='x_move', distance=-(i+1), export_idx=True)
            task_batch.append(raw_batch)
        # for i in range(3):
        for i in range(20):
            raw_batch = self.edit_stuff(raw_batch, i = -3, raw_stuff=21, operation='z_move', distance=-(i+1), external_idx=tree_idx)
            task_batch.append(raw_batch)

        out =  self.batchify_render(task_batch= task_batch, generator= generator)

        self.save_img(out_dir, out, task_name)

        return task_batch
    
    def render_building_lower_video(self,generator, raw_data, out_dir, step_num, fix_scene = True, fix_obj = True, task_name = 'stuff_higher'):
        '''
        render stand still urban scene
        '''
        raw_batch = {k : raw_data[k].clone() for k in  raw_data}
        raw_batch =  self.assign_latentcodes(raw_batch, fix_scene, fix_obj)

        stuff_id = []
        batch_size = raw_batch['frame_id'].shape[0]
 
        task_batch = []
        elevate_height = 1.5
        task_batch.append(raw_batch)
        for i in range(20):
            raw_batch = self.edit_stuff(raw_batch, i = -1, raw_stuff = 11,operation='lower', distance=1)
            task_batch += [raw_batch] * int(self.fps / 5)

        task_batch += [raw_batch] * 6 * int(self.fps / 5)
        out =  self.batchify_render(task_batch= task_batch, generator= generator)

        self.save_img(out_dir, out, task_name)

        return task_batch

    def render_object_editing_video(self,generator, raw_data, out_dir, step_num, fix_scene = True, fix_obj = True, task_name = 'stuff_higher'):
        '''
        render stand still urban scene
        '''
        new_tr1, new_tr2, new_tr3 = np.eye(4), np.eye(4), np.eye(4)
        new_tr1[:3,:3] = create_R((90 * 3.14 / 180,90 * 3.14 / 180,0),(4.5,2,1.85))
        new_tr1[:3,3] = np.array((2,-0.8,8))
        new_tr2[:3,:3] = create_R((90 * 3.14 / 180,90 * 3.14 / 180,0),(5,2,1.45))
        new_tr2[:3,3] = np.array((-2,-0.8,8))
        new_tr3[:3,:3] = create_R((90 * 3.14 / 180,90 * 3.14 / 180,0),(4.3,2,1.85))
        new_tr3[:3,3] = np.array((0,-0.8,12))

        raw_batch = {k : raw_data[k].clone() for k in  raw_data}
        # raw_batch =  self.assign_latentcodes(raw_batch, True,False)

        task_batch = []

        raw_batch =  self.assign_latentcodes(raw_batch, True,True)

        task_batch += [raw_batch] * 3 * int(self.fps / 5)
        raw_batch  = self.add_obj(raw_batch, operation= 'delate', idx=0)
        task_batch += [raw_batch] * 3 * int(self.fps / 5)
        raw_batch  = self.add_obj(raw_batch, operation= 'delate', idx=1)
        task_batch += [raw_batch] * 3 * int(self.fps / 5)
        raw_batch  = self.add_obj(raw_batch, operation= 'delate', idx=2)
        task_batch += [raw_batch] * 3 * int(self.fps / 5)
 
        raw_batch  = self.add_obj(raw_batch, new_tr1, idx=0)
        task_batch += [raw_batch] * 4 * int(self.fps / 5)
        raw_batch = self.add_obj(raw_batch, new_tr2, idx=1)
        task_batch += [raw_batch] * 4 * int(self.fps / 5)
        raw_batch = self.add_obj(raw_batch, new_tr3, idx=2)
        task_batch += [raw_batch] * 4 * int(self.fps / 5)
        raw_batch = self.add_obj(raw_batch, operation= 'delate', idx=2)
        task_batch += [raw_batch] * 4 * int(self.fps / 5)


        # del raw_batch['z_bbox'],raw_batch['z_global']
        for i in range(3):
            restyle_batch = self.assign_latentcodes(raw_batch, True,True)
            task_batch += [restyle_batch] * 4 * int(self.fps / 5)
        raw_batch = task_batch[-1]

        # del raw_batch['z_bbox']
        for i in range(3):
            # del raw_batch['z_bbox']
            restyle_batch =  self.assign_latentcodes(raw_batch, False,True)        
            task_batch += [restyle_batch] * 4 * int(self.fps / 5)
        raw_batch = task_batch[-1]

        step_num = 10 * int(self.fps / 5)
        for i in range(step_num //2):
            rate = (i + 1) / (step_num //2)
            batch_i = self.rotate_obj(raw_batch, rotate_angle=(0,rate * 25 * 3.14 / 180,0), index=-1)
            task_batch.append(batch_i)
        raw_batch = task_batch[-1]

        for i in range(step_num //2):
            rate = (i + 1) / (step_num // 2)
            batch_i = self.rotate_obj(raw_batch, rotate_angle=(0,-rate * 25 * 3.14 / 180,0), index=-1)
            task_batch.append(batch_i)
        raw_batch = task_batch[-1]

        for i in range(step_num // 2):
            rate = (i + 1) / (step_num //2)
            batch_i = self.rotate_obj(raw_batch,rotate_angle=(0,rate * -25 * 3.14 / 180,0),  index=0)
            batch_i = self.rotate_obj(batch_i,rotate_angle=(0,rate * 25 * 3.14 / 180,0),  index=1)
            task_batch.append(batch_i) 
        raw_batch = task_batch[-1]
        
        for i in range(step_num //2):
            rate = (i + 1) /(step_num //2)
            batch_i = self.rotate_obj(raw_batch,rotate_angle=(0,rate * 25 * 3.14 / 180,0),  index=0)
            batch_i = self.rotate_obj(batch_i,rotate_angle=(0,rate * -25 * 3.14 / 180,0),  index=1)
            task_batch.append(batch_i) 
        raw_batch = task_batch[-1]

        for i in range(step_num):
            rate = (i + 1) / step_num
            batch_i = self.rotate_obj(raw_batch, rotate_angle=(0,0,0),translate=(0,0,rate * -3), index=1)
            task_batch.append(batch_i) 
        raw_batch = task_batch[-1]

        for i in range(step_num):
            rate = (i + 1) / step_num
            batch_i = self.rotate_obj(raw_batch,rotate_angle=(0,0,0), translate=(0,0,rate * 5), index=-1)
            task_batch.append(batch_i) 
        raw_batch = task_batch[-1]

        out =  self.batchify_render(task_batch= task_batch, generator= generator) 

        self.save_img(out_dir, out, task_name)
        # self.save_scene(out_dir, out, task_name)

        return task_batch

    def render_clevr_video(self,generator, raw_data, out_dir, step_num, fix_scene = True, fix_obj = True, task_name = 'clevr_video'):
        raw_batch = {k : raw_data[k].clone() for k in  raw_data}
        task_batch = []

        for i in range(5):
            raw_batch =  self.assign_latentcodes(raw_batch,True,True)
            task_batch += [raw_batch] * 2 * int(self.fps / 5)
        # del raw_batch['z_bbox'], raw_batch['z_global']
        for i in range(5):
            raw_batch =  self.assign_latentcodes(raw_batch,False,True)
            task_batch += [raw_batch] * 2 * int(self.fps / 5)

        # raw_batch =  self.assign_latentcodes(raw_batch,,True)
        task_batch.append(raw_batch)

        step_num = 10 * int(self.fps / 5)
        for i in range(step_num):
            rate = (i + 1) / step_num
            rotate_batch =  self.rotate_obj(raw_batch,rotate_angle=np.array((0,3.14,0)) * rate, index=-1)
            task_batch.append(rotate_batch)
        raw_batch = rotate_batch

        for i in range(step_num):
            rate = (i + 1) / step_num
            translate_batch =  self.rotate_obj(raw_batch,translate=np.array((0,0,2)) * rate, index=0)
            translate_batch =  self.rotate_obj(raw_batch ,translate=np.array((-2,0,0)) * rate, index=2)
            task_batch.append(translate_batch)
        raw_batch = translate_batch


        delate_batch =  self.add_obj(raw_batch, idx=0, operation='delate')
        task_batch += [delate_batch] * int(self.fps / 5) * 2
        delate_batch =  self.add_obj(delate_batch, idx=2, operation='delate')
        task_batch += [delate_batch] * int(self.fps / 5) * 2
        delate_batch =  self.add_obj(delate_batch, idx=1, operation='delate')
        # task_batch += [delate_batch] * int(self.fps / 5) * 2
        raw_batch = delate_batch  

        # trs = raw_batch['bbox_tr'][0].cpu().numpy()
        # new_trs = raw_batch['bbox_tr_fake'][0].cpu().numpy()
        new_trs_1,new_trs_2,new_trs_3, new_trs_4  = np.eye(4),  np.eye(4),  np.eye(4), np.eye(4)
        new_trs_1[:3,:3] =  create_R((-1.57,0.2,0),(0.8,0.8,0.8))
        new_trs_1[:3,3] = np.array((2,0.4,2))
        new_trs_2[:3,:3] = create_R((-1.57,0.2,0),(1.5,1.5,1.5))
        new_trs_2[:3,3] = np.array((-1,0.75,0.75))
        new_trs_3[:3,:3] =  create_R((-1.57,0.2,0),(1.2,1.2,1.2))
        new_trs_3[:3,3] = np.array((2,0.6,0.6))


        add_batch =  self.add_obj(raw_batch, idx=-1, new_tr=new_trs_1, operation='add')
        task_batch += [add_batch] * int(self.fps / 5) * 2
        add_batch =  self.add_obj(add_batch, idx=-1, new_tr=new_trs_2, operation='add')
        task_batch += [add_batch] * int(self.fps / 5) * 2
        add_batch =  self.add_obj(add_batch, idx=-1, new_tr=new_trs_3, operation='add')
        task_batch += [add_batch] * int(self.fps / 5) * 2
        raw_batch = add_batch
 

        if self.exp_name == 'giraffe_clevr':
             task_batch += [raw_batch] * step_num
        else:
            for i in range(step_num):
                rate = (i + 1) / step_num
                high1_batch = self.edit_stuff(raw_batch,operation='higher', raw_stuff=2, i = 0 ,distance=int(20 * rate))
                high1_batch = self.edit_stuff(high1_batch,operation='higher', raw_stuff=2, i = 1 ,distance=int(30 * rate))
                task_batch.append(high1_batch)
            raw_batch = high1_batch

        elevate_batch = {k : raw_batch[k].clone() for k in  raw_batch}
        step_num_nimi = int(step_num / 4)
        for i in range(step_num_nimi):
            rate = (i + 1) / step_num_nimi
            K = raw_batch['camera_mat'].cpu().numpy()
            K[:,0,0] *= (0.9 * rate + 1.0 * (1-rate))
            K[:,1,1] *= (0.9 * rate + 1.0 * (1-rate))
            elevate_batch = self.modify_camera(elevate_batch ,K = K * rate, rotate_angle=  np.array((0,0,0)), translate = np.array((0,0,0)))
            task_batch.append(elevate_batch)
        raw_batch = elevate_batch

        step_num_large = int(step_num * 3)
        for i in range(step_num_large):
            rate = (i + 1) / step_num_large
            K = raw_batch['camera_mat'].cpu().numpy()
            K[:,0,0] *= (1.4 * rate + 0.9 * (1-rate))
            K[:,1,1] *= (1.4 * rate + 0.9 * (1-rate))
            translate = np.array((0,-1,3)) * rate
            rotate = np.array((10 * 3.14/180,-21 * 3.14/180,-20 * 3.14/180)) * rate
            down_batch = self.modify_camera(raw_batch, K = K ,rotate_angle= rotate, translate = translate)
            task_batch.append(down_batch)
        # raw_batch = self.modify_camera(raw_batch ,rotate_angle= np.array((0,0,0)), translate = np.array((0,-2,0)))
        # task_batch.append(raw_batch)

        out =  self.batchify_render(task_batch= task_batch, generator= generator, valid_keys=['frame_id', 'rgb'])
        self.save_img(out_dir, out, task_name)
        return task_batch

    def render_teaser_video1(self, generator,raw_data, out_dir, step_num, fix_scene = True, fix_obj = True, task_name = 'teaser'):


        task_batch = []
        raw_batch = {k : raw_data[k].clone() for k in  raw_data}
        raw_batch =  self.assign_latentcodes(raw_batch, fix_scene, fix_obj)


        task_batch = []


        step_num = 30 * int(self.fps / 5)
        distance = 20
        angle_list = []
        max_degree = 5 * 3.14 / 180
        for i in range(step_num):
            theta = (20 * 2/ int(self.fps / 5)) * i * 3.14 / 180
            angle_list += [(np.cos(theta) * max_degree, np.sin(theta) * max_degree, 0)]
        # render_batch = raw_batch
        raw_batch['world_mat']
        task_batch = []
        raw_camera_pose = raw_batch['world_mat']
        batch_size = raw_batch['world_mat'].shape[0]
        for i in range(step_num):
            # camera_pose = 
            translate_i = (0, 0, i * distance / step_num)
            rotate_i = angle_list[i]
            batch_i = self.modify_camera(raw_batch, rotate_angle=rotate_i, translate = translate_i)
            task_batch.append(batch_i)
        raw_batch = batch_i

        step_num_mini = 10 * int(self.fps / 5)
        elevate_height = 2.5
        zoom_in_times = 0.92
        for i in range(step_num_mini):
            rate = (i+1) / step_num_mini
            # translate_i = (0, i * -elevate_height / step_num,0)
            K = raw_batch['camera_mat'].cpu().numpy()
            # K[:,0,0] *= 1 +  rate * (zoom_in_times - 1)
            # K[:,1,1] *= 1 +  rate * (zoom_in_times - 1)
            translate_i = (0, rate * -elevate_height,0)
            rotate_i =  (rate * -20 * 3.14 / 180,0,0)
            batch_i = self.modify_camera(raw_batch, K = K, rotate_angle=rotate_i, translate = translate_i)
            task_batch.append(batch_i)
        raw_batch = batch_i

        for i in range(step_num // 2):
            rate = (i+1) / (step_num // 2)
            translate_i = (0,0,rate * -distance)
            # rotate_i = (0,0,0)   
            batch_i = self.modify_camera(raw_batch,  translate = translate_i)
            task_batch.append(batch_i)
        raw_batch = batch_i
            
        for i in range(step_num_mini):
            rate = (i+1) / step_num_mini
            # translate_i = (0, i * -elevate_height / step_num,0)
            K = raw_batch['camera_mat'].cpu().numpy()
            # K[:,0,0] *= (1 / (1 +  rate * (zoom_in_times - 1)))
            # K[:,1,1] *= (1 / (1 +  rate * (zoom_in_times - 1)))
            translate_i = (0, rate * elevate_height,0)
            rotate_i =  (rate * 20* 3.14 / 180,0,0)
            batch_i = self.modify_camera(raw_batch, K = K, rotate_angle=rotate_i, translate = translate_i)
            task_batch.append(batch_i)
        raw_batch =  batch_i



        target_pose_list = [[(0,0,0),(0,0,0)]]
        target_pose_list += [[(0,-3,10), (-15,0,0)]]
        target_pose_list += [[(0,-1,20), (0,0,0)]]
        target_pose_list += [[(0,-1,20), (0,-15,-15)]]
        target_pose_list += [[(0,-1,30), (0,15,15)]]
        target_pose_list += [[(0,-4,10), (-15,0,0)]]
        target_pose_list += [[(0,0,0), (0,0,0)]]
        target_pose_list = np.array(target_pose_list)
        elevate_height  = 4
        step_num = 5 * int(self.fps / 5)
        for i in range(target_pose_list.shape[0] - 1):
            for j in range(step_num):
                theta = (j + 1) * (3.14 /2) / step_num
                rotate_j = (np.cos(theta) * target_pose_list[i,1] + np.sin(theta) * target_pose_list[i+1,1]) * 3.14 / 180
                translate_j = np.cos(theta) *target_pose_list[i,0] + np.sin(theta) * target_pose_list[i+1,0]
                batch_j = self.modify_camera(raw_batch, rotate_angle=rotate_j, translate = translate_j)
                task_batch.append(batch_j)
  


        out =  self.batchify_render(task_batch= task_batch, generator= generator, valid_keys=['camera_intrinsic','camera_pose', 'rgb', 'rgb_gt', 'semantic_gird','bbox_tr','bbox_semantic']) 
        self.save_img(out_dir, out, task_name)
        self.save_scene(out_dir, out, task_name, ['frame_id','camera_intrinsic','camera_pose','semantic_gird','bbox_tr','bbox_semantic'])

        return task_batch



    def render_teaser_video2(self, generator,raw_data, out_dir, step_num, fix_scene = True, fix_obj = True, task_name = 'teaser_video2'):


        new_tr1 = np.array([[ 0.1933, -2.0816,  0.0884, 2.4618],
          [-0.0514, -0.1277, -1.4529, -0.8732],
          [ 4.9960,  0.0792, -0.0184, 17.7895],
          [ 0.0000,  0.0000,  0.0000,  1.0000]])
        new_tr2 = np.array([[ 0.1933, -2.0816,  0.0884, -2.4618],
          [-0.0514, -0.1277, -1.4529, -0.8732],
          [ 4.9960,  0.0792, -0.0184, 17.7895],
          [ 0.0000,  0.0000,  0.0000,  1.0000]])
        new_tr3 = np.array([[ 0.1933, -2.0816,  0.0884, 0.4618],
                            

          [-0.0514, -0.1277, -1.4529, -0.8732],
          [ 4.9960,  0.0792, -0.0184, 20.7895],
          [ 0.0000,  0.0000,  0.0000,  1.0000]])

        raw_batch = {k : raw_data[k].clone() for k in  raw_data}
        for i in range(20):
            raw_batch =  self.assign_latentcodes(raw_batch, fix_scene, fix_obj)

        task_batch = []


        # add_obj_batch = self.add_obj(add_obj_batch, new_tr2)
        task_batch.append(raw_batch)



        for i in range(10):
            raw_batch = self.modify_camera(raw_batch, rotate_angle=(0,0,0), translate = (0,0,7 / 10))
            task_batch.append(raw_batch)


        raw_batch = self.add_obj(raw_batch, new_tr1, idx=-1)
        task_batch.append(raw_batch)
        raw_batch = self.add_obj(raw_batch, new_tr2, idx = -1)
        task_batch.append(raw_batch)
        raw_batch = self.add_obj(raw_batch, new_tr3, idx = -1)
        task_batch.append(raw_batch)
        raw_batch = self.add_obj(raw_batch, idx = 0, operation='delate')
        task_batch.append(raw_batch)

        # for i, scene in enumerate(task_batch):
        #     self.save_scene(scene, '%d'%i)
        # raw_batch = self.modify_camera(raw_batch, rotate_angle=(20 * 3.14 / 180,0,0), translate = (0,2.5,2))
        tree_batch = {k : raw_batch[k].clone() for k in  raw_batch}
        for i in range(10):
             tree_batch = self.edit_stuff(tree_batch, i = i, raw_stuff=11, target_stuff=21)
             task_batch.append(tree_batch)
        tree_batch = self.edit_stuff(tree_batch, i = -1, raw_stuff=11, target_stuff=21)
        task_batch.append(tree_batch)
        # building2tree_batch = self.render_edit_stuff(task_batch[-1], i = 4, raw_stuff=11, target_stuff=21)

        grass_batch = {k : raw_batch[k].clone() for k in  raw_batch}
        for i in range(20):
            grass_batch = self.edit_stuff(grass_batch, i = -4, raw_stuff=name2label['road'].id, target_stuff=name2label['terrain'].id, z_threshold=20 - i)
            grass_batch = self.edit_stuff(grass_batch, i = -4, raw_stuff=name2label['sidewalk'].id, target_stuff=name2label['terrain'].id, z_threshold=20 - i)
            grass_batch = self.edit_stuff(grass_batch, i = -4, raw_stuff=name2label['fence'].id, target_stuff=name2label['terrain'].id, z_threshold=20 - i)
            grass_batch = self.edit_stuff(grass_batch, i = -4, raw_stuff=name2label['guard rail'].id, target_stuff=name2label['terrain'].id, z_threshold=20 - i)
            task_batch+= [grass_batch] 

        low_batch = {k : raw_batch[k].clone() for k in  raw_batch}
        for i in range(20):
            low_batch = self.edit_stuff(low_batch, i = -1, raw_stuff = 11,operation='lower', distance=1)
            task_batch += [low_batch]


        task_batch.append(raw_batch) 

        out =  self.batchify_render(task_batch= task_batch, generator= generator, valid_keys=['camera_intrinsic','camera_pose', 'rgb', 'rgb_gt', 'semantic_gird','bbox_tr']) 
        self.save_img(out_dir, out, task_name)
        self.save_scene(out_dir, out, task_name, ['frame_id','camera_intrinsic','camera_pose','semantic_gird','bbox_tr'])

        return task_batch



#?----------------------------------------------------------------------------------
#?-------------------------KITTI-360 Base------------------------------------------
#?----------------------------------------------------------------------------------
    def add_obj(self, raw_batch, new_tr = None,idx = -1, operation = 'add', keep_layout = True):
        assert 'bbox_tr' in raw_batch.keys()

        batch_size = raw_batch['bbox_tr'].shape[0]
        trs = raw_batch['bbox_tr'].detach().cpu().numpy()
        semantics = raw_batch['bbox_semantic'].detach().cpu().numpy()

        new_bbox_trs, new_bbox_semantics = [], []
        batch = {k : raw_batch[k].clone() for k in  raw_batch}
        for b in range(batch_size):
            bbx_tr = trs[b].copy()
            bbox_samentic = semantics[b].copy()
            if idx == -1:
                idx = np.argwhere(bbox_samentic == -1)
                if idx.shape[0] == 0:
                    idx = bbox_samentic.shape[0] - 1
                else:
                    idx = idx[0]
            if operation == 'add':
                bbx_tr[idx] = new_tr
                bbox_samentic[idx] = 26
            elif operation == 'delate':
                bbx_tr[idx] = np.eye(4)
                bbox_samentic[idx] = -1            
            
            new_bbox_trs.append(bbx_tr)
            new_bbox_semantics.append(bbox_samentic)

        new_bbox_trs, new_bbox_semantics = np.stack(new_bbox_trs), np.stack(new_bbox_semantics)
        
        batch['bbox_tr'] = torch.tensor(new_bbox_trs, device=raw_batch['bbox_tr'].device)
        batch['bbox_semantic'] =  torch.tensor(new_bbox_semantics, device=raw_batch['bbox_tr'].device)
        return batch
   
    def rotate_obj(self, raw_batch ,rotate_angle = (0,0,0), translate = (0,0,0) ,index = 0):
        '''
        '''
        assert 'bbox_tr' in raw_batch.keys()
        batch_size, bbox_num = raw_batch['bbox_tr'].shape[0],raw_batch['bbox_tr'].shape[1]
        batch = {k : raw_batch[k].clone() for k in  raw_batch}
        if index == -1:
            bbox_idx = range(bbox_num)
        else:
            bbox_idx = [index]
        for i in bbox_idx:
            for b in range(batch_size):
                old_tr = batch['bbox_tr'][b,i].detach().cpu().numpy()
                if False:
                    scale, _,old_eular_angle = parse_R(old_tr[:3,:3], True)
                    R_mat =  create_R(old_eular_angle + rotate_angle,scale)
                else:
                    # create_R(rotate_angle)
                    R_mat =  create_R(rotate_angle) @  old_tr[:3,:3]
                new_tr = old_tr.copy()
                new_tr[:3,:3] =  R_mat
                new_tr[:3,3] = old_tr[:3,3] + np.array(translate)
                batch['bbox_tr'][b,i] = torch.tensor(new_tr, device=raw_batch['bbox_tr'].device)
 
        return batch

    def modify_camera(self, raw_batch ,rotate_angle = (0,0,0), translate = (0,0,0), K = np.array(()), c2w = None  ,index = 0):
        # assert 'bbox_tr' in raw_batch.keys()
        # batch = copy.deepcopy(raw_batch)
        # batch = raw_batch.copy()
        batch = {k : raw_batch[k].clone() for k in raw_batch}
        
        batch_size = raw_batch['world_mat'].shape[0]
        # batch['rays'] = torch.zeros((batch_size, self.H_ray, self.W_ray,9), dtype=torch.float32, device= batch['world_mat'].device)
        for b in range(batch_size):
            c2w = batch['world_mat'][b].cpu().numpy()
            if K.size == 0:
                k = batch['camera_mat'][b].cpu().numpy()
            else:
                k = K[b]
            # new_c2w = c2w.copy()
            if False:
                _, _,old_eular_angle = parse_R(old_c2w, True)
                R_mat =  create_R(old_eular_angle + rotate_angle)
            else:
                c2w[:3,:3] =  create_R(rotate_angle) @ c2w[:3,:3]
            # c2w[:3,:3] = R_mat
            c2w[:3, 3] += translate
            rays_new =build_rays(H = self.H_ray, W = self.W_ray, c2w=c2w, K=k).reshape(self.H_ray, self.W_ray,9)
            batch['world_mat'][b] = torch.tensor(c2w)
            batch['camera_mat'][b] = torch.tensor(k)
            batch['rays'][b] = torch.tensor(rays_new)
        return batch

    def edit_stuff(self,raw_data, raw_stuff = 11, target_stuff = 21, operation = 'change', distance = 10, i = -1, fix_obj = True, task_name = 'edit_stuff', external_idx = [], export_idx = False, z_threshold = 63):
        '''
        render stand still urban scene
        i = -1 : choose biggest tree/building
        i = -2 : choose biggest tree/building
        i = -3 : by external input
        i = -4 : by z value
        '''
        raw_batch = {k : raw_data[k].clone() for k in  raw_data}
        # raw_batch =  self.assign_latentcodes(raw_batch, fix_scene, fix_obj)
        if i == -3:
            assert len(external_idx) > 0
        if operation != 'change':
            target_stuff = raw_stuff 

        stuff_id = []
        batch_size = raw_batch['frame_id'].shape[0]
        
        for b in range(batch_size):
            frame_ids = raw_batch['frame_id'][b] % (1000 * (1 + 99 * self.is_kitti))
            seq_num = raw_batch['frame_id'][b] // (1000 * (1 + 99 * self.is_kitti))
            if (i > -1) or  (i == -2):
                stuff_id.append(self.load_stuff_instanceId(seq_num = seq_num, frame_ids= frame_ids))
            else:
                stuff_id.append(raw_batch['stuff_semantic_grid'][b].cpu().numpy() * 10000) 


        valid_idx_list = []
        for b in range(batch_size):
            idx_mask = (stuff_id[b] - raw_stuff * 10000 < 10000) * (stuff_id[b] - raw_stuff * 10000 >= 0)
            id_list = np.unique(stuff_id[b][idx_mask]) 

            if i == -1:   
                # All instance
                idx_mask = idx_mask
            elif i == -2:
                # Biggest instance
                a = [np.sum(stuff_id[b] == j) for j in id_list ]
                biggest_idx = np.argmax(a)
                idx_mask = (stuff_id[b] == id_list[biggest_idx])
            elif i == -3:
                # Eternel index
                idx_mask = idx_mask * 0
                idx_mask[external_idx[b][:,0],external_idx[b][:,1],external_idx[b][:,2]] = 1
            elif i == -4:
                # Based on z value
                idx_mask[:,:,:z_threshold] = 0
            elif i < len(id_list):
                # Given instance id
                idx_mask = (stuff_id[b] == id_list[i])
            else:
                continue
            valid_idx = np.argwhere(idx_mask == 1)

            raw_batch['stuff_semantic_grid'][b][idx_mask == 1] = 0
            print(np.sum(idx_mask))
            idx_mask[:,:,:] = 0
            if operation == 'higher':
                for i in range(int(distance)):
                    valid_idx[:,0] = valid_idx[:,0] +( -1 +  2 * (1 - self.is_kitti))
                    valid_idx = np.clip(valid_idx,0,63)   
                    idx_mask[valid_idx[:,0],valid_idx[:,1],valid_idx[:,2]] = 1
                print('a')
                print(np.sum(idx_mask))

            elif operation == 'lower':
                valid_idx[:,0] = valid_idx[:,0] + int(distance)
                valid_idx = np.clip(valid_idx,0,63)   
                print('b')
            elif operation == 'x_move':
                valid_idx[:,1] = valid_idx[:,1] + int(distance)
                valid_idx = np.clip(valid_idx,0,63)   
                print('b')
            elif operation == 'z_move':
                valid_idx[:,2] = valid_idx[:,2] + int(distance)
                valid_idx = np.clip(valid_idx,0,63)   
                print('b')
            elif operation == 'change':
                pass 
            else:
                raise ValueError
            idx_mask[valid_idx[:,0],valid_idx[:,1],valid_idx[:,2]] = 1
            raw_batch['stuff_semantic_grid'][b][idx_mask == 1] = target_stuff
            valid_idx_list += [valid_idx]

        if export_idx:
            return raw_batch, valid_idx_list
        else:
            return raw_batch

    def render_building_to_tree(self,generator, raw_data, out_dir, step_num, fix_scene = True, fix_obj = True, task_name = 'building_to_tree'):
        '''
        render stand still urban scene
        '''
        raw_batch = {k : raw_data[k].clone() for k in  raw_data}
        raw_batch =  self.assign_latentcodes(raw_batch, fix_scene, fix_obj)

        stuff_id = []
        batch_size = raw_batch['frame_id'].shape[0]
 
        task_batch = []
        # elevate_height = 1.5
        # # for i in range(5):
        # translate_i = (0,-2.5,0) 
        # raw_batch = self.modify_camera(raw_batch, translate = translate_i)
        task_batch.append(raw_batch)
 

        for i in range(4):
            raw_batch = self.edit_stuff(raw_batch, i = i, raw_stuff=name2label['building'].id, target_stuff=21)
            # if i % 2 == 0:
        task_batch.append(raw_batch)

        raw_batch = self.edit_stuff(raw_batch, i = -1, raw_stuff=name2label['building'].id, target_stuff=21)
        task_batch.append(raw_batch)
        out =  self.batchify_render(task_batch= task_batch, generator= generator)

        self.save_img(out_dir, out, task_name)

        return task_batch
    
    def render_building_lower(self,generator, raw_data, out_dir, step_num, fix_scene = True, fix_obj = True, task_name = 'stuff_higher'):
        '''
        render stand still urban scene
        '''
        raw_batch = {k : raw_data[k].clone() for k in  raw_data}
        raw_batch =  self.assign_latentcodes(raw_batch, fix_scene, fix_obj)

        stuff_id = []
        batch_size = raw_batch['frame_id'].shape[0]
 
        task_batch = []
        elevate_height = 1.5
        task_batch.append(raw_batch)
        raw_batch = self.edit_stuff(raw_batch, i = -1, raw_stuff = 11,operation='lower', distance=10)
        task_batch.append(raw_batch)
        raw_batch = self.edit_stuff(raw_batch, i = -1, raw_stuff=21, operation='higher', distance=10)
        task_batch.append(raw_batch)

        # for i in range(5):
        #     raw_batch = self.edit_stuff(raw_batch, i = i, raw_stuff = 21,operation='x_move', distance=10)
        #     task_batch.append(raw_batch)
        out =  self.batchify_render(task_batch= task_batch, generator= generator)

        self.save_img(out_dir, out, task_name)

        return task_batch

    def render_stuff_translate(self,generator, raw_data, out_dir, step_num, fix_scene = True, fix_obj = True, task_name = 'stuff_higher'):
        '''
        for frame 8062 only
        render stand still urban scene
        '''
        raw_batch = {k : raw_data[k].clone() for k in  raw_data}
        raw_batch =  self.assign_latentcodes(raw_batch, fix_scene, fix_obj)

        stuff_id = []
        batch_size = raw_batch['frame_id'].shape[0]
 
        task_batch = []
        elevate_height = 1.5
        translate_i = (0,-elevate_height,0) 
        raw_batch = self.modify_camera(raw_batch, translate = translate_i)
        task_batch.append(raw_batch)
        # for i in range(3):
        raw_batch, tree_idx = self.edit_stuff(raw_batch, i = -2, raw_stuff = 21,operation='x_move', distance=5, export_idx=True)
        task_batch.append(raw_batch)
        # for i in range(3):
        raw_batch = self.edit_stuff(raw_batch, i = -3, raw_stuff=21, operation='z_move', distance=10, external_idx=tree_idx)
        task_batch.append(raw_batch)

        # for i in range(5):
        #     raw_batch = self.edit_stuff(raw_batch, i = i, raw_stuff = 21,operation='x_move', distance=10)
        #     task_batch.append(raw_batch)
        out =  self.batchify_render(task_batch= task_batch, generator= generator)

        self.save_img(out_dir, out, task_name)

        return task_batch

    def render_road_to_grass(self, generator,raw_data, out_dir, step_num, fix_scene = True, fix_obj = True, task_name = 'road_to_grass'):
        task_batch = []
        raw_batch = {k : raw_data[k].clone() for k in  raw_data}
        raw_batch =  self.assign_latentcodes(raw_batch, fix_scene, fix_obj)
    
        # Road only
        for i in range(32):
            batch_i = {k : raw_data[k].clone() for k in  raw_data}
            batch_i = self.edit_stuff(batch_i, i = -4, raw_stuff=name2label['road'].id, target_stuff=name2label['terrain'].id, z_threshold = (64 - 2 * i))
            batch_i = self.edit_stuff(batch_i, i = -4, raw_stuff=name2label['sidewalk'].id, target_stuff=name2label['terrain'].id, z_threshold = (64 - 2 * i))
            batch_i = self.edit_stuff(batch_i, i = -4, raw_stuff=name2label['fence'].id, target_stuff=name2label['terrain'].id, z_threshold = (64 - 2 * i))
            # batch_i = self.edit_stuff(batch_i, i = -1, raw_stuff=name2label['guard rail'].id, target_stuff=name2label['terrain'].id)
            task_batch.append(batch_i) 
            
        # for i, scene in enumerate(task_batch):
        #     self.save_scene(scene, '%d'%i)
        out =  self.batchify_render(task_batch= task_batch, generator= generator) 
        self.save_img(out_dir, out, task_name)

        return task_batch
    
#?----------------------------------------------------------------------------------
#?-------------------------Utils----------------------------------------------------
#?----------------------------------------------------------------------------------
    def save_img(self, save_dir, data, task_name = '', valid_keys = ['rgb']):

        frame_num, batch_size = data['rgb'].shape[0],data['rgb'].shape[1]
        frame_id = data['frame_id'][0]
        # del data['frame_id']

        for b in range(batch_size):
            img_dir = os.path.join(save_dir, task_name,'%06d'%frame_id[b])
            if not os.path.exists(img_dir):
                os.makedirs(img_dir)
            shutil.rmtree(img_dir)
            for i in range(frame_num):
                for k in valid_keys:
                    save_tensor_img(data[k][i][b], save_dir = os.path.join(img_dir, k), name='%03d.jpg'%i, type = k)

            for k in valid_keys:
                make_gif(os.path.join(img_dir,k), '%s_%s.gif'%(task_name,k), img_dir, fps = self.fps)
                if k == 'rgb':
                    make_video(os.path.join(img_dir,k), task_name, img_dir, fps = self.fps)

                # make_gif(os.path.join(render_out_dir,'depth'), 'depth', 'out/gif', )
                print('movie has been save to:' + os.path.join(img_dir,'%s_%s.gif'%(task_name,k)))
        # for k in data:
        #     grid_name = task_name + '_' + k + '.jpg'
        #     save_tensor_img(data[k], save_dir = grid_dir,  name=grid_name)
        #     print('img grid has been save to:' + os.path.join(grid_dir ,grid_name))
    
    def save_scene(self, save_dir, data, task_name, valid_keys = \
        ['frame_id','camera_intrinsic','camera_pose','feat_raw','semantic_gird','feature_grid','bbox_tr']):

        frame_num, batch_size = data['frame_id'].shape[0],data['frame_id'].shape[1]
        frame_id = data['frame_id'][0]

        for b in range(batch_size):
            scene_dir = os.path.join(save_dir, task_name,'%06d'%frame_id[b], 'scene')
            if not os.path.exists(scene_dir):
                os.makedirs(scene_dir)
            # shutil.rmtree(scene_dir)
            for i in range(frame_num):
                scene_data = {k : data[k][i,b].cpu().numpy() for k in valid_keys}
                with open(os.path.join(scene_dir, '%03d.pkl'%i), 'wb+') as f:
                    pkl.dump(scene_data,f)
            print('scene has been save to:' + os.path.join(scene_dir,'%s.pkl'%(task_name)))
            
    def load_stuff_instanceId(self, seq_num, frame_ids):

        H, W, L = 64, 64, 64
        if self.is_kitti:
            sequence = os.path.join('2013_05_28_drive_' + '%04d'%seq_num + '_sync')
            data_dir = os.path.join(self.data_root, 'semantic_voxel', sequence, '(H:16:64,W64:64,L64:64)')

            stuff_insId_path = os.path.join(data_dir, '%010d_insId.pkl'%frame_ids)
            stuff_semantic_path = os.path.join(data_dir, '%010d.pkl'%frame_ids)
            with open(stuff_semantic_path, 'rb+') as fp: 
                stuff_semantic_idx = pkl.load(fp)
            with open(stuff_insId_path, 'rb+') as fp: 
                stuff_instance_idx = pkl.load(fp)    
        else:
            data_path = os.path.join(self.data_root, '%01d'%seq_num, 'Voxel' ,'CLEVRTEX_train_%06d.pkl'%(seq_num * 1000 + frame_ids))
            with open(data_path, 'rb') as f:
                d = pkl.load(f)
                stuff_instance_idx = {k : d['instance'][k].reshape(-1) for k in ['ground', 'wall']}
                stuff_semantic_idx = {k : d['semantic'][k].reshape(-1) for k in  ['ground', 'wall']}


        stuff_semanticId, stuff_id = np.zeros(H * W * L), np.zeros(H * W * L)
        for s in self.stuff_semantic_list:
            if stuff_semantic_idx[s].shape[0] == 0:
                continue
            if self.is_kitti:
                stuff_semanticId[stuff_semantic_idx[s]] = name2label[s].id
            else:
                stuff_semanticId[stuff_semantic_idx[s]] = {'ground':1, 'wall':2}[s]
            stuff_id[stuff_semantic_idx[s]] = stuff_instance_idx[s] + stuff_semanticId[stuff_semantic_idx[s]] * 10000
            # a = stuff_id[stuff_semantic_idx[s]]

        return stuff_id.reshape(H, W, L)
        
    def batchify_render(self,generator, task_batch, valid_keys = ['rgb', 'rgb_gt']):
        import time
        batch_num = len(task_batch)
        task_output = {k : [] for k in valid_keys + ['frame_id']}
        for i in range(batch_num):
            with torch.no_grad():
                t_start = time.time()
                output = generator(task_batch[i], mode = 'test')
                t_render = time.time() - t_start
                # print(t_render)
            for k in task_output:
                task_output[k].append(output[k])

        for k in task_output.keys():
            task_output[k] = torch.stack(task_output[k])
        # task_output['frame_id'] = task_batch[0]['frame_id']
        return task_output

    def sample_z(self, size, tmp=1.):
        z = torch.randn(*size) * tmp
        return z

    def assign_latentcodes(self, raw_data, fix_scene = False, fix_obj = False):

        raw_batch = {k : raw_data[k].clone() for k in  raw_data}
        device = raw_batch['idx'].device
        batch_size = raw_batch['idx'].shape[0]
        
        if fix_scene:
            def sample_z(x): return self.sample_z(x, tmp=.65)
            if 'z_global'in raw_batch.keys():
                del raw_batch['z_global']
            raw_batch['z_global'] = sample_z((batch_size, self.z_dim_global)).to(device).clamp(-1,1)
            # .repeat((batch_size,1))
            # raw_batch['z_global'] = raw_batch['z_global']

        if fix_obj and 'bbox_tr' in raw_batch.keys():
            if 'z_bbox'in raw_batch.keys():
                del raw_batch['z_bbox']
            def sample_z(x): return self.sample_z(x, tmp=.65)
            max_obj_num = raw_batch['bbox_tr'].shape[1]
            raw_batch['z_bbox'] = sample_z((batch_size,max_obj_num ,self.z_dim_obj)).to(device)
            # raw_batch['z_bbox'] = raw_batch['z_bbox']
        return raw_batch

#?----------------------------------------------------------------------------------
#?----------------------------------------------------------------------------------
#?----------------------------------------------------------------------------------
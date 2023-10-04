
import json
from locale import normalize
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
import matplotlib.pyplot as plt
import trimesh
from math import sin, cos, tan
from lib.utils.transform_utils import RT2tr, create_R, rotate_mat2Euler, parse_R
from lib.utils.camera_utils import build_rays, project_3dbbox

class Dataset:
    def __init__(self, data_root, seq_list, split, use_full = False):
        super(Dataset, self).__init__()
        # path and initialization
        self.use_occupancy_mask = cfg.use_occupancy_mask
        self.min_pixel_num = cfg.min_visible_pixel
        self.data_root = data_root
        self.split = split
        self.seq_list = seq_list
        self.render_obj = cfg.render_obj
        # self.render_road = cfg.render_road
        self.render_bg = cfg.render_bg
        self.render_stuff = cfg.render_stuff
        self.render_sky = cfg.render_sky
        self.max_obj_num = cfg.max_obj_num
        self.valid_object = cfg.valid_object
        self.use_depth = cfg.use_depth

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

        # load intrinsics
        self.sr_multiple = cfg.super_resolution
        assert self.sr_multiple in [1,2,4]
        calib_dir = os.path.join(data_root, 'calibration')
        self.intrinsic_file = os.path.join(calib_dir, 'perspective.txt')
        self.load_intrinsic(self.intrinsic_file)
        self.ratio = cfg.ratio
        self.H = int(self.height * cfg.ratio)
        self.W = int(self.width  * cfg.ratio)
        self.H_ray = int(self.height * cfg.ratio / self.sr_multiple)
        self.W_ray = int(self.width  * cfg.ratio / self.sr_multiple)
        self.raw_intrinsic = self.K_00[:, :-1] 
        self.intrinsic = self.K_00[:, :-1].copy()
        self.intrinsic[:2] = self.intrinsic[:2] * (cfg.ratio / self.sr_multiple)

        # Build Rays
        # for camera on the car with 1.55m height and has ~5 degree inclination the, so rectify mat can make the origin "on the ground"
        # see http://www.cvlibs.net/datasets/kitti-360/documentation.php for more detail 
        pi = 3.1415
        # rect_mat =  RT2tr(create_R((-5 * pi / 180 , 0 , 0 ),(1,1,1)), (0,-1.55,0))
        rect_mat = RT2tr(create_R((-5 * pi / 180 , 0 , 0 ), (1, 1, 1)), (0, 1.55, 0))
        # rect_mat = 0
        self.c2w_default = rect_mat
        #! p_cam @ c2w == p_world
        self.rays = build_rays(H=self.H_ray, W=self.W_ray, K=self.intrinsic, c2w=rect_mat)
        #rays_ = build_rays(H = self.H, W = self.W, K = self.intrinsic)
        # Save tmp file to reduce loading time cost

        if self.render_stuff:
            self.stuff_semanticId_list = np.array(sorted([name2label[n].id for n in self.stuff_semantic_name]))
            H, W, L = 16, 64, 64
            vox_origin = np.array((0, -32, -2))
            point_num_H, point_num_W, point_num_L, = 64, 64, 64
            vertices_gridx,  vertices_gridy, vertices_gridz= np.meshgrid(np.linspace(-W/2, W/2, point_num_W), np.linspace(2-H, 2, point_num_H), np.linspace(0, L, point_num_L), indexing='xy') 
            self.stuff_loc_gird = np.concatenate((vertices_gridx[:,:,:,None], vertices_gridy[:,:,:,None], vertices_gridz[:,:,:,None]), axis = -1)#! shape [H, W, L, 3]

            # NEw    
            vertices_gridx,  vertices_gridy, vertices_gridz= np.meshgrid(np.linspace(0, L, point_num_L), np.linspace(0, W, point_num_W), np.linspace(0, H, point_num_L), indexing='xy') # shape [W, L ,H]
            stuff_loc_gird = np.concatenate((vertices_gridx[:,:,:,None], vertices_gridy[:,:,:,None], vertices_gridz[:,:,:,None]), axis = -1) # []
            stuff_loc_gird = stuff_loc_gird + vox_origin

        
        if self.render_obj:
            self.patch_size = cfg.patch_size
            self.use_pacth_mask = cfg.use_patch_occupancy_mask
            self.vertex_threshold = cfg.vertex_threshold
            self.rate_threshold= cfg.rate_threshold
            if cfg.valid_object == []:
                obj_semantic_name = ['cars']
            else:
                obj_semantic_name = cfg.valid_object 

            self.obj_semanticId_list = np.array(sorted([name2label[n].id for n in obj_semantic_name]))

        # Load object laouot, road seamntic plane and 
        idx2frame = []
        self.layout_dict = {}
        self.layout_patehcs_dict = {}
        self.rgb_image_dict, self.depth_image_dict, self.occupancy_mask_dict = {}, {}, {}
        self.seg_image_dict = {}
        # self.road_semantic_dict, self.rays_intersection_idx_dict, self.rays_intersection_loc_dict, self.rays_semantic_dict = {}, {}, {}, {}
        self.stuff_dict = {}
        self.trajectory_dict = {}

        self.N_seq = 100000
        for seq_num in self.seq_list:
            sequence = os.path.join('2013_05_28_drive_' + '%04d'%seq_num + '_sync')
            valid_frames_file_path = os.path.join(self.data_root, 'layout', sequence , 'valid_frames.json')
            frames_trajectory_path = os.path.join(self.data_root, 'trajectory', sequence + '.pkl')

            with open(valid_frames_file_path, 'r') as fp:
                frame_ids = json.load(fp)
                if cfg.inversion:
                    frame_ids = frame_ids[:64]
            if self.use_trajectory:
                with open(frames_trajectory_path, 'rb') as fp:
                    seq_trajectory = pkl.load(fp)
                    self.trajectory_dict.update(seq_trajectory)

                idx2frame += [f + self.N_seq  * seq_num for f in frame_ids]
            self.load_image_path(seq_num=seq_num, frame_ids=frame_ids, layout_dict=self.layout_dict)
                
            if self.render_obj:
                self.load_obj_layout(seq_num, frame_ids, resume=False)
            if self.render_stuff:
                self.load_stuff_path(seq_num, frame_ids, grid_param=(H, W, L,point_num_H, point_num_W, point_num_L))

            if self.use_occupancy_mask:
                self.make_occupancy_mask(seq_num=seq_num, frame_ids= frame_ids)

            # if True:
            #     plt.imsave('tmp/aaa.jpg', self.rays_semantic_dict[300508])
            print('Load seq:%04d done!'% seq_num)

        idx2frame = [k for k in self.rgb_image_dict]

        if self.render_stuff :            
            idx2frame = [k for k in idx2frame if k in self.stuff_dict and self.stuff_dict[k] != '']

        if self.render_obj and split == 'train':
             idx2frame = [k for k in idx2frame if (k in self.layout_dict and self.layout_dict[k][-1] != '_')]
        for k in idx2frame:
            if k not in self.layout_patehcs_dict:
                self.layout_patehcs_dict[k] = ['/'] * self.max_obj_num


        # if split == 'train' and cfg.train.train_given_frame:
        #     idx2frame = [i for i in idx2frame if i in cfg.train.train_frame_list]
        if split == 'render':
            assert cfg.render.render_given_frame
            idx2frame = [i for i in idx2frame if i in cfg.render.render_frame_list]
        elif split == 'test' and cfg.test.test_given_frame:
            idx2frame = [i for i in idx2frame if i in cfg.test.test_frame_list]
        elif self.use_full_dataset:
            idx2frame = list(set(idx2frame))
        else:
            chunk_size =  self.spilt_chunk_size
            val_frame = []
            # spilt by chunk
            chunk_num = len(idx2frame) // chunk_size
            val_idx = np.random.choice(range(0, chunk_num), size=int(chunk_num * self.spilt_rate), replace=False, p=None)
            for i in val_idx:
                val_frame += idx2frame[i * chunk_size: (i + 1) * chunk_size]
            if split == 'train':
                idx2frame = list(set(idx2frame) - set(val_frame))
            elif split == 'test':
                idx2frame = list(set(val_frame))


        frame_id_list = idx2frame.copy()
        frame_id_list_gt = idx2frame.copy()
        frame_id_list_fake = idx2frame.copy()

        random.shuffle(frame_id_list)
        random.shuffle(frame_id_list_gt)
        random.shuffle(frame_id_list_fake)

        self.frame_id_list = frame_id_list
        self.frame_id_list_gt =frame_id_list_gt
        self.frame_id_list_fake = frame_id_list_fake

        self.idx2frame = {idx : frame  for idx,frame in enumerate(idx2frame)}
        self.frame2idx = {frame : idx  for idx,frame in enumerate(idx2frame)}


        print('Load dataset done!')

    
    def load_intrinsic(self, intrinsic_file):
        with open(intrinsic_file) as f:
            intrinsics = f.read().splitlines()
        for line in intrinsics:
            line = line.split(' ')
            if line[0] == 'P_rect_00:':
                K = [float(x) for x in line[1:]]
                K = np.reshape(K, [3, 4])
                self.K_00 = K
            elif line[0] == 'P_rect_01:':
                K = [float(x) for x in line[1:]]
                K = np.reshape(K, [3, 4])
                intrinsic_loaded = True
                self.K_01 = K
            elif line[0] == 'R_rect_01:':
                R_rect = np.eye(4)
                R_rect[:3, :3] = np.array([float(x) for x in line[1:]]).reshape(3, 3)
            elif line[0] == "S_rect_01:":
                width = int(float(line[1]))
                height = int(float(line[2]))
        assert (intrinsic_loaded == True)
        assert (width > 0 and height > 0)
        self.width, self.height = width, height
        self.R_rect = R_rect

    def load_image_path(self, seq_num, frame_ids, layout_dict, N_seq = 100000, save_cache_file = True, resume = False):
        #image_dict = {}
        rgb_image_dict, depth_image_dict ,occupancy_mask_dict = {}, {}, {}
        #assert frame_ids ==  list(layout_dict.keys())

        sequence = os.path.join('2013_05_28_drive_' + '%04d'%seq_num + '_sync')
        rgb_image_root = os.path.join(self.data_root, 'data_2d_raw', sequence, 'image_00', 'data_rect') 
        # depth_image_root = os.path.join(self.data_root, 'data_2d_depth_correct', sequence, 'image_00', 'depth_rect') 
        depth_image_root = os.path.join(self.data_root, 'stereo_depth', sequence, 'image_00', 'data_rect') 
        semantic_image_root =  os.path.join(self.data_root, 'data_2d_semantics', 'train', sequence, 'image_00', 'instance') 

        for idx in frame_ids:
            #rgb_image_path = imageio.imread(os.path.join(rgb_image_root, '%010d.png'%idx))
            rgb_image_path = os.path.join(rgb_image_root, '%010d.png'%idx)
            depth_image_path = os.path.join(depth_image_root, '%010d.png'%idx)
            # occupancy_mask_path = os.path.join(occupancy_mask_root, '%010d.jpg'%idx)
            semantic_image_path = os.path.join(semantic_image_root, '%010d.png'%idx)
            
            self.rgb_image_dict[idx + seq_num * N_seq] = rgb_image_path
            self.seg_image_dict[idx + seq_num * N_seq] = semantic_image_path
            self.depth_image_dict[idx + seq_num * N_seq] = depth_image_path
            # occupancy_mask_dict[idx + seq_num * N_seq] = occupancy_mask_path
            # if self.render_obj:
            #     self.layout_patehcs_dict[idx + seq_num * N_seq] = layout_patches


    def make_occupancy_mask(self, seq_num, frame_ids, N_seq = 100000, resume = False):
        info = []
        sequence = os.path.join('2013_05_28_drive_' + '%04d'%seq_num + '_sync')
        if self.render_stuff or self.render_bg:
            stuff_info = 'stuff(normal'
            if'building' in self.stuff_semantic_name:
                stuff_info += ',building'   
            if'car' in self.stuff_semantic_name:
                stuff_info += ',car' 
            stuff_info += ')'
            info.append(stuff_info)

        if self.render_sky or self.render_bg:
            info.append('sky')
        if self.render_obj:
            for n in self.valid_object:
                info.append(n + ',p:%d'%self.min_pixel_num[n])
            info.append('N:'+str(cfg.max_obj_num))
        info = '_'.join(info)
        occupancy_mask_root = os.path.join(self.data_root, 'occupancy', info, sequence)
        if not os.path.exists(occupancy_mask_root):
            os.makedirs(occupancy_mask_root)
        # patches_root = os.path.join(self.data_root, 'object_patches', sequence)
        # if not os.path.exists(patches_root):
        #     os.mkdir(patches_root)
        for idx in frame_ids:
            semantic_image_path = self.seg_image_dict[idx + seq_num * N_seq]
            rgb_image_path = self.rgb_image_dict[idx + seq_num * N_seq]
            occupancy_mask_path = os.path.join(occupancy_mask_root, '%010d.jpg'%idx)

            if not os.path.exists(occupancy_mask_path) or resume :
                # Get occupancy mask
                seg_image = imageio.imread(semantic_image_path)
                seg_semanticId = np.asarray(seg_image// 1000 )
                seg_globalId = seg_semanticId.astype('int32') * 10000 + seg_image % 1000
                #instanceId = np.asarray( globalId % 1000 )    
                occupancy_mask = np.zeros_like(seg_image)
        
                # Select semantics for occupancy mask
                if self.render_stuff or self.render_bg:
                    for s in self.stuff_semanticId_list:
                        occupancy_mask += np.where(seg_semanticId == s, np.ones_like(occupancy_mask) * s, np.zeros_like(occupancy_mask))
                if self.render_sky or self.render_bg:
                    sky_occupancy_mask = np.where(seg_semanticId == name2label['sky'].id, np.ones_like(occupancy_mask), np.zeros_like(occupancy_mask))
                    occupancy_mask += sky_occupancy_mask 

                if self.render_obj:
                    layout_path = self.layout_dict[idx + seq_num * N_seq]
                    layout = self.load_layout(layout_path)                    
                    layout_globalId = np.array([ o['globalId'] for o in layout])
                    for g in layout_globalId:
                        if g != -1:
                            occupancy_mask += np.where(seg_globalId == g, np.ones_like(occupancy_mask) * (g // 10000), np.zeros_like(occupancy_mask))

                occupancy_mask= occupancy_mask.astype('uint8')
                # with open(occupancy_mask_path, 'wb+') as fp: 
                imageio.imsave(occupancy_mask_path, occupancy_mask)
            self.occupancy_mask_dict[idx + seq_num * N_seq] = occupancy_mask_path


    def load_obj_layout(self, seq_num, frame_ids, N_seq = 100000, resume = True):
        layout_dict, patches_dict = {}, {}
        sequence = os.path.join('2013_05_28_drive_' + '%04d'%seq_num + '_sync')
        cache_root = os.path.join(self.data_root, 'cache')
        # Check if already get well processed layout
        info = []
        for n in self.valid_object:
            info.append(n + ',p:%d'%self.min_pixel_num[n])
        info.append('N:%d'%self.max_obj_num)
        info = '_'.join(info)
        
        fine_layout_root = os.path.join(cache_root, 'layout_fine', info, sequence)
        obj_patches_root = os.path.join(self.data_root, 'object_patches','patchsize:%d_ratio:%.3f'%(self.patch_size, self.ratio), sequence)
        # if not os.path.exists(obj_patches_root):
        #     os.makedirs(obj_patches_root)
        if not os.path.exists(fine_layout_root):
            os.makedirs(fine_layout_root)

        for idx in frame_ids:
            fine_layout_path = os.path.join(fine_layout_root, '%010d.pkl'%(idx))
            if os.path.exists(fine_layout_path) and not resume:
                layout_dict[idx + self.N_seq * seq_num] = fine_layout_path
                patches_dict[idx + self.N_seq * seq_num] = self.load_patches_path(seq_num, idx,self.load_layout(fine_layout_path), obj_patches_root)
            elif os.path.exists(fine_layout_path + '_') and not resume:
                layout_dict[idx + self.N_seq * seq_num] = fine_layout_path + '_'
            else:
                layout_dict[idx + self.N_seq * seq_num], patches_dict[idx + self.N_seq * seq_num] = self.make_obj_layout(seq_num, idx, fine_layout_path, obj_patches_root)
        self.layout_dict.update(layout_dict)
        self.layout_patehcs_dict.update(patches_dict)
        #if not os.path.exists(tmp_layout_root):

    def load_patches_path(self, seq_num, idx, layout, obj_patches_root):
        # Empty layout
        patches_path = []
        if layout[0]['globalId'] == -1:
            pass
        else:
            obj_patches_dir = os.path.join( obj_patches_root, '%010d'%idx)
            for o in layout:
                if o['globalId'] == -1:
                    break
                rgb_patch_path = os.path.join(obj_patches_dir, '%05d_rgb.jpg'%o['globalId'])
                if idx == 207987:
                    print('?>')
                if not os.path.exists(rgb_patch_path):
                    seg_img = imageio.imread(self.seg_image_dict[idx + self.N_seq * seq_num])
                    rgb_img = imageio.imread(self.rgb_image_dict[idx + self.N_seq * seq_num])
                    self.make_obj_patch(o, 
                                        rgb_img= rgb_img,
                                        seg_img= seg_img,
                                        patch_dir= obj_patches_dir,
                                        patch_size= self.patch_size,
                                        save_it_anyway = True)

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



    def make_obj_layout(self, seq_num, idx, fine_layout_path, obj_patches_root, N_seq = 100000):
        if idx == 480:
            print('s')

        layout, patches_path = [], []
        sequence = os.path.join('2013_05_28_drive_' + '%04d'%seq_num + '_sync')
        raw_layout_root = os.path.join(self.data_root, 'layout', sequence)
        raw_layout_path = os.path.join(raw_layout_root, '%010d.pkl'%idx)
        if not os.path.exists(raw_layout_path):
                raise RuntimeError('%s does not exist!' % raw_layout_path)
        with open(raw_layout_path, 'rb+') as fp:
            raw_layout = pkl.load(fp)

       
        sequence = os.path.join('2013_05_28_drive_' + '%04d'%seq_num + '_sync')
        obj_patches_dir = os.path.join(obj_patches_root, '%010d'%idx)
        # if not os.path.exists(obj_patches_dir):
        #         os.makedirs(obj_patches_dir)
        seg_img = imageio.imread(self.seg_image_dict[idx + self.N_seq * seq_num])
        rgb_img = imageio.imread(self.rgb_image_dict[idx + self.N_seq * seq_num])
        temp_layout = {}

        for k, v  in raw_layout.items():
            if (v.semanticId in self.obj_semanticId_list and v.pixel_num > self.min_pixel_num[v.name]):
                temp_layout[k] = v
        layout_, patches_path_ = [], []
        for k in temp_layout:
            obj = temp_layout[k]
            bbox_tr = RT2tr(obj.R, obj.T)
            o = {               
                    "globalId": k, 
                    "annotationId": obj.annotationId, 
                    "semanticId": obj.semanticId,  
                    "bbox_tr": bbox_tr,
                    # "bbox_uv": bbox_uv,
                    # "valid_vertices_num":valid_vertices_num,
                    "pixel_num": obj.pixel_num}
            is_valid, patch_path = self.make_obj_patch(o,rgb_img,seg_img, obj_patches_dir,self.patch_size,self.vertex_threshold,self.rate_threshold)
            if is_valid:
                layout_.append(o)
                patches_path_ += [patch_path]

        # layout_ = [o for o in layout_ if o["valid_vertices_num"] == 8] 
        # or o["pixel_num"] > self.min_pixel_num['car'] * 10]

        reidx = np.argsort(-np.array([k["pixel_num"] for k  in layout_]))

        #argsort(-x)
        layout = [layout_[i] for i in reidx]
        patches_path = [patches_path_[i] for i in reidx]
        mask_tr = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        if len(layout_) > self.max_obj_num:
            layout = layout[:self.max_obj_num]
            patches_path = patches_path[:self.max_obj_num]
        else: 
            # padding
            patches_path += ['/']*  (self.max_obj_num - len(layout))
            layout += [{
                "globalId": -1, 
                "semanticId": -1,  
                "bbox_tr": mask_tr} ]*  (self.max_obj_num - len(layout))
        #layout_dict[idx + self.N_seq * seq_num] = layout
        if layout[0]["globalId"] == -1:
            fine_layout_path += '_'
            
        with open(fine_layout_path, 'wb+') as f: 
            pkl.dump(layout, f)

        # # Padding
        # patches_path += '/' * (len(layout) - len(patches_path))
        return fine_layout_path, patches_path


    def make_obj_patch(self, obj, rgb_img, seg_img, patch_dir, patch_size = 64, vertex_threshold = 5, occupancy_ratio_threshold = 0.25 , save_it_anyway = False):
        obj_globalId,  obj_tr = obj['globalId'], obj['bbox_tr']
        valid_vertex_num, _ = project_3dbbox(obj_tr, self.c2w_default, self.raw_intrinsic, render = False)
        if valid_vertex_num < vertex_threshold:
            return False, '/'
        
        rgb_patch = os.path.join(patch_dir, '%05d_rgb.jpg'%(obj_globalId))
        pose_patch = os.path.join(patch_dir, '%05d_pose.jpg'%obj_globalId)
        occupancy_patch =  os.path.join(patch_dir, '%05d_occupancy.jpg'%obj_globalId)

        seg_semanticId = np.asarray(seg_img // 1000 )
        seg_globalId = seg_semanticId.astype('int32') * 10000 + seg_img % 1000
        if os.path.exists(rgb_patch):
            pose_img  = imageio.imread(pose_patch)
            obj_rgb = imageio.imread(rgb_patch)
            obj_occupancy = imageio.imread(occupancy_patch)
        else:
            pose_img = project_3dbbox(obj_tr, self.c2w_default, self.raw_intrinsic, render = True)
            uv_img = np.sum(pose_img, axis = -1)
            t = np.argwhere(uv_img != 3)
            if t.size == 0:
                t = np.array([[0,1],[1,0]])
                print('........................!!!!!!!!!!!!!!!!!!!!!!')
            v_max, v_min, u_max, u_min = t[:,0].max(), t[:,0].min(),t[:,1].max(),t[:,1].min()
            # obj_pose = pose_img[v_min:v_max,u_min:u_max]
            # obj_pose = cv2.resize(obj_pose, (patch_size,patch_size), interpolation=cv2.INTER_AREA)


            # obj_occupancy = seg_globalId[v_min:v_max,u_min:u_max]
            occupancy_img = np.where(seg_globalId == obj_globalId, np.ones_like(obj_globalId) * 50, np.zeros_like(obj_globalId)).astype('uint8')
            
            imageio.imsave('tmp/aa.jpg',rgb_img)

            rgb_img = cv2.resize(rgb_img, (self.W, self.H), interpolation=cv2.INTER_LANCZOS4)
            occupancy_img = cv2.resize(occupancy_img, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
            pose_img = cv2.resize(pose_img, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
            
            v_min , v_max, u_min, u_max = \
                int(v_min * self.ratio)  , int(v_max * self.ratio) , int(u_min * self.ratio),int(u_max * self.ratio)
            
            obj_rgb = rgb_img[v_min :v_max ,u_min :u_max ]
            obj_occupancy = occupancy_img[v_min :v_max ,u_min :u_max ]
            obj_pose = pose_img[v_min :v_max ,u_min :u_max ]


            h_patch, w_patch=  obj_rgb.shape[0], obj_rgb.shape[1]
            if h_patch <= w_patch:
                pad_num = int((w_patch - h_patch)/2)
                obj_rgb = cv2.copyMakeBorder(obj_rgb,pad_num,pad_num,0,0, cv2.BORDER_CONSTANT, value=(0,0,0))
                obj_occupancy = cv2.copyMakeBorder(obj_occupancy,pad_num,pad_num,0,0, cv2.BORDER_CONSTANT, value=(0))
                obj_pose = cv2.copyMakeBorder(obj_pose,pad_num,pad_num,0,0, cv2.BORDER_CONSTANT, value=(1,1,1))
            else: 
                pad_num = int((h_patch - w_patch)/2)
                obj_rgb = cv2.copyMakeBorder(obj_rgb,0,0,pad_num,pad_num, cv2.BORDER_CONSTANT, value=(0,0,0))
                obj_occupancy = cv2.copyMakeBorder(obj_occupancy,0,0,pad_num,pad_num, cv2.BORDER_CONSTANT, value=(0))
                obj_pose = cv2.copyMakeBorder(obj_pose,pad_num,pad_num,0,0, cv2.BORDER_CONSTANT, value=(255,255,255))

            
            imageio.imsave('tmp/bb.jpg',obj_rgb)
            obj_rgb = cv2.resize(obj_rgb, (patch_size,patch_size), interpolation=cv2.INTER_AREA)
            obj_occupancy = cv2.resize(obj_occupancy, (patch_size,patch_size), interpolation=cv2.INTER_NEAREST)
            obj_pose = cv2.resize(obj_pose, (patch_size,patch_size), interpolation=cv2.INTER_NEAREST)

        occupancy_ratio = np.sum(obj_occupancy / 50) / (patch_size * patch_size)

        if occupancy_ratio >= occupancy_ratio_threshold or save_it_anyway:
            if not os.path.exists(patch_dir):
                os.makedirs(patch_dir)
            if not os.path.exists(rgb_patch):
                imageio.imsave(pose_patch,obj_pose)
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
    def load_img(path, W = 352, H = 94, type = 'rgb', sr_multiple = 1):
        image = imageio.imread(path).astype(np.float32)
        # /data/ybyang/KITTI-360/data_2d_depth/2013_05_28_drive_0000_sync/image_00/depth_rect/0000000320.png
        if type == 'rgb':
            mean=127.5
            std=127.5
            rgb_image = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)
            rgb_image = (rgb_image - mean) / std
            return rgb_image
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

        if self.render_stuff:
            ret['stuff_semantic_grid'] = self.load_stuff_semantic(self.stuff_dict[frame_id], self.stuff_semantic_name)
            ret['stuff_loc_grid'] = self.stuff_loc_gird

            ret['stuff_semantic_grid_fake']= self.load_stuff_semantic(self.stuff_dict[frame_id_fake], self.stuff_semantic_name)
            ret['stuff_loc_grid_fake'] = self.stuff_loc_gird

        if self.render_obj:
            layout_gt = self.load_layout(self.layout_dict[frame_id_gt])
            ret["bbox_patches_gt"] = self.load_bbox_pathces(self.layout_patehcs_dict[frame_id_gt],self.patch_size, self.use_pacth_mask)[0].transpose(0,3,1,2)
            ret["bbox_patches_fake"] = self.load_bbox_pathces(self.layout_patehcs_dict[frame_id_fake], self.patch_size, self.use_pacth_mask)[0].transpose(0,3,1,2)
            ret["bbox_patches"] = self.load_bbox_pathces(self.layout_patehcs_dict[frame_id], self.patch_size, self.use_pacth_mask)[0].transpose(0,3,1,2)
            
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

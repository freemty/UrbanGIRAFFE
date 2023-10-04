
import json
import numpy as np
import os
import random
from yaml import load
from lib.config import cfg, args
import imageio

from tools.kitti360Scripts.helpers.labels import labels, name2label
from lib.utils.camera_utils import build_rays
import pickle as pkl
import cv2

# from ..base.urbangiraffe import Dataset as urbangiraffeBase
from utils_scripts.voxel_convert import init_voxel_grid_default, init_voxel_grid_kitti, convert2defaultvox
class Dataset():
    def __init__(self, data_root, seq_list, split, use_full = False):
        super().__init__(data_root, seq_list, split, use_full)
        # path and initialization

        # load intrinsics
        camera_file = os.path.join(data_root,'data_3d_voxel','08','L64:1.00,W64:1.00,H16:0.25','000000.pkl')
        with open(camera_file,'rb+') as fp:
            camera_info = pkl.load(fp)
        raw_intrinsic = camera_info['K']
        self.height = 370
        self.width = 1226
        self.intrinsic = raw_intrinsic.copy()
        self.intrinsic[:2] = self.intrinsic[:2] * (cfg.ratio / self.sr_multiple)
        

        self.H_ray = round(self.height * cfg.ratio / self.sr_multiple)
        self.W_ray = round(self.width  * cfg.ratio / self.sr_multiple)
        self.H = self.H_ray * self.sr_multiple
        self.W = self.W_ray * self.sr_multiple
        c2w_kitti = camera_info['c2w']
        kitti2default = np.array(
                    [[0,-1,0,0],
                    [0,0,-1,0],
                    [1,0,0,0],
                    [0,0,0,1]]) # Coodriante transformation

        c2w_default = kitti2default @ c2w_kitti
        self.rays = build_rays(H = self.H_ray, W = self.W_ray, K = self.intrinsic, c2w= c2w_default)
        self.c2w_default = c2w_default

        self.semanticId_list = []

        if self.render_stuff:
            self.semanticId_list += [name2label[n].id  for n in self.stuff_semantic_name]
            voxel_grid_param_kitti = {
                'scene_scale':(64, 64, 16),
                'vox_origin':(0, -32, -2),
                'vox_size':(1, 1, 0.25)}
            voxel_grid_param_default = {
                'scene_scale':(64, 16, 64),
                'vox_origin':(-32, 14, -2),
                'vox_size':(1, 0.25, 1)}
            
            stuff_loc_gird_kitti = init_voxel_grid_kitti(p_LHW = np.array((64,64,64)))
            stuff_loc_gird_defult = init_voxel_grid_default(p_LHW = np.array((64,64,64)))
            self.stuff_loc_gird_defult = stuff_loc_gird_defult

        if self.render_obj:
            self.semanticId_list += [name2label['car'].id]
            self.patch_size = cfg.patch_size
            self.use_pacth_mask = cfg.use_patch_occupancy_mask
            self.vertex_threshold = cfg.vertex_threshold
            self.rate_threshold= cfg.rate_threshold
            if cfg.valid_object == []:
                obj_semantic_name = ['cars']
            else:
                obj_semantic_name = cfg.valid_object 

            self.obj_semanticId_list = np.array(sorted([name2label[n].id for n in obj_semantic_name]))

        if self.render_sky:
            self.semanticId_list += [name2label['sky'].id]

        # Load object laouot, road seamntic plane and 
        idx2frame = []
        self.rgb_dict, self.semantic_dict = {}, {}
        if self.render_obj:
            self.layout_dict, self.layout_patehcs_dict = {}, {}
        if self.render_stuff:
            self.stuff_dict = {}
        if self.use_depth:
            self.depth_dict= {}
        if self.use_trajectory:
            self.trajectory_dict = {}
        
        self.N_seq = 10000
        for seq_num in self.seq_list:
            seq = '%02d'%seq_num
            frame_ids = [int(idx[:-4]) for idx in os.listdir(os.path.join(data_root, 'data_2d_raw', seq, 'kitti', 'testing', 'image_2'))]
            frame_ids.sort()

            idx2frame += [f + self.N_seq  * seq_num for f in frame_ids]
            self.load_image_path(seq_num=seq_num, frame_ids= frame_ids)
                
            # if self.render_obj:
            #     self.load_obj_layout(seq_num, frame_ids)
            if self.render_stuff:
                self.load_stuff_path(seq_num, frame_ids)

            print('Load seq:%04d done!'% seq_num)

        idx2frame = [k for k in self.rgb_dict]

        if self.render_stuff :            
            idx2frame = [k for k in idx2frame if k in self.stuff_dict and self.stuff_dict[k] != '']

        # if self.render_obj:
        #     idx2frame = [k for k in idx2frame if (k in self.layout_dict and self.layout_dict[k][-1] != '_')]

        if split == 'render':
            assert cfg.render.render_given_frame
            idx2frame = [i for i in idx2frame if i in cfg.render.render_frame_list]
        # elif split == 'test' and cfg.test.test_given_frame:
        #     idx2frame = [i for i in idx2frame if i in cfg.test.test_frame_list]
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
        idx2frame.sort()

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


    def load_image_path(self, seq_num, frame_ids, N_seq = 10000):
        seq = '%02d'%seq_num
        rgb_dir = os.path.join(self.data_root, 'data_2d_raw', seq,'kitti', 'testing', 'image_2')
        semantic_dir = os.path.join(self.data_root, 'data_2d_semantic', seq, 'ckpt', 'test', 'pred')

        for idx in frame_ids:
            rgb_image_path = os.path.join(rgb_dir, '%06d.png'%idx)
            semantic_image_path = os.path.join(semantic_dir, '%06d.png'%idx)
            self.rgb_dict[idx + seq_num * N_seq] = rgb_image_path
            self.semantic_dict[idx + seq_num * N_seq] = semantic_image_path

    def load_stuff_path(self, seq_num, frame_ids,grid_param = (64,1.0,64,1.0,16,0.25) ,  N_seq = 10000):
        seq = '%02d'%seq_num
        # l, lres,  w, wres,  h, hres, grid_param
        voxel_pram = 'L%d:%.2f,W%d:%.2f,H%d:%.2f'%grid_param
        voxel_dir = os.path.join(self.data_root, 'data_3d_voxel', seq, voxel_pram)
        for idx in frame_ids:
            stuff_file_path = os.path.join(voxel_dir, '%06d.pkl'%idx)
            if not os.path.exists(stuff_file_path):
                self.stuff_dict[idx + seq_num * N_seq] = ''
            else:
                self.stuff_dict[idx + seq_num * N_seq] = stuff_file_path

    def make_occupancy_mask(self, seq_num, frame_ids, N_seq = 10000, resume = False):
        info = []
        seq = '%02d'%seq_num
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
                info.append(n)
        info = '_'.join(info)
        mask_dir = os.path.join(self.data_root, 'occupancy', seq, info)
        if not os.path.exists(mask_dir):
            os.makedirs(mask_dir)

        for idx in frame_ids:
            semantic_path = self.semantic_dict[idx + seq_num * N_seq]
            rgb_path = self.rgb_dict[idx + seq_num * N_seq]
            occupancy_mask_path = os.path.join((mask_dir, '%06d.jpg'%idx))

            if not os.path.exists(occupancy_mask_path) or resume:
                # Get occupancy mask
                seg_image = imageio.imread(semantic_path)
                seg_semanticId = np.asarray(seg_image// 1000 )
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
                    obj_occupancy_mask = np.where(seg_semanticId == name2label['sky'].id, np.ones_like(occupancy_mask), np.zeros_like(occupancy_mask))
                    occupancy_mask += sky_occupancy_mask 

                occupancy_mask= occupancy_mask.astype('uint8')
                with open(occupancy_mask_path, 'wb+') as fp: 
                        imageio.imsave(fp, occupancy_mask)
            self.mask_dict[idx + seq_num * N_seq] = occupancy_mask_path


    def load_obj_layout(self, seq_num, frame_ids, N_seq = 10000, resume = True):
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
            fine_layout_path = os.path.join(fine_layout_root, '%06d.pkl'%(idx))
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
            obj_patches_dir = os.path.join( obj_patches_root, '%06d'%idx)
            for o in layout:
                if o['globalId'] == -1:
                    break
                rgb_patch_path = os.path.join(obj_patches_dir, '%05d_rgb.jpg'%o['globalId'])
                if idx == 11652:
                    print('?>')
                if not os.path.exists(rgb_patch_path):
                    seg_img = imageio.imread(self.semantic_dict[idx + self.N_seq * seq_num])
                    rgb_img = imageio.imread(self.rgb_dict[idx + self.N_seq * seq_num])
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

    def get_occupancy_mask(self, semantic_path,W , H,semanticId_list = np.array([]) ):

        seg_image = imageio.imread(semantic_path)
        semanticId_list = semanticId_list
        #instanceId = np.asarray( globalId % 1000 )    
        valid_idx = np.isin(seg_image, semanticId_list)

        seg_image[valid_idx == 0] = 0

        occupancy_mask = cv2.resize(seg_image, (W, H),interpolation=cv2.INTER_NEAREST)[:,:,None]

        return occupancy_mask

    def load_semantic_voxel(self, voxle_path):

        with open(voxle_path, 'rb') as fp:
            f = pkl.load(fp)
            seamntic_voxel_kitti = f['stuff_semantic_voxel'] #X Y Z
        seamntic_voxel_default = convert2defaultvox(seamntic_voxel_kitti.transpose((1,0,2)))

        return seamntic_voxel_default

    
    def __getitem__(self, index):

        camera_mat = self.intrinsic
        
        # Load scene for G training
        frame_id = self.frame_id_list[index]
        frame_id_gt = self.frame_id_list_gt[index]
        frame_id_fake = self.frame_id_list_fake[index]

        idx = self.frame2idx[frame_id]
        idx_gt = self.frame2idx[frame_id_gt]
        idx_fake = self.frame2idx[frame_id_fake]

        # Load gt image for D training
        rgb = self.rgb_dict[frame_id]
        rgb_gt = self.rgb_dict[frame_id_gt]
        rgb_fake = self.rgb_dict[frame_id_fake]

        c2w = c2w_fake = self.c2w_default
        rays = rays_fake = self.rays

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

        if self.render_stuff:
            loc_grid = self.stuff_loc_gird_defult
            ret['stuff_semantic_grid'] = self.load_semantic_voxel(self.stuff_dict[frame_id])
            ret['stuff_loc_grid'] = loc_grid

            ret['stuff_semantic_grid_fake']= self.load_semantic_voxel(self.stuff_dict[frame_id_fake])
            ret['stuff_loc_grid_fake'] = loc_grid

        if self.use_occupancy_mask:
            semantic = self.semantic_dict[frame_id]
            semantic_gt = self.semantic_dict[frame_id_gt]
            semantic_fake = self.semantic_dict[frame_id_fake]


            ret['occupancy_mask'] = self.get_occupancy_mask(semantic,W = self.W, H = self.H,  semanticId_list=self.semanticId_list).transpose(2,0,1) 
            ret['occupancy_mask_gt'] = self.get_occupancy_mask(semantic_gt,W = self.W, H = self.H,  semanticId_list=self.semanticId_list).transpose(2,0,1) 
            ret['occupancy_mask_fake'] = self.get_occupancy_mask(semantic_fake,W = self.W, H = self.H,  semanticId_list=self.semanticId_list).transpose(2,0,1) 

        if self.use_depth:
            ret['depth'] = self.load_img(depth,W = self.W, H = self.H, type = 'depth',sr_multiple = self.sr_multiple).transpose(2,0,1)
            ret['depth_fake'] = self.load_img(depth_fake,W = self.W, H = self.H, type = 'depth', sr_multiple = self.sr_multiple).transpose(2,0,1)
            ret['depth_gt'] =  self.load_img(depth_gt,W = self.W, H = self.H, type = 'depth',sr_multiple = self.sr_multiple).transpose(2,0,1)
            # ret['occupancy_mask_sky' ] = self.load_occupancy_mask(occupancy_mask, semantic = 1,to_bin = True,erode = True).transpose(2,0,1) 
            # ret['occupancy_mask_sky_gt'] =  self.load_occupancy_mask(occupancy_mask_gt, semantic = 1,to_bin = True,erode = True).transpose(2,0,1) 

        if self.render_obj:
            # layout_gt = self.load_layout(self.layout_dict[frame_id_gt])
            # ret["bbox_patches_gt"] = self.load_bbox_pathces(self.layout_patehcs_dict[frame_id_gt],self.patch_size, self.use_pacth_mask)[0].transpose(0,3,1,2)
            # ret["bbox_patches_fake"] = self.load_bbox_pathces(self.layout_patehcs_dict[frame_id], self.patch_size, self.use_pacth_mask)[0].transpose(0,3,1,2)
            # ret["bbox_patches"] = self.load_bbox_pathces(self.layout_patehcs_dict[3400], self.patch_size, self.use_pacth_mask)[0].transpose(0,3,1,2)
            # ret["bbox_pose_patches"] = self.load_bbox_pathces(self.layout_patehcs_dict[frame_id],self.patch_size)[1].transpose(0,3,1,2)
            
            # ret["bbox_pose_patches_gt"] = self.load_bbox_pathces(self.layout_patehcs_dict[frame_id_gt],self.patch_size, self.use_pacth_mask)[1].transpose(0,3,1,2)
            # ret["bbox_pose_patches_fake"] = self.load_bbox_pathces(self.layout_patehcs_dict[frame_id_fake], self.patch_size)[1].transpose(0,3,1,2)


            # ret['bbox_id_gt'] = np.array([bbox['globalId'] for bbox in layout_gt])
            # ret['bbox_semantic_gt'] = np.array([bbox['semanticId'] for bbox in layout_gt])
            # ret['bbox_tr_gt'] = np.array([bbox['bbox_tr'] for bbox in layout_gt]).astype(np.float32)
            # ret['bbox_pose_gt'] = self.parse_bbox_pose(ret['bbox_tr_gt'], self.c2w_default).astype(np.float32)

            layout = self.load_layout('/data/ybyang/KITTI-360/cache/layout_fine/car,p:5000_N:4/2013_05_28_drive_0000_sync/0000000250.pkl')
            ret['bbox_id'] = np.array([bbox['globalId'] for bbox in layout])
            ret['bbox_semantic'] = np.array([bbox['semanticId'] for bbox in layout])
            ret['bbox_tr'] = np.array([bbox['bbox_tr'] for bbox in layout]).astype(np.float32)
            ret['bbox_pose'] = self.parse_bbox_pose(ret['bbox_tr'], ret['world_mat']).astype(np.float32)


            # layout_fake = self.load_layout(self.layout_dict[frame_id_fake])
            # ret['bbox_id_fake'] = np.array([bbox['globalId'] for bbox in layout_fake])
            # ret['bbox_semantic_fake'] = np.array([bbox['semanticId'] for bbox in layout_fake])
            # ret['bbox_tr_fake'] = np.array([bbox['bbox_tr'] for bbox in layout_fake]).astype(np.float32)
            # ret['bbox_pose_fake'] = self.parse_bbox_pose(ret['bbox_tr_fake'], ret['world_mat_fake']).astype(np.float32)


        return ret

    def __len__(self):
        return len(self.frame_id_list)


import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision

import re
from tools.kitti360Scripts.helpers.labels import name2label
from lib.config import cfg
from lib.utils.img_utils import save_tensor_img
from .utils import *
import time

from collections import OrderedDict
from .base import GANGenerator

from lib.networks.models import *
from lib.networks.volumetric_rendering import skyRayMarcher, NeRFRenderer, stuffRayMarcher, bboxRayMarcher

bbox_semantic_translate = {-1:0,11:1,17:2,19:3,20:4,26:5,27:6,28:7,34:8,36:9}

neural_renderer_dict = {
    'NR_legacy': CNNRender_legacy,
    'baseNR': SuperresolutionBase,
    'eg3dNR': SuperresolutionHybrid,
    'legacyNR': CNNRender_legacy,  
    'giraffeNR': RenderNet2d
}


class Generator(GANGenerator):
    def __init__(self,
        # obj_decoder_type = 'giraffe_ins',
        # stuff_decoder_type = 'local_feature',
        # sky_decoder_type = 'sky',
        z_grid_generator_type = 'SPADE3D',
        z_map_generator_type = 'SPADE',
        neural_renderer_type = 'gancraftNR',
        use_neural_renderer = False,
        ray_voxel_sampling = True,
        sample_before_intersection = False,
        use_obj_mapper = False,
        use_z_map = True,
        z_trainable = False,
        use_scale_condition = False,
        z_dim_global = 64,
        feature_dim = 32,
        local_feature_dim = 16,
        semantic_dim = 16,
        z_dim_obj = 80,
        z_dim_road = 80,
        z_dim_stuff = 80,
        n_samples_obj = 24,
        n_samples_stuff = 12,
        n_vox_intersection = 2,
        n_samples_obj_render = 24,
        n_samples_stuff_render = 12,
        n_vox_intersection_render = 2,
        **kwargs):
 
        super(Generator, self).__init__()
        self.nerf_renderer = NeRFRenderer()

        self.render_option.render_stuff = cfg.render_stuff
        self.render_option.render_sky = cfg.render_sky
        self.render_option.render_obj = cfg.render_obj
        self.render_option.use_neural_renderer = use_neural_renderer
        self.render_option.sr_multiple = cfg.super_resolution
        self.render_option.use_scale_condition = use_scale_condition

        # Feature dim
        kwargs['stuff_decoder_kwargs'].out_channel = kwargs['sky_decoder_kwargs'].out_channel = kwargs['neural_render_kwargs'].in_channel = kwargs['obj_decoder_kwargs'].out_channel = feature_dim
        # Local Feature dim
        local_feature_dim =  kwargs['stuff_decoder_kwargs'].w_channel = kwargs['z_grid_generator_kwargs'].out_channel
        # z_dim
        kwargs['stuff_decoder_kwargs'].style_channel = kwargs['z_grid_generator_kwargs'].style_dim = kwargs['sky_decoder_kwargs'].style_channel = kwargs['neural_render_kwargs'].style_channel = z_dim_global
        # z_dim_obj
        kwargs['obj_decoder_kwargs'].z_dim = z_dim_obj

        # super_resolution
        kwargs['neural_render_kwargs'].sr_multiple = self.render_option.sr_multiple 


        self.render_bg = True if cfg.task == 'GIRAFFE' else False

        
        # urbangiraffe three submodule
        if self.render_option.render_obj:

            if not self.render_option.use_scale_condition:
                kwargs['obj_decoder_kwargs']['c_dim'] = 0
            self.nerf_obj = giraffeInsDecoder(
                **kwargs['obj_decoder_kwargs'])
            
            self.nerf_renderer.ray_marcher['obj'] = bboxRayMarcher(decoder = self.nerf_obj)

        if self.render_option.render_stuff:
            self.stuff_generator_type = z_grid_generator_type
            self.semantic_nc = kwargs['z_grid_generator_kwargs'].semantic_channel
            self.voxel_range = torch.tensor(cfg.voxel_range, device=self.device)
            self.nerf_stuff = StuffDecoder(**kwargs['stuff_decoder_kwargs'])
            self.nerf_renderer.ray_marcher['stuff'] = stuffRayMarcher(decoder = self.nerf_stuff)
            # stuff_raymarcher = stuffRayMarcher(decoder = self.stuff_decoder)
            if z_grid_generator_type == 'VolumeGenerator':
                self.stuff_representation_type = 'grid'
                self.z_grid_generator = FeatureVolumeGenerator(**kwargs['z_grid_generator_kwargs'])
            elif z_grid_generator_type == 'Triplane':
                self.stuff_representation_type = 'triplane'
                triplane_kwargs = kwargs['triplane_generator_kwargs']
                triplane_kwargs['img_channels'] = local_feature_dim
                triplane_kwargs['z_dim'] = z_dim_global
                triplane_kwargs['img_resolution'] = 256
                self.z_grid_generator = TriPlaneGenerator(**kwargs['triplane_generator_kwargs'])
            else:
                raise ValueError
        
        if self.render_option.render_sky:
            self.sky_decoder = SkyDecoder(**kwargs['sky_decoder_kwargs'])
            self.nerf_renderer.ray_marcher['sky'] = skyRayMarcher(decoder = self.sky_decoder)
        # self.ray_marcher = MipRayMarcher2()

        if self.render_option.use_neural_renderer:
            neural_render_kwargs = kwargs['neural_render_kwargs']
            neural_render_kwargs.in_res = cfg.img_size_raw[0] * cfg.ratio /  cfg.super_resolution
            neural_render_kwargs.out_res = cfg.img_size_raw[0] * cfg.ratio
            self.neural_renderer = neural_renderer_dict[neural_renderer_type](**neural_render_kwargs)
            self.feature_dim = feature_dim
        else:
            self.feature_dim = 0
            self.neural_renderer = None


        # TODO init render option
#-------------------Render Option--------------------------------------------
        self.render_option.n_samples_obj = n_samples_obj
        self.render_option.n_samples_stuff = n_samples_stuff
        self.render_option.n_vox_intersection = n_vox_intersection
        self.render_option.sample_before_intersection = sample_before_intersection
        self.render_option.semantic_dim = semantic_dim
        self.render_option.z_dim_global = z_dim_global
        self.render_option.z_dim_obj = z_dim_obj
        self.render_option.z_dim_road = z_dim_road
        self.render_option.z_dim_stuff = z_dim_stuff

        self.render_option.render_obj = cfg.render_obj
        self.render_option.render_stuff = cfg.render_stuff
        self.render_option.render_sky = cfg.render_sky
        self.render_option.use_patch_discriminator = cfg.use_patch_discriminator
        self.render_option.ins_semantic_list = cfg.valid_object
        self.render_option.stuff_representation = self.stuff_representation_type
        self.render_option.z_trainable = z_trainable
        self.render_option.use_z_map = use_z_map
        self.render_option.use_occupancy_mask = cfg.use_occupancy_mask
        self.render_option.use_neural_renderer = use_neural_renderer
        self.render_option.ray_voxel_sampling = ray_voxel_sampling
        self.render_option.n_samples_obj_render = n_samples_obj_render
        self.render_option.n_samples_stuff_render = n_samples_stuff_render 
        self.render_option.n_vox_intersection_render = n_vox_intersection_render
        self.render_option.voxel_range  = self.voxel_range 

        self.render_option.use_obj_mapper = use_obj_mapper
        self.render_option.patch_size = cfg.patch_size
        self.render_option.sr_multiple = cfg.super_resolution
        self.render_option.use_scale_condition = use_scale_condition
        self.render_option.use_patch_mask = cfg.use_patch_occupancy_mask
        self.render_option.is_kitti360 = cfg.is_kitti360

        self.render_option.z_far = cfg.z_far
        self.render_option.z_far_render = cfg.z_far_render

        self.render_option.is_debug = cfg.is_debug
#-------------------------------------------------------------------------------
        
        self.n_samples_obj = n_samples_obj
        # self.n_samples_road = n_samples_road
        self.n_samples_stuff = n_samples_stuff
        self.n_vox_intersection = n_vox_intersection
        self.sample_before_intersection = sample_before_intersection

        self.semantic_dim = semantic_dim
        self.z_dim_global = z_dim_global
        self.z_dim_obj = z_dim_obj
        self.z_dim_road = z_dim_road
        self.z_dim_stuff = z_dim_stuff
        

        self.get_feature_embedding()

    def get_feature_embedding(self):
        #road_semantic_list = ['ground', 'road', 'sidewalk', 'parking','rail track', 'terrain']        
        self.obj_semantic_embedding = nn.Embedding(10, int(self.semantic_dim))
        self.road_semantic_embedding = nn.Embedding(10, int(self.semantic_dim))
        self.stuff_semantic_embedding = nn.Embedding(42, int(self.semantic_dim))

        if self.render_option.z_trainable:
            self.global_embedding = nn.Embedding(100000, int(self.z_dim_global))
            self.bbox_embedding = nn.Embedding(50000, int(self.z_dim_obj))

    # def get_global_code(self, global_idx, tmp=1.):
    #     z_dim_global  = self.render_option.z_dim_global
    #     batch_size = global_idx.shape[0]
    #     def sample_z(x): return self.sample_z(x, tmp=tmp)
    #     # Sample z global(for each frame)
    #     if self.render_option.z_trainable:
    #         z_global = self.global_embedding(global_idx).reshape((batch_size, z_dim_global))
    #     else:
    #         z_global = sample_z((batch_size, z_dim_global))

    #     return z_global

    def get_instance_code(self, bbox_idx, tmp=1.):
        z_dim_obj = self.z_dim_obj
        assert bbox_idx != None
        batch_size =  bbox_idx.shape[0]
        n_boxes = bbox_idx.shape[1]
        def sample_z(x): return self.sample_z(x, tmp=tmp)
        if self.render_option.z_trainable:
            z_bbox = self.bbox_embedding(bbox_idx)
        else:
            z_bbox = sample_z((batch_size, n_boxes, z_dim_obj))
        # latent_codes['z_bbox'] = z_bbox
        return z_bbox

    def get_features(self, batch, latent_codes):

        features = {}

        z_global = latent_codes['z_global']
        if self.render_option.render_obj:
            z_bbox = latent_codes['z_bbox']
            if self.render_option.use_scale_condition and self.render_option.use_patch_discriminator:
                c_bbox = batch['bbox_poses'][...,-3:] #scale_condittion
                features['c_bbox'] = c_bbox
            else:
                features['c_bbox'] = z_bbox[...,-3:]

        if self.render_option.render_obj:
            bbox_semantic = batch['bbox_semantic']
            bbox_semantic_ = bbox_semantic.clone()
            for k in bbox_semantic_translate:
                bbox_semantic_[bbox_semantic == k] = bbox_semantic_translate[k]
    
            bbox_semantic_feature = self.obj_semantic_embedding(bbox_semantic_)
            features['bbox_semantic'] = bbox_semantic_ 
            features['bbox_semantic_feature'] = bbox_semantic_feature

        if self.render_option.render_stuff:
            stuff_semantic_grid = batch['stuff_semantic_grid']
            stuff_loc_grid = batch['stuff_loc_grid']
            one_hot_stuff_semantic_grid = F.one_hot(stuff_semantic_grid.to(torch.int64), num_classes = self.semantic_nc).permute(0,4,1,2,3).to(torch.float32)

            if self.stuff_generator_type == 'VolumeGenerator':
                stuff_feature_grid = self.z_grid_generator(input = one_hot_stuff_semantic_grid, z = z_global)

                features['stuff_feature_grid'] = stuff_feature_grid.permute((0,2,3,4,1))

            elif self.stuff_generator_type == 'Triplane':
                triplane_feature = self.z_grid_generator(z = z_global, c = one_hot_stuff_semantic_grid)['image']
                features['stuff_feature_grid'] = triplane_feature

        features['stuff_semantic_grid'] = stuff_semantic_grid.unsqueeze(-1)
        features['stuff_loc_grid'] = stuff_loc_grid
        return features

    def volume_render_image(self, batch, latent_codes ,mode='train'):
        camera_mat, camera_pose, rays = batch['camera_mat'], batch['camera_pose'], batch['rays']

        camera_extrinsic_global = torch.inverse(camera_pose)
        ''' camera_extrinsic = 
        [[ux vx nx tx]
        [uy vy ny ty]
        [uz vz nz tz]
        [0 0 0 1]]
        U :right; V:up; N:look dir
        '''
        up_camera = camera_extrinsic_global[:,1,:3] 
        dir_camera = camera_extrinsic_global[:,2,:3]
        render_stuff = self.render_option.render_stuff
        render_sky = self.render_option.render_sky
        render_obj = self.render_option.render_obj

        self.render_option.batch_size = rays.shape[0]
        self.render_option.H_ray = rays.shape[1]
        self.render_option.W_ray = rays.shape[2]

        batch_size, n_pixels = rays.shape[0], rays.shape[1]
        H, W = rays.shape[1], rays.shape[2]
        n_pixels = H * W

        rays_dir, rays_pixel, rays_origin = rays[...,6:9], rays[...,3:6] , rays[...,0:3]
        rays_dir, rays_pixel, rays_origin = rays_dir.reshape((batch_size, n_pixels, 3)), rays_pixel.reshape((batch_size, n_pixels, 3)), rays_origin.reshape((batch_size, n_pixels, 3))


        dir_camera = dir_camera[:,None,:].repeat(1,n_pixels,1)
        rays_dir[torch.abs(rays_dir) < 1e-3] = 1e-3
        sigma, normal, feat, scope, depth = {}, {}, {}, {}, {}
        semantic = {}

        #?--------------Objecr Ray Marching---------------------------------------------
        if render_obj :
            bbox_trs, bbox_semantic = batch['bbox_trs'], batch['bbox_semantic']
            samples_obj = self.nerf_renderer.ray_marcher['obj'](latent_codes = latent_codes, rays = rays, bbox_semantic = bbox_semantic, bbox_trs = bbox_trs, render_option = self.render_option)
            if isinstance(samples_obj, dict):
                sigma['obj'] = samples_obj['sigma']
                feat['obj'] = samples_obj['feat']
                scope['obj'] = samples_obj['scope']
                depth['obj'] = samples_obj['depth']
                semantic['obj'] = samples_obj['semantic']
        #?---------------Stuff Ray Matching--------------------------------------------
        if render_stuff:
            samples_stuff = self.nerf_renderer.ray_marcher['stuff'](rays = rays, latent_codes = latent_codes, c2w = camera_pose, K = camera_mat, semantic_embedding = self.stuff_semantic_embedding , render_option = self.render_option)
            sigma['stuff'] = samples_stuff['sigma']
            feat['stuff'] = samples_stuff['feat']
            scope['stuff'] = samples_stuff['scope']
            depth['stuff'] = samples_stuff['depth']
            semantic['stuff'] = samples_stuff['semantic']
        #?---------------SKY Ray Matching--------------------------------------------
        if render_sky:
            sample_sky = self.nerf_renderer.ray_marcher['sky'](z = latent_codes['z_global'], rays = rays, render_option = self.render_option)
            sigma['sky'] = sample_sky['sigma']
            feat['sky'] = sample_sky['feat']
            scope['sky'] = sample_sky['scope']
            depth['sky'] = sample_sky['depth']
            semantic['sky'] = sample_sky['semantic']
        #?---------------Conduct Composition Operation-----------------------------
        # if True:
        all_samples = OrderedDict({
            'sigma' : sigma,
            'feat' : feat,
            'depth' : depth,
            'semantic' : semantic})
        composite_samples = self.nerf_renderer.composite_multi_fields(all_samples, mode = mode)
        sigma_all, feat_all, depth_all, semantic_all= composite_samples['sigma'],composite_samples['feat'],composite_samples['depth'],composite_samples['semantic']

        #?---------------Volume Rendering-------------------------------------------
        all_samples = {
            'depth' : depth_all, 
            'sigma' : sigma_all, 
            'feat' : feat_all, 
            'semantic' : semantic_all,
            'rays_dir': rays_dir
        }
        render_results = self.nerf_renderer(all_samples, is_compositional = False, render_option = self.render_option)
        depth_map, feat_map, alpha_map = render_results['depth_map'], render_results['feat_map'],  render_results['alpha_map']
        weights = render_results['weights']

        output = {}
        output['depth'] = depth_map
        output['feat_raw'] = feat_map.contiguous()
        output['rgb_raw'] = feat_map[:,:3].contiguous()
        output['alpha'] = alpha_map

        for i in scope.keys():
            if i == 'obj':
                maxbbox = self.render_option.maxbbox
                bbox_masks = scope[i].reshape(batch_size, maxbbox, H, W)
                bbox_alphas = []
                for i in range(maxbbox):
                    t = weights.clone()
                    a = semantic_all > (i + 1) * 1000
                    b = semantic_all < (i + 2) * 1000 
                    t[a & b == False] = 0
                    acc_map = torch.sum(t, dim=-1, keepdim=True)
                    acc_map = acc_map.permute(0, 2, 1).reshape(batch_size, -1, H, W)
                    bbox_alphas.append(acc_map)
                bbox_alphas = torch.stack(bbox_alphas)
                if cfg.is_debug:
                    save_tensor_img(bbox_alphas,'tmp','patch_alpha.jpg', valid_frame_num=self.render_option.maxbbox)
                    save_tensor_img(bbox_masks.permute(1,0,2,3).unsqueeze(2),'tmp','patch_mask.jpg', valid_frame_num=self.render_option.maxbbox)
                
                output['bbox_alphas'] = bbox_alphas.permute(1,0,2,3,4).reshape(-1,1,H,W)
                output['bbox_masks'] = bbox_masks
                bbox_mask = torch.sum(bbox_masks, dim = 1, keepdim=True)
                bbox_mask[bbox_mask > 0] = 1
                output['bbox_mask'] =  bbox_mask

        if cfg.is_debug:
            save_tensor_img(torchvision.utils.make_grid(output['depth']), 'tmp', 'depth.jpg', 'depth')
            
        return output
    
    def crop_obj_patches(self, rgb_image, bbox_masks, bbox_semantic, bbox_alphas, use_mask = True, cat_mask = True):
        '''
        rgb_image: [B, 3, H, W]
        bbox_masks: [B, N, H, W]
        bbox_semantic: [B, N]
        bbox_alphas: [B * N, 1, H, W]
        '''
        batch_size, bbox_num = bbox_masks.shape[0], bbox_masks.shape[1]
        H, W = bbox_masks.shape[2], bbox_masks.shape[3]
        bbox_alphas = bbox_alphas.reshape(batch_size, bbox_num, 1, H, W)
        batch_bbox_patches = []
        # rgba_image = torch.cat((rgb_image, bbox_alpha), dim=1)
        for b in  range(batch_size):
            bbox_patches = []
            for i in range(bbox_num):
                if bbox_semantic[b,i] != -1 and bbox_masks[b,i].sum() > 0:
                    mask, alpha = bbox_masks[b,i], bbox_alphas[b,i,0]
                    bbox_patch = rgb_image[b:b+1]
                    # alpha[:,:] = 0
                    # if self.is_kitti360:
                    uv_min = torch.argwhere(mask== 1).min(dim=0)[0]
                    uv_max = torch.argwhere(mask == 1).max(dim=0)[0]
                    if ((uv_max[0] - uv_min[0]) * (uv_max[1] - uv_min[1])) < 25:
                        bbox_semantic[b,i] = -1
                        print('a')
                        bbox_patch = torch.zeros((1,3 + cat_mask,self.render_option.patch_size, self.render_option.patch_size)).to(rgb_image.device)
                    else:
                        if use_mask:
                            bbox_patch = bbox_patch * bbox_alphas[b:b+1,i]
                        if cat_mask:
                            al = (bbox_alphas[b:b+1,i] - 0.5 ) * 2
                            # al[al >= 0] = 1
                            # al[al < 0] = -1
                            bbox_patch = torch.cat((bbox_patch, al), dim=1)

                        bbox_patch = bbox_patch[...,uv_min[0]:uv_max[0],uv_min[1]:uv_max[1]]
                        h_patch, w_patch=  bbox_patch.shape[-2], bbox_patch.shape[-1]
                        pad_num = int(abs(w_patch - h_patch)/2)
                        if h_patch <= w_patch:
                            pad = nn.ConstantPad2d((0,0,pad_num,pad_num), -1)
                        else: 
                            pad = nn.ConstantPad2d((pad_num,pad_num,0,0), -1)
                        bbox_patch = pad(bbox_patch)
                        bbox_patch = F.interpolate(bbox_patch, size=(self.render_option.patch_size, self.render_option.patch_size))
                    # save_tensor_img(bbox_patch, 'tmp', 'aaa.jpg')
                else:
                    bbox_patch = torch.zeros((1,3 + cat_mask,self.render_option.patch_size, self.render_option.patch_size)).to(rgb_image.device)
                bbox_patches.append(bbox_patch)

            batch_bbox_patches.append(torch.cat(bbox_patches, dim=0))
        batch_bbox_patches = torch.stack(batch_bbox_patches)
        if cfg.is_debug:
            save_tensor_img(batch_bbox_patches.permute(1,0,2,3,4)[:,:,0:3], 'tmp', 'bbox_patches.jpg',valid_frame_num = bbox_num)
            save_tensor_img(batch_bbox_patches.permute(1,0,2,3,4)[:,:,-1:], 'tmp', 'bbox_patches_a.jpg',valid_frame_num = bbox_num)
        return batch_bbox_patches[:,:,0:3]
   
    def select_date(self, raw_batch, data_type):

        batch = {}
        batch['camera_mat'] = raw_batch['camera_mat']
        if data_type == 'gen':
            batch['frame_id'] =  raw_batch['frame_id']
            if self.render_option.use_occupancy_mask:
                batch['occupancy_mask'] = raw_batch['occupancy_mask']
            batch['camera_pose'] =  raw_batch['world_mat']
            batch['rays'] =  raw_batch['rays']
            batch['rgb_gt'] = raw_batch['rgb']
        elif data_type == 'dis':
            batch['frame_id'] =  raw_batch['frame_id_fake']
            if self.render_option.use_occupancy_mask:
                batch['occupancy_mask'] = raw_batch['occupancy_mask_fake']
            batch['camera_pose'] =  raw_batch['world_mat_fake']
            batch['rays'] =  raw_batch['rays_fake']
            batch['rgb_gt'] = raw_batch['rgb_fake']
        else:
            raise KeyError

        if self.render_option.render_obj:
            if data_type == 'gen':
                batch['bbox_idx'], batch['bbox_semantic'], batch['bbox_trs'] = raw_batch['bbox_id'], raw_batch['bbox_semantic'], raw_batch['bbox_tr']
                if self.render_option.use_patch_discriminator:
                    batch['bbox_poses']  = raw_batch['bbox_pose']
            elif data_type == 'dis':
                batch['bbox_idx'], batch['bbox_semantic'], batch['bbox_trs'] = raw_batch['bbox_id_fake'], raw_batch['bbox_semantic_fake'], raw_batch['bbox_tr_fake']
                
                if self.render_option.use_patch_discriminator:
                    batch['bbox_poses'] = raw_batch['bbox_pose_fake']
            else:
                raise KeyError

        if self.render_option.render_stuff:
            if data_type == 'gen':
                batch['stuff_semantic_grid'] = raw_batch['stuff_semantic_grid']
                batch['stuff_loc_grid'] = raw_batch['stuff_loc_grid']
            elif data_type == 'dis':
                batch['stuff_semantic_grid'] = raw_batch['stuff_semantic_grid_fake']
                batch['stuff_loc_grid']= raw_batch['stuff_loc_grid_fake']
            else:
                raise KeyError
            
        #get_latent_code
        if 'z_global' in raw_batch.keys():
            batch['z_global'] = raw_batch['z_global']
        if 'z_bbox' in raw_batch.keys():
            batch['z_bbox'] = raw_batch['z_bbox']

        return batch

    def forward(self, raw_batch, data_type='gen', mode='train'):

        self.render_option.mode = mode
        self.render_option.data_type = data_type

        batch, latent_codes = {}, {}
        batch = self.select_date(raw_batch, data_type=data_type) 

        # Get latent codes
        if 'z_global' in batch.keys():
            latent_codes['z_global'] = batch['z_global']
        # else:
        #     latent_codes['z_global'] = self.get_global_code(batch['frame_id'])
        if 'z_bbox' in batch.keys():
            latent_codes['z_bbox'] = batch['z_bbox']
        elif self.render_option.render_obj:
            latent_codes['z_bbox'] = self.get_instance_code(batch['bbox_idx'])
             
        # Get feature vecotor(semanti feature, stuff local feature grid and bbox triplane)
        t = time.time()
        features = self.get_features(batch=batch, latent_codes=latent_codes)
        latent_codes.update(features)
        # t_gen = time.time() - t
        # t = time.time()
        # Volume Rendering
        output = {}
        volume_render_output = self.volume_render_image(
            batch = batch, latent_codes=latent_codes,mode=mode)
            # road_rays_intersection = rays_intersection_loc, road_rays_semantic = rays_semantic,)
        t_vr = time.time() - t
        t = time.time()

        output = volume_render_output
        output['rgb_gt'] = batch['rgb_gt']
        if self.neural_renderer is not None:
            s = volume_render_output['feat_raw'].detach().cpu().numpy()
            rgb_refine = self.neural_renderer(x = volume_render_output['feat_raw'],
                                              rgb = volume_render_output['rgb_raw'], 
                                              z = latent_codes['z_global'])

            if self.render_option.sr_multiple > 1:
                for k in output:
                    if re.search('bbox', k) != None or re.search('alpha', k) != None or re.search('depth', k) != None:
                        H, W = output[k].shape[-2] * self.render_option.sr_multiple, output[k].shape[-1] * self.render_option.sr_multiple
                        output[k] = F.interpolate(output[k],(H, W),mode = 'bilinear' if re.search('depth', k) != None else 'nearest')

            output['rgb'] = rgb_refine 
        else:
            output['rgb'] = output['rgb_raw']
        t_sr = time.time() - t
        if False:
            print('time grid: %.3f,vr: %.3f,sr: %.3f'%(t_gen,t_vr,t_sr))
        if 'bbox_masks' in output:
            bbox_masks, bbox_semantic, bbox_alphas = output['bbox_masks'], batch['bbox_semantic'], output['bbox_alphas']
            n_bbox = bbox_masks.shape[1]
            bbox_patches = self.crop_obj_patches(
                    rgb_image= output['rgb'],
                    bbox_masks = bbox_masks,
                    bbox_semantic = bbox_semantic,
                    bbox_alphas = bbox_alphas,
                    use_mask= self.render_option.use_patch_mask)
            
            output['bbox_patches'] = bbox_patches[:,:n_bbox]
            output['bbox_semantic'] = bbox_semantic[:,:n_bbox]
            if self.render_option.use_patch_discriminator:
                output['bbox_poses'] = batch['bbox_poses'][:,:n_bbox]


        output['frame_id'] = batch['frame_id']
        output['camera_intrinsic'] = batch['camera_mat']
        output['camera_pose'] = batch['camera_pose']
        if cfg.is_debug:
            save_tensor_img(torchvision.utils.make_grid(output['rgb']), 'tmp', 'rgb.jpg')
            
            if self.render_option.render_stuff:
                output['semantic_gird']= batch['stuff_semantic_grid']
                output['feature_grid'] = latent_codes['stuff_feature_grid']
            if self.render_option.render_obj:
                output['bbox_tr'] = batch['bbox_trs']

        if self.render_option.use_occupancy_mask: 
            occupancy_mask = batch['occupancy_mask']
            output['occupancy_mask'] = occupancy_mask

        
        return output
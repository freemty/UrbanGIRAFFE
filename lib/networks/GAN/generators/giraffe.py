
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
import numpy as np

import re
from tools.kitti360Scripts.helpers.labels import name2label
from lib.config import cfg
from lib.networks.volumetric_rendering.decoders import *
from lib.utils.img_utils import save_tensor_img
from lib.networks.volumetric_rendering import ray_voxel_intersection_sampling, sample_from_3dgrid
from .utils import *
from lib.networks.reference.stylegan2 import MappingNetwork

# obj_semantic_name = ['cars']
bbox_semantic_translate = {-1:0,11:1,17:2,19:3,20:4,26:5,27:6,28:7,34:8,36:9}
# road_semantic_translate = {-1:0,6:1,7:2,8:3,9:4,10:5,22:6}
# # stuff_semantic_translate = {-1:0,6:1,7:2,8:3,9:4,10:5,12:6,13:7,21:8,22:9}


decoder_dict = {
    'giraffe_ins': giraffeInsDecoder,
    'giraffe_bg': giraffeDecoder,
    'eg3d_ins': eg3dDecoder,
    # 'style': styleDecoder,
    # 'local_feature': localfeatureDecoder,
    'sky': skyDecoder,
    'stuff': stuffDecoder
}

 

z_grid_generator_dict = {
    'SPADE3D': SPADEGenerator3D,
}

neural_renderer_dict = {
    # 'giraffeNR': giraffeNR,
    'gancraftNR': gancraftNR,
}


# # AABB
def get_near_far(bounds, ray_o, ray_d, return_mask = True):
    """calculate intersections with 3d bounding box"""
    # bounds = bounds + np.array([-0.01, 0.01], dtype=np.float32)[:, None]
    nominator = bounds[None] - ray_o[:, None]
    # calculate the step of intersections at six planes of the 3d bounding box
    ray_d[np.abs(ray_d) < 1e-3] = 1e-3
    d_intersect = (nominator / ray_d[:, None]).reshape(-1, 6)
    # calculate the six interections
    p_intersect = d_intersect[..., None] * ray_d[:, None] + ray_o[:, None]
    # calculate the intersections located at the 3d bounding box
    min_x, min_y, min_z, max_x, max_y, max_z = bounds.ravel()
    eps = 1e-7
    p_mask_at_box = (p_intersect[..., 0] >= (min_x - eps)) * \
                    (p_intersect[..., 0] <= (max_x + eps)) * \
                    (p_intersect[..., 1] >= (min_y - eps)) * \
                    (p_intersect[..., 1] <= (max_y + eps)) * \
                    (p_intersect[..., 2] >= (min_z - eps)) * \
                    (p_intersect[..., 2] <= (max_z + eps))
    # obtain the intersections of rays which intersect exactly twice
    mask_at_box = p_mask_at_box.sum(-1) == 2

    if return_mask:
        return mask_at_box
    else:
        if mask_at_box.sum()>0:
            return True 
        else:
            return False

class Generator(nn.Module):
    def __init__(self,
        obj_decoder_type = 'giraffe_ins',
        bg_decoder_type = 'giraffe_bg',
        neural_renderer_type = 'gancraftNR',
        return_depth_map = True,
        use_neural_renderer = False,
        use_z_map = True,
        z_trainable = False,
        use_scale_condition = False,
        z_dim_global = 64,
        feature_dim = 32,
        semantic_dim = 16,
        z_dim_obj = 80,
        depth_range = 24,
        n_samples_obj = 24,
        n_samples_bg = 12,
        semantic_aware_obj = False,
        **kwargs):
 

        super(Generator, self).__init__()
        #self.ray_sampler = RaySampler()
        if cfg.use_cuda: 
            if cfg.distributed:
                self.device = torch.device('cuda:{}'.format(cfg.local_rank))
            else:
                self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.render_obj = cfg.render_obj
        self.semantic_aware_obj = semantic_aware_obj
        self.ins_semantic_list = cfg.valid_object

        self.render_stuff = cfg.render_stuff
        self.render_sky = cfg.render_sky
        self.train_sky = cfg.train_sky
        self.z_trainable = z_trainable
        self.use_z_map = use_z_map
        self.use_occupancy_mask = cfg.use_occupancy_mask
        self.use_neural_renderer = use_neural_renderer
        self.return_depth_map = return_depth_map
        # self. = cfg.
        
        self.sr_multiple = cfg.super_resolution
        self.use_scale_condition = use_scale_condition

        self.depth_range = depth_range
        # n_samples_bg = 32
        self.x_max, self.x_min = 32, -32
        self.y_max, self.y_min = 3, -14
        self.z_max, self.z_min = 64, 0
        

        # Feature dim
        kwargs['obj_decoder_kwargs'].out_channels_feat = kwargs['neural_render_kwargs'].in_channels = feature_dim

        # z_dim
        kwargs['neural_render_kwargs'].style_channels = kwargs['bg_decoder_kwargs'].z_dim = z_dim_global
        kwargs['obj_decoder_kwargs'].z_dim = z_dim_obj

        # super_resolution
        kwargs['neural_render_kwargs'].super_res = self.sr_multiple

        #! GIRAFFE only
        self.render_obj = cfg.render_obj
        self.render_bg = cfg.render_bg
        
        if self.render_bg:
            print('Running GIRAFFE base line')
            self.render_sky  = self.render_stuff = False
            # assert (self.render_sky or self.render_stuff) == False
            self.nerf_bg =  giraffeDecoder(**kwargs['bg_decoder_kwargs'])

        # urbangiraffe three submodule
        if self.render_obj:
            if not self.use_scale_condition:
                kwargs['obj_decoder_kwargs']['c_dim'] = 0
            self.nerf_obj = decoder_dict[obj_decoder_type](
                **kwargs['obj_decoder_kwargs'])


        if self.use_neural_renderer:
            self.neural_renderer = neural_renderer_dict[neural_renderer_type](**kwargs['neural_render_kwargs'])
            self.feature_dim = feature_dim
        else:
            self.feature_dim = 0
            self.neural_renderer = None

        self.n_samples_obj = n_samples_obj
        self.n_samples_bg = n_samples_bg

        self.semantic_dim = semantic_dim
        self.z_dim_global = z_dim_global
        self.z_dim_obj = z_dim_obj

        self.use_max_composition = False

        self.get_feature_embedding()

    def get_depth_range(self, tr):
        
        vertices = [[0.5,0.5,0.5,1],[-0.5,-0.5,-0.5,1],[0.5,-0.5,-0.5,1],[-0.5,0.5,-0.5,1],\
        [-0.5,-0.5,0.5,1],[0.5,0.5,-0.5,1],[0.5,-0.5,0.5,1],[-0.5,0.5,0.5,1]]
        v_0 = torch.tensor(vertices).to(self.device)
        v = ( tr @ v_0.T).permute(0,1,3,2)
        d_v = v[...,2]
        d_max, _ = torch.max(d_v, dim = -1, keepdim=True)
        d_min, _ = torch.min(d_v, dim = -1, keepdim=True)


        return d_min.clamp(1e-3,80.), d_max.clamp(1e-3,80.)

    def get_feature_embedding(self):
        #road_semantic_list = ['ground', 'road', 'sidewalk', 'parking','rail track', 'terrain']        
        self.obj_semantic_embedding = nn.Embedding(10, int(self.semantic_dim))
        self.road_semantic_embedding = nn.Embedding(10, int(self.semantic_dim))
        self.stuff_semantic_embedding = nn.Embedding(42, int(self.semantic_dim))


        if self.z_trainable:
            self.global_embedding = nn.Embedding(60000, int(self.z_dim_global))
            self.bbox_embedding = nn.Embedding(50000, int(self.z_dim_obj))

    def get_global_code(self, global_idx, tmp=1.):
        z_dim_global  = self.z_dim_global
        batch_size = global_idx.shape[0]
        def sample_z(x): return self.sample_z(x, tmp=tmp)
        # Sample z global(for each frame)
        if self.z_trainable:
            z_global = self.global_embedding(global_idx).reshape((batch_size, z_dim_global))
        else:
            z_global = sample_z((batch_size, z_dim_global))

        return z_global

    def get_instance_code(self, bbox_idx, tmp=1.):
        z_dim_obj = self.z_dim_obj
        assert bbox_idx != None
        batch_size =  bbox_idx.shape[0]
        n_boxes = bbox_idx.shape[1]
        def sample_z(x): return self.sample_z(x, tmp=tmp)
        if self.z_trainable:
            z_bbox = self.bbox_embedding(bbox_idx)
        else:
            z_bbox = sample_z((batch_size, n_boxes, z_dim_obj))
        # latent_codes['z_bbox'] = z_bbox
        return z_bbox



    def get_features(self, batch, latent_codes):

        features = {}

        z_global = latent_codes['z_global']
        if self.render_obj:
            latent_codes['c_bbox'] = latent_codes['z_bbox'][...,-3:] #scale_condittion

        if self.render_obj:
            bbox_semantic = batch['bbox_semantic']
            batch_size, n_boxes = bbox_semantic.shape[0], bbox_semantic.shape[1]
            bbox_semantic_ = bbox_semantic.clone()
            for k in bbox_semantic_translate:
                bbox_semantic_[bbox_semantic == k] = bbox_semantic_translate[k]
    
            bbox_semantic_feature = self.obj_semantic_embedding(bbox_semantic_)
            features['bbox_semantic'] = bbox_semantic_ 
            features['bbox_semantic_feature'] = bbox_semantic_feature

        return features

    def sample_z(self, size, to_device=True, tmp=1.):
        z = torch.randn(*size) * tmp
        if to_device:
            z = z.to(self.device)
        return z
  
    def add_noise_to_interval(self, di):
        di_mid = .5 * (di[..., 1:] + di[..., :-1])
        di_high = torch.cat([di_mid, di[..., -1:]], dim=-1)
        di_low = torch.cat([di[..., :1], di_mid], dim=-1)
        noise = torch.rand_like(di_low)
        ti = di_low + (di_high - di_low) * noise
        return ti


    def get_evaluation_points(self, rays, origin_camera, bbox_trs, i, mode = 'train'):
        ''' Get the evaluation points (camera cordinate)

        Args:
            pixels(tensor 1*1*1): Positions of all pixels on camera plane.
            camera(tensor 1*1*1): Posttion of camera
            di(): All points on all camera rays
            bbox_tes: {s, t, R} of the boundingbox
            i: the index of the input boundng box(the last one is background)
        Return:

        '''
        batch_size = rays.shape[0]
        n_samples = self.n_samples_obj
        n_pixels = rays.shape[1]

        # ray_i_z1_ = rays_z1.detach().cpu().numpy()
        # p_i = origin_camera + (di - origin_camera[...,2,None]) * ray_i / ray_i[...,2]
        rays_d = torch.cat((rays, torch.zeros_like(rays)[...,0:1]), dim=-1)
        rays_o = torch.cat((origin_camera, torch.ones_like(rays)[...,0:1]), dim=-1)

        # q = torch.bmm(torch.inverse(bbox_trs[:,i,...]), rays_o.permute(0,2,1)).permute(0,2,1)[...,3].detach().cpu().numpy()
        rays_o_local = torch.bmm(torch.inverse(bbox_trs[:,i,...]), rays_o.permute(0,2,1)).permute(0,2,1)[...,:3].reshape(-1,3)
        rays_d_local = torch.bmm(torch.inverse(bbox_trs[:,i,...]), rays_d.permute(0,2,1)).permute(0,2,1)[...,:3].reshape(-1,3)

        ## batchify ray-AABB intersetction
        bounds = torch.tensor([[-0.5,-0.5,-0.5],[0.5,0.5,0.5]], device=rays_o.device)
        nominator = bounds[None] - rays_o_local[:, None]
        # calculate the step of intersections at six planes of the 3d bounding box
        norm_d = torch.linalg.norm(rays_d_local, dim=-1, keepdims=True)
        rays_dir = rays_d_local / norm_d
        rays_dir[torch.abs(rays_dir) < 1e-3] = 1e-3
        # d_intersect = (nominator / rays_d_local[:, None]).reshape(-1, 6)
        # # calculate the six interections
        # p_intersect = d_intersect[..., None] * rays_d_local[:, None] + 
        # viewdir[(viewdir > -1e-5) & (viewdir < 1e-10)] = -1e-5
        tmin = (bounds[0:1] - rays_o_local) / rays_dir
        tmax = (bounds[1:2] - rays_o_local) / rays_dir
        t1 = torch.minimum(tmin, tmax)
        t2 = torch.maximum(tmin, tmax)
        near = torch.max(t1, dim=-1)[0]
        far = torch.min(t2, dim=-1)[0]
        mask_at_box = (near <  far) & (near > 0)
        # mask_at_box = (near < far)  & (near > 0)
        near = near[mask_at_box] / norm_d[mask_at_box, 0]
        far = far[mask_at_box] / norm_d[mask_at_box, 0]


        valid_ray_num = torch.sum(mask_at_box.reshape(batch_size,-1), dim=-1)
        t_i_valid = near[:,None] + torch.linspace(0,1,n_samples, device=near.device)[None,:] * (far - near)[:,None]

        if mode == 'train':
            t_i_valid = self.add_noise_to_interval(t_i_valid)
        p_i_valid = rays_o_local[mask_at_box,None] + rays_d_local[mask_at_box,None,:] * t_i_valid[...,None]

        ray_i_valid = rays_d_local[mask_at_box,None,:].repeat(1,n_samples,1)

        p_mask = mask_at_box.reshape(batch_size,-1)

        return p_i_valid, t_i_valid, ray_i_valid, p_mask

    def composite_function(self, sigma, feat):
        n_bboxes = sigma.shape[0]
        if n_bboxes > 1:
            if self.use_max_composition:
                bs, rs, ns = sigma.shape[1:]
                sigma_sum, ind = torch.max(sigma, dim=0)
                feat_weighted = feat[ind, torch.arange(bs).reshape(-1, 1, 1),
                                     torch.arange(rs).reshape(
                                         1, -1, 1), torch.arange(ns).reshape(
                                             1, 1, -1)]
            else:
                denom_sigma = torch.sum(sigma, dim=0, keepdim=True)
                denom_sigma[denom_sigma == 0] = 1e-4
                w_sigma = sigma / denom_sigma
                sigma_sum = torch.sum(sigma, dim=0)
                feat_weighted = (feat * w_sigma.unsqueeze(-1)).sum(0)
        else:
            sigma_sum = sigma.squeeze(0)
            feat_weighted = feat.squeeze(0)
        return sigma_sum, feat_weighted

    def calc_volume_weights(self, z_vals, ray_vector, sigma, last_dist=1e10):
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.ones_like(
            z_vals[..., :1]) * last_dist], dim=-1)
        dists = dists * torch.norm(ray_vector, dim=-1, keepdim=True)
        alpha = 1.-torch.exp(-F.relu(sigma)*dists)

        weights = alpha * \
            torch.cumprod(torch.cat([
                torch.ones_like(alpha[:, :, :1]),
                (1. - alpha + 1e-10), ], dim=-1), dim=-1)[..., :-1]
        return weights

    def batchify_rays_render(self, p, r, z, c = None, s = None, mode = 'train', decoder = 'obj', **kwargs):
        #batchify_rays render
        ray_as_input = False
        if len(p.shape) == 3:
            ray_as_input = True
            ray_num, n_samples = p.shape[0],p.shape[1]
            p = p.view(ray_num * n_samples, -1)
            z = z.view(ray_num * n_samples, -1)
            r = r.view(ray_num * n_samples, -1)
            if s != None:
                s = s.view(ray_num * n_samples, -1)
            if c != None:
                c = c.view(ray_num * n_samples, -1)

        feat, sigma = [], []
        chunk = 300000
        if decoder == 'obj':
            nerf = self.nerf_obj
        elif decoder ==  'bg':
            nerf = self.nerf_bg
        else: 
            raise KeyboardInterrupt
        for i in range(0, p.shape[0], chunk):
            feat_chunk, sigma_chunk =nerf(pts = p[i:i+chunk],ray_d= r[i:i+chunk], z = z[i:i+chunk], c = None if decoder ==  'bg' else c[i:i+chunk], semantic = None if decoder ==  'bg' else s[i:i+chunk])
            feat.append(feat_chunk[:,-3 - self.feature_dim:])
            sigma.append(sigma_chunk)
            torch.cuda.empty_cache()
        feat = torch.cat(feat, dim = 0)
        sigma = torch.cat(sigma, dim = 0)
        if mode == 'train':
            # As done in NeRF, add noise during training
            sigma += torch.randn_like(sigma)
        # sigma = F.relu(sigma) #
        if ray_as_input:
            sigma = sigma.reshape(ray_num, n_samples, -1)
            feat = feat.reshape(ray_num, n_samples, -1)

        return sigma, feat

    def volume_render_image(self, batch, latent_codes ,mode='train'):
        camera_mat, camera_pose, rays = batch['camera_mat'], batch['camera_pose'],batch['rays']

        # road_rays_intersection = None, road_rays_semantic = None
        # render_road = self.render_road
        render_stuff = self.render_stuff
        render_sky = self.render_sky
        if self.render_obj:

            bbox_trs, bbox_semantic = batch['bbox_trs'], batch['bbox_semantic'].clone()
            
            bbox_mask = torch.where(bbox_semantic == -1, torch.zeros_like(bbox_semantic), torch.ones_like(bbox_semantic)).to(self.device)
            # c = bbox_semantic.detach().cpu().numpy()
            n_bboxes = torch.max(torch.sum(bbox_mask, dim = -1))
            bbox_semantic[bbox_semantic == -1] = 100
            for i in range(n_bboxes):
                bbox_semantic[:,i]  = bbox_semantic[:,i] + (i + 1) * 1000
            render_obj = True if (n_bboxes.item() > 0) else False
        else:
            render_obj = False
            bbox_mask = None
            n_bboxes = 0

        return_depth_map = self.return_depth_map
        batch_size, n_pixels = rays.shape[0], rays.shape[1]
        H, W = rays.shape[1], rays.shape[2]
        n_pixels = H * W
        n_samples_obj = self.n_samples_obj
        rays_dir, rays_pixel, rays_origin = rays[...,6:9], rays[...,3:6] , rays[...,0:3]
        rays_dir, rays_pixel, rays_origin = rays_dir.reshape((batch_size, n_pixels, 3)), rays_pixel.reshape((batch_size, n_pixels, 3)), rays_origin.reshape((batch_size, n_pixels, 3))

        # rays_dir_debug = rays_dir.reshape((2,94,352,3)).detach().cpu().numpy()
        origin_camera = rays_origin
        # pixels_camera = rays_pixel
        # a = torch.linalg.norm(rays_dir, dim = -1)
        # Render objects
        # batch_size x n_bbox x n_points x n_steps
        # feat, sigma = [], []
        scope_obj = []
        N = 0
        sigma, feat, scope, depth = {}, {}, {}, {}
        semantic = {}
        if render_obj :
            assert bbox_mask != None and bbox_trs != None
            sigma_obj, feat_obj, depth_obj = [], [], []
            z_bbox, c_bbox = latent_codes['z_bbox'], latent_codes['c_bbox']
            N += n_bboxes

            # unify sample within aabb
            depth_range = self.get_depth_range(bbox_trs[:,:n_bboxes])
            # if True:
            #     di = depth_range[0] + \
            #     torch.linspace(0., 1., steps=n_samples_obj).reshape(1, 1, -1).to(self.device) * (
            #         depth_range[1] - depth_range[0])  
            #     di = di.tile((1,1,n_pixels)).reshape((batch_size, n_bboxes, n_pixels, n_samples_obj)).to(self.device)
            # else: 
            #     d0 = (depth_range[0] - origin_camera[2]) / rays_dir
            #     length = (depth_range[1] - depth_range[1])/ rays_dir
            #     di = d0 + \
            #     torch.linspace(0., 1., steps=n_samples_obj).reshape(1, 1, -1).to(self.device) * length  

            # if mode == 'train':
            #     di = self.add_noise_to_interval(di)
            p_valid, r_valid,z_valid,c_valid, semantic_valid= [], [], [], [], []
            valid_num, valid_idx = [], []
            t_valid = []


            for i in range(n_bboxes):
                # print(p_i_mask.numpy())
                p_i, t_i,r_i, mask_box_i = self.get_evaluation_points(rays_dir, origin_camera, bbox_trs, i, mode = mode)
                p_valid.append(p_i)
                t_valid.append(t_i)
                r_valid.append(r_i)
                # t_valid.append(t_i)
  
                valid_num += torch.sum(mask_box_i, dim = -1)
                valid_idx += mask_box_i[:,None]

                # if self.triplane_generator == None:
                for b in range(batch_size):
                    if valid_num[i  * batch_size + b] < 10:
                        bbox_semantic[b,i] = 100
                        batch['bbox_semantic'][b,i] = -1
                    z_valid.append(z_bbox[b,i][None,None,:].repeat(valid_num[i  * batch_size + b],n_samples_obj,1))
                    c_valid.append(c_bbox[b,i][None,None,:].repeat(valid_num[i  * batch_size + b],n_samples_obj,1))
                    semantic_valid.append(bbox_semantic[b,i][None,None].repeat(valid_num[i  * batch_size + b],n_samples_obj,1))
            
            # a = valid_num.detach().cpu().numpy()
            # padding zero for p_valid
            p_loc = nn.utils.rnn.pad_sequence(p_valid, batch_first = True)
    

            p_valid, r_valid = torch.cat(p_valid, dim = 0), torch.cat(r_valid, dim = 0)
            z_valid, c_valid = torch.cat(z_valid, dim = 0), torch.cat(c_valid, dim = 0)
            t_valid =  torch.cat(t_valid, dim = 0)
            semantic_valid = torch.cat(semantic_valid, dim = 0)
            valid_idx =  torch.stack(valid_idx, dim = 0).view(n_bboxes,batch_size,-1)
            valid_num = torch.stack(valid_num).view(n_bboxes,batch_size)
            # a = z_valid.detach().cpu().numpy()

            valid_point_is_empty =  True if min(p_valid.shape) == 0 else False
            if valid_point_is_empty:
                render_obj = False
                p_valid, r_valid = torch.zeros((1,self.n_samples_obj,3)).to(self.device), torch.zeros((1,self.n_samples_obj,3)).to(self.device)
                # p_valid, r_valid,= p_i[0,0:1].to(self.device), r_i[0,0:1].to(self.device)
                z_valid = torch.zeros((1,self.n_samples_obj,self.z_dim_obj)).to(self.device) 
                c_valid = torch.zeros((1,self.n_samples_obj,3)).to(self.device) 
                # if self.triplane_generator == None else torch.zeros((1,3,16)).to(self.device)
                semantic_valid = torch.zeros((1,self.n_samples_obj,1)).to(self.device) 
                valid_num[0] = 1
                valid_idx[0][0] = True

            sigma_valid, feat_valid = self.batchify_rays_render(p = p_valid, 
            r = r_valid, z = z_valid, c = c_valid, s = semantic_valid % 1000, mode=mode)

            m = 0
            if render_obj:
                for i in range(n_bboxes):
                    for b in range(batch_size):
                        #valid_idx_debug = valid_idx[i].detach().cpu().numpy()
                        depth_i = torch.ones(n_pixels, n_samples_obj ,1).to(self.device) * 2000
                        sigma_i = torch.zeros(n_pixels, n_samples_obj ,1).to(self.device)
                        feat_i = torch.zeros(n_pixels, n_samples_obj, self.feature_dim +3).to(self.device)
                        sigma_i[valid_idx[i,b] == 1] = sigma_valid[m : m + valid_num[i,b]]
                        feat_i[valid_idx[i,b] == 1] = feat_valid[m : m + valid_num[i,b]]
                        depth_i[valid_idx[i,b] == 1] =  t_valid[m : m + valid_num[i,b]][:,:,None]

                        # get bbox scope
                        bbox_scope_i = torch.ones_like(sigma_i).to(self.device)
                        bbox_scope_i[valid_idx[i,b] == 0] = 0.
                        bbox_scope_i = bbox_scope_i.reshape(H, W, n_samples_obj)
                        bbox_scope_i = torch.sum(bbox_scope_i, dim = -1, keepdim=False)

                        m += valid_num[i,b]
                        feat_obj.append(feat_i.reshape(n_pixels, n_samples_obj, -1))
                        sigma_obj.append(sigma_i.reshape( n_pixels, n_samples_obj))
                        depth_obj.append(depth_i.reshape( n_pixels, n_samples_obj))
                        scope_obj.append(bbox_scope_i.reshape(H, W, 1))

                sigma_obj = torch.stack(sigma_obj, dim=0)
                #sigma_obj = sigma_obj.detach().cpu().numpy()
                sigma_obj = sigma_obj.reshape((n_bboxes, batch_size, sigma_obj.shape[1],sigma_obj.shape[2]))
                feat_obj = torch.stack(feat_obj, dim=0)
                feat_obj = feat_obj.reshape((n_bboxes, batch_size,feat_obj.shape[1],feat_obj.shape[2],feat_obj.shape[3]))
                scope_obj = torch.stack(scope_obj, dim = 0)
                scope_obj = scope_obj.reshape((n_bboxes, batch_size,scope_obj.shape[1],scope_obj.shape[2],scope_obj.shape[3]))
                scope_obj = torch.where(scope_obj != 0, torch.ones_like(scope_obj), torch.zeros_like(scope_obj))

                depth_obj = torch.stack(depth_obj, dim = 0)
                depth_obj = depth_obj.reshape((n_bboxes, batch_size,depth_obj.shape[1],depth_obj.shape[2]))


                sigma['obj'] = sigma_obj
                feat['obj'] = feat_obj
                scope['obj'] = scope_obj
                depth['obj'] = depth_obj
                semantic['obj'] = bbox_semantic.reshape((batch_size,bbox_semantic.shape[1] , 1,1)).repeat((1,1, n_pixels,n_samples_obj)).permute((1,0,2,3))
                # semantic_debug = semantic['obj'].detach().cpu().numpy().reshape((batch_size, n_bboxes, H, W, n_samples_obj))

        if self.render_bg: 
        # # GIRAFFE_only

            di = torch.linspace(0., 1., steps=self.n_samples_bg).reshape(1, 1, -1).to(self.device) * self.depth_range
            di = di.repeat((batch_size,n_pixels,1)).to(self.device)
            if mode == 'train':
                di = self.add_noise_to_interval(di)

            ri =  rays_dir.reshape((batch_size,n_pixels,1,3)).repeat((1,1,self.n_samples_bg,1))

            zi = latent_codes['z_global'].reshape((batch_size,1,1,-1)).repeat((1,n_pixels,self.n_samples_bg,1))

            p = origin_camera.unsqueeze(-2) + di.unsqueeze(-1) *ri
            # t_bg = ((p - origin_camera.unsqueeze(-2))ri )[:,2]

            pi = p - torch.tensor((self.x_min,self.y_min,self.z_min), device=self.device)
            pi = pi / torch.tensor((self.x_max - self.x_min,self.y_max - self.y_min,self.z_max - self.z_min), device=self.device) - 0.5

            # Mask Point out side
            padd = 0.0
            mask_box = torch.all( pi  <= .5 + padd, dim=-1) & torch.all( pi  >= -.5 - padd, dim=-1) 
            p_valid = pi [mask_box == 1]
            z_valid = zi[mask_box == 1]
            r_valid = ri[mask_box == 1]
            sigma_valid, feat_valid = self.batchify_rays_render(p = p_valid, 
            r = r_valid, z = z_valid,  decoder = 'bg')
            feat_valid = feat_valid[:,-3 - self.feature_dim:]
            sigma_valid = sigma_valid.reshape(-1)

            sigma_bg = torch.zeros_like(mask_box.to(torch.float32))
            feat_bg = torch.zeros_like(mask_box.to(torch.float32)).unsqueeze(-1).repeat((1,1,1,feat_valid.shape[-1]))
            sigma_bg[mask_box == 1] = sigma_valid
            feat_bg[mask_box == 1] = feat_valid

            d_bg,semantic_bg = di ,  torch.ones_like(sigma_bg) * -1
            scope_bg = torch.any(mask_box.reshape((batch_size, -1, self.n_samples_bg)) != 0, dim = -1).reshape((batch_size, H, W, 1))
            sigma['bg'] = sigma_bg.reshape((batch_size, n_pixels, self.n_samples_bg))
            feat['bg'] = feat_bg.reshape((batch_size, n_pixels, self.n_samples_bg, -1))
            depth['bg'] = d_bg.reshape((batch_size, n_pixels, self.n_samples_bg))
            scope['bg'] = scope_bg
            semantic['bg'] = semantic_bg.reshape((batch_size, n_pixels, self.n_samples_bg))


        sigma_all, feat_all, depth_all, semantic_all = [], [], [], []
        for i in sigma.keys():
            if i == 'obj':
                sigma_all.append(sigma[i].permute(1,2,0,3).reshape((batch_size, n_pixels, -1)))
                feat_all.append(feat[i].permute((1,2,0,3,4)).reshape((batch_size, n_pixels, -1, self.feature_dim+3)))
                depth_all.append(depth[i].permute((1,2,0,3)).reshape((batch_size, n_pixels, -1)))
                semantic_all.append(semantic[i].permute((1,2,0,3)).reshape((batch_size, n_pixels, -1)))
            else:
                sigma_all.append(sigma[i])
                feat_all.append(feat[i])
                depth_all.append(depth[i])
                semantic_all.append(semantic[i])

        sigma_all = torch.cat(sigma_all, dim = -1)
        # torch.cuda.empty_cache()
        

        feat_all = torch.cat(feat_all, dim = -2)
        depth_all= torch.cat(depth_all, dim = -1)
        semantic_all= torch.cat(semantic_all, dim = -1)

        n_ray_samples = sigma_all.shape[-1]

        # Only keep points that has sigma value
        if mode == 'test':
            depth_all, sigma_all, feat_all,semantic_all =  remove_empty_points(depth=depth_all, sigma=sigma_all, feat=feat_all,semantic = semantic_all)
        # depth_all_debug = depth_all.detach().cpu().numpy()
        depth_all, sigma_all, feat_all, semantic_all =  sort_in_depth_order(depth=depth_all, sigma=sigma_all, feat=feat_all,semantic=semantic_all)
        # depth_all  = depth_all - origin_camera[...,-1,None]


        # Get Volume Weights
        # torch.cuda.empty_cache()
        weights = self.calc_volume_weights(depth_all, rays_dir, sigma_all)
        feat_map = torch.sum(weights.unsqueeze(-1) * feat_all, dim=-2)
        # color_map = torch.sum(weights.unsqueeze(-1) * feat_all[...,:3], dim=-2)

        weights_depth = weights.clone()

        
        weights_depth = F.normalize(weights_depth, p = 1, dim= -1)

        depth_map = torch.sum(weights_depth.unsqueeze(-1) * depth_all.unsqueeze(-1), dim=-2)
        # Reformat output
        depth_map = depth_map.permute(0, 2, 1).reshape(
            batch_size, -1, H, W)
        feat_map = feat_map.permute(0, 2, 1).reshape(
            batch_size, -1, H, W)  # B x feat x h x w
        #feat_map = feat_map.permute(0, 1, 3, 2)  # new to flip x/y


        output = {}
        output['depth'] = depth_map
        output['feat_raw'] = feat_map[:,:self.feature_dim]
        output['rgb_raw'] = feat_map[:,-3:]
        alpha_all = torch.sum(weights, dim = -1).reshape((batch_size, 1, H, W))
        output['alpha'] = alpha_all
        # print(output['feat_raw'][2,:,5,190])
        scope_all = []

        z_near, z_far = 1e-3, cfg.z_far
        for k in output:
            if re.search('depth', k) != None:
                output[k] = ((output[k] - z_near) / (z_far - z_near)).clamp(0,1.)

        if cfg.is_debug:
            save_tensor_img(torchvision.utils.make_grid(output['depth']), 'tmp', 'depth.jpg')
        return output
    
    def select_date(self, raw_batch, data_type):

        batch = {}
        batch['camera_mat'] = raw_batch['camera_mat']
        if data_type == 'gen':
            batch['idx'] =  raw_batch['idx']
            batch['camera_pose'] =  raw_batch['world_mat']
            batch['rays'] =  raw_batch['rays']
            # batch['semantic'] = raw_batch['semantic']
        elif data_type == 'dis':
            batch['idx'] =  raw_batch['idx_fake']
            batch['camera_pose'] =  raw_batch['world_mat_fake']
            batch['rays'] =  raw_batch['rays_fake']
            # batch['semantic'] = raw_batch['semantic_fake']
        else:
            raise KeyError

        if self.render_obj:
            if data_type == 'gen':
                batch['bbox_idx'], batch['bbox_semantic'], batch['bbox_trs']=  raw_batch['bbox_id'], raw_batch['bbox_semantic'], raw_batch['bbox_tr']
            elif data_type == 'dis':
                batch['bbox_idx'], batch['bbox_semantic'], batch['bbox_trs'] = \
                raw_batch['bbox_id_fake'], raw_batch['bbox_semantic_fake'], raw_batch['bbox_tr_fake']
            else:
                raise KeyError

            
        #get_latent_code
        if 'z_global' in raw_batch.keys():
            batch['z_global'] = raw_batch['z_global']
        if 'z_bbox' in raw_batch.keys():
            batch['z_bbox'] = raw_batch['z_bbox']

        return batch

    def forward(self, raw_batch, data_type  = 'gen',mode = 'train'):

        batch, latent_codes = {}, {}
        batch = self.select_date(raw_batch, data_type=data_type) 

        # Get latent codes
        if 'z_global' in batch.keys():
            latent_codes['z_global'] = batch['z_global']
        else:
            latent_codes['z_global'] = self.get_global_code(batch['idx'])
        if 'z_bbox' in batch.keys():
            latent_codes['z_bbox'] = batch['z_bbox']
        elif self.render_obj:
            latent_codes['z_bbox'] = self.get_instance_code(batch['bbox_idx'])
             
        # Get feature vecotor(semanti feature, stuff local feature grid and bbox triplane)
        features = self.get_features(batch = batch,latent_codes = latent_codes)
        latent_codes.update(features)

        # Volume Rendering
        output = {}
        volume_render_output = self.volume_render_image(
            batch = batch,latent_codes=latent_codes,mode=mode)
            # road_rays_intersection = rays_intersection_loc, road_rays_semantic = rays_semantic,)
        

        output = volume_render_output
        if self.neural_renderer is not None:
            rgb_refine = self.neural_renderer(x = volume_render_output['feat_raw'], z = latent_codes['z_global'])
    
            if self.sr_multiple > 1 and self.render_obj:
                for k in output:
                    if re.search('bbox', k) != None:
                        H, W = output[k].shape[-2] * self.sr_multiple, output[k].shape[-1] * self.sr_multiple
                        output[k] = F.interpolate(output[k],(H, W),mode = 'nearest')
    
                
            output['rgb'] = rgb_refine
            # output['rgb'] = rgb_refine * valid_rays
            # print(output['rgb'][2,:,5,190])

            if self.render_obj and self.semantic_aware_obj :
                for s in self.ins_semantic_list:
                    valid_rays_s = torch.where(output['depth_%s'%s] != 0, torch.ones_like(output['depth_%s'%s]), torch.zeros_like(output['depth_%s'%s]))
                    output['rgb_%s'%s] = rgb_refine * valid_rays_s 
        else:
            output['rgb'] = volume_render_output['rgb_raw']

        if cfg.is_debug:
            save_tensor_img(torchvision.utils.make_grid(output['rgb']), 'tmp', 'rgb.jpg')
            output['frame_id'] = batch['idx']
            output['camera_intrinsic'] = batch['camera_mat']
            output['camera_pose'] = batch['camera_pose']
            if self.render_obj:
                output['bbox_tr'] = batch['bbox_trs']

        if self.use_occupancy_mask: 
            occupancy_mask = batch['occupancy_mask']
            output['occupancy_mask'] = occupancy_mask

        
        return output
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
The ray marcher takes the raw output of the implicit representation and uses the volume rendering equation to produce composited colors and depths.
Based off of the implementation in MipNeRF (this one doesn't do any cone tracing though!)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce

from lib.config import cfg


def ray_AABB():
    pass

class NeRFRayMarcher(nn.Module):
    def __init__(self, decoder, **kwargs):
        super().__init__()
        self.decoder = decoder


    def batchify_rays_render(self, p, r, z, c, s = None, mode = 'train', **kwargs):
        #batchify_rays render
        # if len(p.shape) == 3:
        ray_num, n_samples = p.shape[0],p.shape[1]
        p = p.view(ray_num * n_samples, -1)
        z = z.view(ray_num * n_samples, -1)
        r = r.view(ray_num * n_samples, -1)
        s = s.view(ray_num * n_samples, -1)
        c = c.view(ray_num * n_samples, -1)

        feat, sigma = [], []
        chunk = 300000
        for i in range(0, p.shape[0], chunk):
            feat_chunk, sigma_chunk =self.decoder(pts = p[i:i+chunk],ray_d= r[i:i+chunk], z = z[i:i+chunk], c = c[i:i+chunk], semantic = s[i:i+chunk])
            feat.append(feat_chunk)
            sigma.append(sigma_chunk)
            torch.cuda.empty_cache()
        feat = torch.cat(feat, dim = 0)
        sigma = torch.cat(sigma, dim = 0)
        if mode == 'train':
            sigma += torch.randn_like(sigma)
        sigma = sigma.reshape(ray_num, n_samples)
        feat = feat.reshape(ray_num, n_samples, -1)

        return sigma, feat
    
    def add_noise_to_interval(self, di):
        '''
        Add noise to interval during training, as in original NeRF 
        '''
        di_mid = .5 * (di[..., 1:] + di[..., :-1])
        di_high = torch.cat([di_mid, di[..., -1:]], dim=-1)
        di_low = torch.cat([di[..., :1], di_mid], dim=-1)
        noise = torch.rand_like(di_low)
        ti = di_low + (di_high - di_low) * noise
        return ti

class bboxRayMarcher(NeRFRayMarcher):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def get_evaluation_points(self, rays_o, rays_d, bbox_trs, n_samples, mode = 'train'):
        ''' Get the evaluation points (camera cordinate)

        Args:
            pixels(tensor 1*1*1): Positions of all pixels on camera plane.
            camera(tensor 1*1*1): Posttion of camera
            di(): All points on all camera rays
            bbox_tes: {s, t, R} of the boundingbox
            i: the index of the input boundng box(the last one is background)
        Return:

        '''
        batch_size, n_pixels = rays_d.shape[0] , rays_d.shape[1]

        rays_d = torch.cat((rays_d, torch.zeros_like(rays_d)[...,0:1]), dim=-1)
        rays_o = torch.cat((rays_o, torch.ones_like(rays_o)[...,0:1]), dim=-1)

        # q = torch.bmm(torch.inverse(bbox_trs[:,i,...]), rays_o.permute(0,2,1)).permute(0,2,1)[...,3].detach().cpu().numpy()
        rays_o_local = torch.bmm(torch.inverse(bbox_trs), rays_o.permute(0,2,1)).permute(0,2,1)[...,:3].reshape(-1,3)
        rays_d_local = torch.bmm(torch.inverse(bbox_trs), rays_d.permute(0,2,1)).permute(0,2,1)[...,:3].reshape(-1,3)
        rays_d_local_debug = rays_d_local.detach().cpu().numpy()
        
        ## batchify ray-AABB intersetction
        bounds = torch.tensor([[-0.5,-0.5,-0.5],[0.5,0.5,0.5]], device=rays_o.device)
        nominator = bounds[None] - rays_o_local[:, None]
        # calculate the step of intersections at six planes of the 3d bounding box
        norm_d = torch.linalg.norm(rays_d_local, dim=-1, keepdims=True)
        view_dir = rays_d_local / norm_d
        view_dir[torch.abs(view_dir) < 1e-3] = 1e-3

        tmin = (bounds[0:1] - rays_o_local) / view_dir
        tmax = (bounds[1:2] - rays_o_local) / view_dir
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

        rays_mask = mask_at_box.reshape(batch_size,-1)

        return {
            'p_pts' : p_i_valid, 
            'p_t': t_i_valid, 
            'viewdirs': ray_i_valid, 
            'rays_mask': rays_mask}


    def forward(self, latent_codes, rays, render_option, **kwargs):

        batch_z, batch_c = latent_codes['z_bbox'], latent_codes['c_bbox']
        batch_trs, batch_semantic = kwargs['bbox_trs'].clone(), kwargs['bbox_semantic'].clone()
        batch_size, H, W = render_option.batch_size, render_option.H_ray, render_option.W_ray
        # max_bbox =  batch_trs.shape[1]
        device = rays.device
        if render_option.mode == 'train':
            n_samples_obj = render_option.n_samples_obj
        else:
            n_samples_obj =  render_option.n_samples_obj_render

        valid_bbox_mask = batch_semantic != -1
        # valid_idx = torch.argwhere(valid_bbox_mask == 1)
        bbox_idx = valid_bbox_mask == 1
        maxbbox = bbox_idx.sum(dim = 1).max()
        # max_bbox =  bbox_idx.shape[1]
        nbbox = torch.sum(bbox_idx)
        if nbbox == 0:
            return -1 
        # Rays
        rays = repeat(rays, 'B H W n -> B nb (H W) n', nb = bbox_idx.shape[1])
        
        # Select bbox
        for i in range(batch_semantic.shape[1]):
            # Assign each bbox an instance ID
            batch_semantic[:,i] = batch_semantic[:,i] + 1000 * i
        bbox_semantic = batch_semantic[bbox_idx]
        bbox_trs = batch_trs[bbox_idx]
        bbox_rays = rays[bbox_idx]
        bbox_z, bbox_c = batch_z[bbox_idx], batch_c[bbox_idx]
        bbox_rays_o, bbox_rays_d = bbox_rays[...,0:3], bbox_rays[...,3:6]

        #? Step1 Transform Rays and sampling within object AABB
        valid_samples = self.get_evaluation_points(bbox_rays_o, bbox_rays_d, bbox_trs, n_samples = n_samples_obj, mode = render_option.mode)
        p_pts = valid_samples['p_pts']
        p_depth = valid_samples['p_t']
        p_view = valid_samples['viewdirs']
        rays_mask = valid_samples['rays_mask']
        rays_idx = rays_mask == True
               
        if p_pts.shape[0] == 0:
            # There is no ray intersect with bounding box, don not need to render bbox
            return -1

        p_semantic = repeat(bbox_semantic, 'N -> N HW n_sample' ,HW = H*W, n_sample = n_samples_obj)[rays_idx]
        p_c = repeat(bbox_c, 'N C -> N HW n_sample C' ,HW =H*W, n_sample = n_samples_obj)[rays_idx]
        p_z =  repeat(bbox_z, 'N C -> N HW n_sample C' ,HW =H*W, n_sample = n_samples_obj)[rays_idx]

        # p_view = repeat(rays_viewdir, 'n_ray C -> n_ray n_sample C', n_sample = n_samples_obj)[rays_idx]

        #? Step2 Neural Netweork Quiry
        p_sigma, p_feat = self.batchify_rays_render(p = p_pts, 
        r = p_view, z = p_z, c = p_c, s = p_semantic % 1000, mode=render_option.mode)


        #? Step3 Put valid points into corrspending bbox
        bbox_sigma = torch.zeros((nbbox, H * W, n_samples_obj), device = device)
        bbox_feat = torch.zeros((nbbox, H * W, n_samples_obj, 32), device = device)
        bbox_semantic = torch.ones((nbbox, H * W, n_samples_obj), device = device, dtype=torch.int64) * -1
        bbox_depth =torch.ones((nbbox, H * W, n_samples_obj), device = device) * 1000
        bbox_scope =torch.zeros((nbbox, H * W), device = device)

        bbox_sigma[rays_idx] = p_sigma
        bbox_scope[rays_idx] = 1
        bbox_feat[rays_idx] = p_feat
        bbox_semantic[rays_idx] = p_semantic
        bbox_depth[rays_idx] = p_depth

        #? Step4 Put bboxs into the batch
        render_option.maxbbox = maxbbox
        bbox_idx = bbox_idx[:,:maxbbox]
        batch_sigma = torch.zeros((batch_size , maxbbox, H * W, n_samples_obj), device = device)
        batch_feat = torch.zeros((batch_size, maxbbox, H * W, n_samples_obj, 32), device = device)
        batch_semantic = torch.ones((batch_size, maxbbox, H * W, n_samples_obj), device = device, dtype=torch.int64) * -1
        batch_depth =torch.ones((batch_size, maxbbox, H * W, n_samples_obj), device = device) * 1000
        batch_scope =torch.ones((batch_size, maxbbox, H * W), device = device) * -1

        batch_sigma[bbox_idx] = bbox_sigma
        batch_scope[bbox_idx] = bbox_scope
        batch_feat[bbox_idx] = bbox_feat
        batch_semantic[bbox_idx] = bbox_semantic
        batch_depth[bbox_idx] = bbox_depth

        obj_samples = {
            'sigma':batch_sigma,
            'feat':batch_feat,
            'scope':batch_scope,
            'depth':batch_depth,
            'semantic':batch_semantic,
        }

        return obj_samples

class stuffRayMarcher(NeRFRayMarcher):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        
    def forward(self, rays, latent_codes , render_option, K, c2w, semantic_embedding,  **kwargs):
        # Prepare Render Option
        device = rays.device
        batch_size = render_option.batch_size
        H, W = render_option.H_ray, render_option.W_ray
        voxel_range = render_option.voxel_range
        n_pixels = H * W
        mode = render_option.mode
        if mode == 'train':
            n_vox_intersection = render_option.n_vox_intersection
            n_samples_stuff = render_option.n_samples_stuff
        else:
            n_vox_intersection = render_option.n_vox_intersection_render
            n_samples_stuff = render_option.n_samples_stuff_render
        rays_o, rays_d = rays[...,0:3], rays[...,3:6]
        # camera_intrinsic, camera_pose = kwargs.camera_intrinsic, kwargs.camera_pose

        if render_option.stuff_representation == 'grid':
            loc_grid = latent_codes['stuff_loc_grid'] # shape [B, H, W, L 3]
            semantic_grid = latent_codes['stuff_semantic_grid'] # shape [B, H, W, L, 1]
            feature_grid = latent_codes['stuff_feature_grid'] # shape [B,H, W, L, 64]
        elif render_option.stuff_representation == 'triplane':
            pass
        else:
            raise ValueError

        #? Step2 Ray-World Intersection
        p_mask, p_pts, p_ptsvox, p_viewdir, p_feature, p_semantic, p_depth = [],[],[],[], [], [],[]
        # valid_idx = torch.zeros((batch_size, n_samples_stuff * n_pixels)).to(device)
        for i in range(batch_size):
            vox_results_i = ray_voxel_intersection_sampling(
                    rays_dir = rays_d[i] ,  
                    camera_intrinsic= K[i] ,
                    camera_pose = c2w[i], 
                    voxel_grid_semantic= semantic_grid[i], 
                    voxel_grid_feature = feature_grid[i], 
                    max_samples= n_samples_stuff,
                    voxel_range = voxel_range,
                    cam_res = (H, W),
                n_vox_intersection = n_vox_intersection,
                deterministic_sampiling = (mode == 'test'),
                altitude_decrease = render_option.is_kitti360,
                feature_type = render_option.stuff_representation)
    #     vox_results = {
    #     'p_mask':idx, 
    #     'p_pts_world' : samples_pts_global,
    #     'p_pts_invoxel': samples_pts_invox,
    #     'p_view_dir' : samples_dir_global,
    #     'p_semantic' : samples_semantic,
    #     'p_feature' : samples_feature,
    #     'p_depth' :  samples_depth ,
    # }
            p_mask.append(vox_results_i['p_mask'])
            p_pts.append(vox_results_i['p_pts_world'])
            p_ptsvox.append(vox_results_i['p_pts_invoxel'])
            p_viewdir.append(vox_results_i['p_view_dir'])
            p_feature.append(vox_results_i['p_feature'])
            p_semantic.append(vox_results_i['p_semantic'])
            p_depth.append(vox_results_i['p_depth'])
            
        p_mask = torch.stack(p_mask, dim=0)
        p_pts = torch.cat(p_pts, dim = 0)
        p_ptsvox = torch.cat(p_ptsvox, dim = 0)
        p_viewdir = torch.cat(p_viewdir, dim = 0)
        p_feature = torch.cat(p_feature, dim = 0)
        p_semantic = torch.cat( p_semantic, dim = 0)
        p_depth = torch.cat( p_depth, dim = 0)


        #? Step1 Ray-Voxel Intersection by default
        p_seg = semantic_embedding(p_semantic)
        axis_range = voxel_range[:,1] - voxel_range[:,0]
        p_pts_normalize = p_pts -  voxel_range[None, :,0]
        p_pts_normalize = 2 * (p_pts_normalize / axis_range - 0.5)
        p_pts = torch.cat((p_pts_normalize, p_ptsvox), dim = 1)

        #? Step3 Neural Network
        p_feat, p_sigma = self.decoder(w = p_feature, raydir =p_viewdir, pts = p_pts, seg = p_seg)
        if mode == 'train':
        # As done in NeRF, add noise during training
            p_sigma = p_sigma + torch.randn_like(p_sigma)


        #? Step3 Assign Valide samples to corrsepending index
        batch_sigma = torch.zeros((batch_size, n_pixels, n_samples_stuff), device=device)
        batch_feat = torch.zeros((batch_size, n_pixels, n_samples_stuff, p_feat.shape[-1]), device=device)
        batch_semantic = torch.zeros((batch_size, n_pixels, n_samples_stuff), device=device, dtype=torch.int32)
        batch_depth = torch.ones((batch_size, n_pixels, n_samples_stuff), device=device) * 1000
        batch_scope = torch.ones((batch_size, n_pixels), device=device, dtype=torch.int32) * -1

        # sigma_stuff_valid = sigma_stuff_valid.reshape(-1)
        # feat_stuff_valid  = feat_stuff_valid[:,:]

        # sigma_stuff = torch.zeros_like(valid_idx.to(torch.float32))
        # feat_stuff = torch.zeros_like(valid_idx.to(torch.float32)).unsqueeze(-1).repeat((1,1,feat_stuff_valid.shape[-1]))
        # scope_stuff = torch.any(valid_idx.reshape((batch_size, -1, n_samples_stuff)) != 0, dim = -1).reshape((batch_size, H, W, 1))

        # sigma_stuff[valid_idx == True] = sigma_stuff_valid
        # feat_stuff[valid_idx == True] = feat_stuff_valid

        # semantic_stuff = torch.ones_like(valid_idx).reshape(batch_size,-1) * -1.
        # semantic_stuff[valid_idx == True] = semantic_valid.to(torch.float32)
        # #!d_stuff = torch.ones_like(valid_idx).reshape(batch_size,-1) * 1e5
        # t_valid = torch.ones_like(valid_idx[:,:,None]).repeat(1,1,3)
        # t_valid[valid_idx == 1] = pts_valid
        # t_valid = t_valid.reshape(batch_size, n_pixels, n_samples_stuff, 3)
        
        
        # t_valid = (t_valid - origin_camera[:,:,None]) / rays_dir[:,:,None]
        # d_valid = t_valid * ray_cosine[:,:,None,None]
        # d_valid = d_valid.reshape(batch_size,-1,3)[valid_idx == True][:,2] 
        # d_stuff = torch.ones_like(valid_idx) * 1000
        # d_stuff[valid_idx == True] = d_valid
        batch_sigma[p_mask] = p_sigma.squeeze(-1)
        batch_feat[p_mask] = p_feat
        batch_depth[p_mask] = p_depth.squeeze(-1)
        batch_semantic[p_mask] = p_semantic
        batch_scope = batch_semantic[...,0]
        batch_scope = batch_scope.reshape((batch_size, H, W, 1))


        stuff_samples = {
            'sigma':batch_sigma,
            'feat':batch_feat,
            'depth':batch_depth,
            'scope':batch_scope,
            'semantic':batch_semantic,

        }
        return stuff_samples

class skyRayMarcher(NeRFRayMarcher):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def forward(self, z, rays, render_option, **kwargs):
        # z_global = latent_codes['z_global']
        
        batch_size, H, W = render_option.batch_size, render_option.H_ray, render_option.W_ray
        rays = rearrange(rays, 'B H W n -> B (H W) n')
        rays_o, rays_d, view_dir = rays[...,0:3], rays[...,3:6], rays[...,6:9]

        t = torch.ones((batch_size, H * W, 1), device=rays_o.device) * render_option.z_far

        p = rays_o + rays_d * t
        feat = self.decoder(view_dir, z)
        feat  = feat[...,:]

        y = p[...,1,None]
        scope = torch.ones_like(y)
        sigma = scope * 100

        sigma = sigma.reshape((batch_size, H * W, 1))
        feat = feat.reshape((batch_size, H * W, 1, -1))
        scope = scope.reshape((batch_size, H, W, 1))
        depth = t
        semantic = torch.ones_like(sigma) * 23


        sky_sample = {
            'sigma':sigma,
            'feat':feat,
            'scope':scope,
            'depth':depth,
            'semantic':semantic,
        }

        return sky_sample

#--------------------------------------------------GIRAFFE------------------------------------------------------------------------
class backgroundRayMarcher(NeRFRayMarcher):
    pass



from .utils import trans_vec_homo,  trans_vec_homo_batch, sample_from_3dgrid, sample_from_planes, generate_planes
from tools.voxlib import ray_voxel_intersection_perspective
import matplotlib.pyplot as plt

def ray_voxel_intersection_sampling(rays_dir, 
camera_intrinsic, camera_pose, voxel_grid_semantic, voxel_grid_feature, voxel_range = [[-32,32], [-14,2], [0,64]], cam_res = (94,352), max_samples = 12, n_vox_intersection= 4, sample_before_intersection = True, 
deterministic_sampiling = False, 
altitude_decrease = True,
feature_type = 'grid' ):
    '''
    voxel_semantic: (H,W,D)
    voxel_range: ((x_start, x_start),(x_start, x_start),(y_start, y_start))
    '''

    # Step1: flip voxel_semantic and make high index heigher
    img_H, img_W = cam_res[0],cam_res[1]
    device = voxel_grid_feature.device

    #! Transform to kitti voxel
    if altitude_decrease:
        voxel_semantic = torch.flip(voxel_grid_semantic, dims = [0]).squeeze(-1).to(torch.int32) 
        voxel_feature = torch.flip(voxel_grid_feature, dims = [0]).to(torch.float32)
    else:
        a = torch.sum(voxel_grid_semantic == 11)
        voxel_semantic = voxel_grid_semantic.squeeze(-1).to(torch.int32) 
        voxel_feature = voxel_grid_feature.to(torch.float32)
    
    voxel_semantic = torch.flip(voxel_grid_semantic, dims = [0]).squeeze(-1).to(torch.int32) 
    voxel_feature = torch.flip(voxel_grid_feature, dims = [0]).to(torch.float32)
    # Step1 prase camera args from intrinsic matrix (X,Y,Z) -> (Y,-X.Z)
    rays_dir_global = rays_dir.reshape(-1, 3)
    cam_f = camera_intrinsic[0,0] #Camera focal length (in pixels).
    cam_c = (camera_intrinsic[1,2], camera_intrinsic[0,2]) #Camera optical center (Y,X).

    #?Step2 transform from global to local coordinate
    Y_sample_num, X_sample_num, Z_sample_num =  voxel_semantic.shape[0], voxel_semantic.shape[1], voxel_semantic.shape[2]
    X_scale = (voxel_range[0][1] - voxel_range[0][0])/X_sample_num
    Y_scale=  (voxel_range[1][1] - voxel_range[1][0])/Y_sample_num
    Z_scale = (voxel_range[2][1] - voxel_range[2][0])/Z_sample_num
    global2local_mat = torch.inverse(torch.tensor(
        [[0,X_scale,0,voxel_range[0][0]],
        [-Y_scale,0,0,voxel_range[1][1]],
        [0,0,Z_scale,voxel_range[2][0]],
        [0,0,0,1]])
    ).to(device)


    rays_dir_local = trans_vec_homo_batch(m=global2local_mat, v=rays_dir_global, is_vec=True, normalize=False).reshape((img_H, img_W, 3))

    s = torch.norm(rays_dir_local, dim=-1)

    camera_extrinsic_global = torch.inverse(camera_pose)
    ''' camera_extrinsic = 
    [[ux vx nx tx]
    [uy vy ny ty]
    [uz vz nz tz]
    [0 0 0 1]]
    U :right; V:up; N:look dir
    '''
    camera_up_global = camera_extrinsic_global[1,:3] 
    camera_dir_global = camera_extrinsic_global[2,:3]
    camera_ori_global = torch.tensor([camera_pose[0,3],camera_pose[1,3],camera_pose[2,3],1]).to(device)

    cam_up_local =  trans_vec_homo(m = global2local_mat, v = camera_up_global, is_vec=True)
    cam_dir_local = trans_vec_homo(m = global2local_mat, v = camera_dir_global, is_vec=True)
    cam_ori_local = trans_vec_homo(m = global2local_mat, v = camera_ori_global, is_vec=False)


    #?Step3 Utilize cancraft's ray_voxel_intersection kernel
    rays_dir_local = rays_dir_local.contiguous()

    voxel_id, depth2, raydirs = ray_voxel_intersection_perspective(
                        voxel_semantic, 
                        rays_dir_local,
                        cam_ori_local, cam_dir_local, cam_up_local, 
                        cam_f, cam_c, cam_res,
                        n_vox_intersection)

    if True:
        va =  voxel_id[:,:,:].detach().cpu().numpy()
        d = depth2.detach().cpu().numpy()
        plt.imsave(os.path.join(cfg.out_tmp_img_dir, 'a.jpg'), va[:, :, 0, 0])
    

    #?Step4 Sampleing based on intersection
    voxel_id = voxel_id
    # depth = (depth2[0:1,...] + torch.rand(depth2[1:2,...].shape).to(device) * (depth2[1:2,...] - depth2[0:1,...])).squeeze(0).squeeze(-1)
    use_box_boundaries = False
    if use_box_boundaries:
        nsamples=  max_samples - n_vox_intersection
    else:
        nsamples=  max_samples + 1

    depth, _, new_idx = sample_depth_batched(depth2.unsqueeze(0), 
        nsamples = nsamples,
        use_box_boundaries=use_box_boundaries,
        deterministic= deterministic_sampiling,
        sample_depth = n_vox_intersection)
    
    depth = depth.squeeze(0).squeeze(-1)
    new_idx = new_idx.squeeze(0)
    mc_masks = torch.gather(voxel_id, -2, new_idx)


    mc_masks_debug = mc_masks.cpu().numpy() 

    idx = (torch.isnan(depth) == False) # * (mc_masks.squeeze(-1) != 0)
    depth_valid =depth[idx].unsqueeze(-1)
    # depth_debug = depth[depth > 64].cpu().numpy()
    rays_dir_local = rays_dir_local.reshape((img_H, img_W, 1, 3)).repeat(1, 1, max_samples, 1)[idx]
    #.permute(0,1,3,2)[voxel_id != 0]
    # rays_dir_local = rays_dir_local  / rays_dir_local[...,-1,None]
    samples_depth = depth_valid / torch.norm(rays_dir_local, dim=-1, keepdim=True)
    p_loc_local = cam_ori_local + samples_depth * rays_dir_local

    # p_loc_local_debug = p_loc_local.cpu().numpy()
    p_loc_local_normalized = 2 * (p_loc_local / torch.tensor([X_sample_num, Y_sample_num, Z_sample_num]).to(device) - 0.5)

    if feature_type == 'grid':
        samples_feature = sample_from_3dgrid(grid=voxel_feature.permute(1,0,2,3).unsqueeze(0), coordinates=p_loc_local_normalized.unsqueeze(0), grid_form_return=False,  mode= 'bilinear', padding_mode='border').squeeze(0)
        samples_feature = F.softplus(samples_feature)
    elif feature_type == 'triplane':
        _, H, W = voxel_grid_feature.shape
        triplane_feature = voxel_grid_feature.reshape((1, 3, -1, H, W))
        samples_feature = sample_from_planes(plane_features=triplane_feature, coordinates=p_loc_local_normalized.unsqueeze(0), mode= 'bilinear', padding_mode='zeros').squeeze(0)
        samples_feature =  samples_feature.mean(0)
        
    samples_semantic = mc_masks[idx].squeeze(-1)

    #samples_semantic = voxel_id[idx]
    local2global_mat = torch.inverse(global2local_mat)
    samples_pts_global = trans_vec_homo_batch(m=local2global_mat, v=p_loc_local, is_vec=False)
    samples_dir_global = trans_vec_homo_batch(m=local2global_mat, v=rays_dir_local, is_vec=True)

    if True:
        samples_pts_invox = p_loc_local % 1

    idx = rearrange(idx, 'H W N -> (H W) N')

    vox_results = {
        'p_mask':idx, 
        'p_pts_world' : samples_pts_global,
        'p_pts_invoxel': samples_pts_invox,
        'p_view_dir' : samples_dir_global,
        'p_semantic' : samples_semantic,
        'p_feature' : samples_feature,
        'p_depth' :  samples_depth ,
    }
    return vox_results
    # return idx.reshape(-1), samples_pts_global, samples_pts_invox, samples_dir_global,samples_feature,samples_semantic





def sample_depth_batched(depth2, nsamples, deterministic=False, use_box_boundaries=True, sample_depth=4):
    r"""    Make best effort to sample points within the same distance for every ray.
    Exception: When there is not enough voxel.

    Args:
        depth2 (N x 2 x 256 x 256 x 4 x 1 tensor):
        - N: Batch.
        - 2: Entrance / exit depth for each intersected box.
        - 256, 256: Height, Width.
        - 4: Number of intersected boxes along the ray.
        - 1: One extra dim for consistent tensor dims.
        depth2 can include NaNs.
        deterministic (bool): Whether to use equal-distance sampling instead of random stratified sampling.
        use_box_boundaries (bool): Whether to add the entrance / exit points into the sample.
        sample_depth (float): Truncate the ray when it travels further than sample_depth inside voxels.
    """

    bs = depth2.size(0)
    dim0 = depth2.size(2)
    dim1 = depth2.size(3)
    dists = depth2[:, 1] - depth2[:, 0]
    dists[torch.isnan(dists)] = 0  # N, 256, 256, 4, 1
    accu_depth = torch.cumsum(dists, dim=-2)  # N, 256, 256, 4, 1
    total_depth = accu_depth[..., [-1], :]  # N, 256, 256, 1, 1

    total_depth = torch.clamp(total_depth, None, sample_depth)

    # Ignore out of range box boundaries. Fill with random samples.
    if use_box_boundaries:
        boundary_samples = accu_depth.clone().detach()
        boundary_samples_filler = torch.rand_like(boundary_samples) * total_depth
        bad_mask = (accu_depth > sample_depth) | (dists == 0)
        boundary_samples[bad_mask] = boundary_samples_filler[bad_mask]

    rand_shape = [bs, dim0, dim1, nsamples, 1]
    # 256, 256, N, 1
    if deterministic:
        rand_samples = torch.empty(rand_shape, dtype=total_depth.dtype, device=total_depth.device)
        rand_samples[..., :, 0] = torch.linspace(0, 1, nsamples+2)[1:-1]
    else:
        rand_samples = torch.rand(rand_shape, dtype=total_depth.dtype, device=total_depth.device)  # 256, 256, N, 1
        # Stratified sampling as in NeRF
        rand_samples = rand_samples / nsamples
        rand_samples[..., :, 0] += torch.linspace(0, 1, nsamples+1, device=rand_samples.device)[:-1]
    rand_samples = rand_samples * total_depth  # 256, 256, N, 1

    # Can also include boundaries
    if use_box_boundaries:
        rand_samples = torch.cat([rand_samples, boundary_samples, torch.zeros(
            [bs, dim0, dim1, 1, 1], dtype=total_depth.dtype, device=total_depth.device)], dim=-2)
    rand_samples, _ = torch.sort(rand_samples, dim=-2, descending=False)

    midpoints = (rand_samples[..., 1:, :] + rand_samples[..., :-1, :]) / 2
    new_dists = rand_samples[..., 1:, :] - rand_samples[..., :-1, :]

    # Scatter the random samples back
    # 256, 256, 1, M, 1 > 256, 256, N, 1, 1
    idx = torch.sum(midpoints.unsqueeze(-3) > accu_depth.unsqueeze(-2), dim=-3)  # 256, 256, M, 1

    # print(idx.shape, idx.max(), idx.min()) # max 3, min 0

    depth_deltas = depth2[:, 0, :, :, 1:, :] - depth2[:, 1, :, :, :-1, :]  # There might be NaNs!
    depth_deltas = torch.cumsum(depth_deltas, dim=-2)
    depth_deltas = torch.cat([depth2[:, 0, :, :, [0], :], depth_deltas+depth2[:, 0, :, :, [0], :]], dim=-2)
    heads = torch.gather(depth_deltas, -2, idx)  # 256 256 M 1
    # heads = torch.gather(depth2[0], -2, idx) # 256 256 M 1

    # print(torch.any(torch.isnan(heads)))
    rand_depth = heads + midpoints  # 256 256 N 1
    return rand_depth, new_dists, idx


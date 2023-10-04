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
The ray sampler is a module that takes in camera matrices and resolution and batches of rays.
Expects cam2world matrices that use the OpenCV camera coordinate system conventions.
"""

import torch
import torch.nn.functional as F
from .utils import trans_vec_homo,  trans_vec_homo_batch, sample_from_3dgrid, sample_from_planes, generate_planes
from tools.voxlib import ray_voxel_intersection_perspective
import matplotlib.pyplot as plt

def unify_sampling(rays, depth_range):
    '''
    rays(tensors):
    depth_range():
    '''
    pass

def unify_sample_around_a_point(rays, center_depth):
    pass 

def unify_sample_within_bbx(rays, bbx_trs):
    pass 

def importance_sample(rays, voxel):
    pass


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
    # a = idx.detach().cpu().numpy()
    # b = rand_depth.detach().cpu().numpy()

    return rand_depth, new_dists, idx



def ray_voxel_intersection_sampling(rays_dir, 
camera_intrinsic, camera_pose, voxel_grid_semantic, voxel_grid_feature, voxel_range = [[-32,32], [-14,2], [0,64]], cam_res = (94,352), max_samples = 12, n_vox_intersection= 4, sample_before_intersection = True, 
deterministic_sampiling = False, 
altitude_decrease = True, feature_type = 'grid' ):
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
    rays_dir_global = rays_dir.reshape(-1,3)
    cam_f = camera_intrinsic[0,0] #Camera focal length (in pixels).
    cam_c = (camera_intrinsic[1,2], camera_intrinsic[0,2]) #Camera optical center (Y,X).

    # Step2 transform from global to local coordinate
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
    # a =  trans_vec_homo(m = global2local_mat, v = torch.tensor((-32.,-14,0)), is_vec=False)
    # b =  trans_vec_homo(m = global2local_mat, v = torch.tensor((32.,2,64)), is_vec=False)

    rays_dir_local = trans_vec_homo_batch(m = global2local_mat, v = rays_dir_global, is_vec= True).reshape((img_H, img_W,3))

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


    # Step3 Utilize cancraft's ray_voxel_intersection kernel
    rays_dir_local = rays_dir_local.contiguous()
    # rays_dir_local_debug =rays_dir_local.detach().cpu().numpy()
    # print( b.stride())
    voxel_id, depth2, raydirs = ray_voxel_intersection_perspective(
                        voxel_semantic, 
                        rays_dir_local,
                        cam_ori_local, cam_dir_local, cam_up_local, 
                        cam_f, cam_c, cam_res,
                        n_vox_intersection)

    if True:
        va =  voxel_id[:,:,:].detach().cpu().numpy()
        d = depth2.detach().cpu().numpy()
        # if va.max() > 6:
        #      print('?')
        # import matplotlib.pyplot as plt
        plt.imsave('tmp/a.jpg', va[:,:,0,0])
    

    # if sample_before_intersection: 
    #     voxel_id = torch.cat((torch.zeros_like(voxel_id)[...,0:1,:],voxel_id) , dim = -2)
    #     empty_depth_out =  depth2[0:1,:,:,0:1,:]
    #     empty_depth_in = depth2[0:1,:,:,0:1,:] - torch.rand_like(empty_depth_out) / 2
    #     empty_depth = torch.cat((empty_depth_in,empty_depth_out), dim = 0)
    #     depth2 = torch.cat((empty_depth, depth2), dim = -2)

    # Step4 Sampleing based on intersection
    voxel_id = voxel_id
    # depth = (depth2[0:1,...] + torch.rand(depth2[1:2,...].shape).to(device) * (depth2[1:2,...] - depth2[0:1,...])).squeeze(0).squeeze(-1)
    use_box_boundaries = False
    if use_box_boundaries:
        nsamples=  max_samples  - n_vox_intersection
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

    # a =  voxel_id.cpu().numpy() 
    # b = mc_masks.cpu().numpy() 
    mc_masks_debug = mc_masks.cpu().numpy() 

    idx = (torch.isnan(depth) == False) # * (mc_masks.squeeze(-1) != 0)
    depth_valid =depth[idx].unsqueeze(-1)
    # depth_debug = depth[depth > 64].cpu().numpy()
    rays_dir_local = rays_dir_local.reshape((img_H, img_W,1,3)).repeat(1,1,max_samples,1)[idx]
    #.permute(0,1,3,2)[voxel_id != 0]
    # rays_dir_local = rays_dir_local  / rays_dir_local[...,-1,None]
    p_loc_local = cam_ori_local + depth_valid * rays_dir_local
    # ! p_loc_local = cam_ori_local + depth_valid + rays_dir_local  / rays_dir_local[...,-1,None]

    # ! p_loc_local = cam_ori_local + (depth_valid - cam_ori_local[2]) * rays_dir_local
    # p_loc_local_debug = p_loc_local.cpu().numpy()
    p_loc_local_normalized = 2 * (p_loc_local / torch.tensor([X_sample_num, Y_sample_num, Z_sample_num]).to(device) - 0.5)

    #! in grid sample index 0, 0, 0 means left top
    if feature_type == 'grid':
        samples_feature = sample_from_3dgrid(grid=voxel_feature.permute(1,0,2,3).unsqueeze(0), coordinates=p_loc_local_normalized.unsqueeze(0), grid_form_return=False,  mode= 'bilinear', padding_mode='border').squeeze(0)
        samples_feature = F.softplus(samples_feature)
    elif feature_type == 'triplane':
        _, H, W = voxel_grid_feature.shape
        triplane_feature = voxel_grid_feature.reshape((1, 3, -1, H, W))
        samples_feature = sample_from_planes(plane_features=triplane_feature, coordinates=p_loc_local_normalized.unsqueeze(0), mode= 'bilinear', padding_mode='zeros').squeeze(0)
        samples_feature =  samples_feature.mean(0)
        
    samples_semantic = mc_masks[idx].squeeze(-1)
    # samples_semantic = sample_from_3dgrid(grid=voxel_semantic.permute(1,0,2).unsqueeze(0).unsqueeze(-1), coordinates=p_loc_local_normalized.unsqueeze(0), grid_form_return=False, padding_mode= 'border', mode = 'nearest').squeeze(0).squeeze(-1).to(torch.int32)
    # a = (sampled_feature.squeeze(0) -p_loc_local).mean(dim = -1).cpu().numpy()
    # b =  np.argwhere(abs(a) > 0.5)


    # Transform sanmpls pts and raydir from local to global
    #samples_semantic = voxel_id[idx]
    local2global_mat = torch.inverse(global2local_mat)
    samples_pts_global = trans_vec_homo_batch(m =local2global_mat , v = p_loc_local, is_vec=False)
    samples_dir_global = trans_vec_homo_batch(m =local2global_mat , v = rays_dir_local, is_vec=True)

    # samples_pts_global_normalized = 
    # samples_feature = sample_from_3dgrid(grid=voxel_grid_feature.unsqueeze(0), coordinates=p_loc_local_normalized.unsqueeze(0), grid_form_return=False,  mode= 'bilinear', padding_mode='border').squeeze(0)
    if True:
        samples_pts_invox = p_loc_local % 1

    return idx.reshape(-1), samples_pts_global, samples_pts_invox, samples_dir_global,samples_feature,samples_semantic

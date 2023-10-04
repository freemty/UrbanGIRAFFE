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
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from collections import OrderedDict

class NeRFRenderer(nn.Module):
    def __init__(self):
        super().__init__()
        self.ray_marcher = OrderedDict()


#--------------------------------Ray Sample----------------------------------------
    # def ray_sample(self, K, c2w):
    #     X, Y = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy')
    #     XYZ = np.concatenate((X[:, :, None], Y[:, :, None], np.ones_like(X[:, :, None]) * (1 - 2 * z_reverse)), axis=-1)
    #     XYZ = XYZ @ np.linalg.inv(K[:3, :3]).T
    #     XYZ = torch.concatenate([XYZ,np.ones_like(XYZ[...,0:1])], axis = -1)
    #     XYZ = XYZ @ c2w.T
    #     rays_p = XYZ.reshape(-1, 4)[...,:3]
    #     rays_o = c2w[:3, 3]
    #     rays_d =(rays_p - rays_o)
    #     a = rays_p[...,3]
    #     if True:
    #         rays_d = rays_d / np.linalg.norm(rays_d, axis=-1)[:, None]
    #     # For test
    #     if False:
    #         import imageio
    #         imageio.imsave('tmp/a2.jpg',abs(rays_d).reshape((H, W,3)))

    # return torch.cat((rays_o[None].repeat(len(rays_d), 0), rays_p, rays_d), axis=-1)
    
#--------------------------------Composition----------------------------------------
    def sort_samples(self , samples):
        all_sigmas, all_feats,all_depths, all_semantics = samples['sigma'], samples['feat'], samples['depth'], samples['semantic']
        _, indices = torch.sort(all_depths , dim=-1)
        all_depths = torch.gather(all_depths, -1, indices)
        all_sigmas = torch.gather(all_sigmas, -1, indices)
        all_semantics = torch.gather(all_semantics, -1, indices)
        all_feats = torch.gather(all_feats, -2, indices[:,:,:,None].expand(-1, -1, -1, all_feats.shape[-1]))
        
        sorted_samples = OrderedDict({
            'sigma' :all_sigmas,
            'feat' :all_feats,
            'depth' :all_depths,
            'semantic' :all_semantics
        })
        return sorted_samples
    
    def remove_empty_samples(self, samples):
        pass
    
    def composite_multi_fields(self,samples , mode = 'train'):
        sigma, feat, depth, semantic = samples['sigma'], samples['feat'], samples['depth'], samples['semantic']
        all_sigmas, all_feats, all_depths, all_semantics = [], [], [], []
        for i in sigma.keys():
            if i == 'obj':
                all_sigmas.append(rearrange(sigma[i], 'bs nbox hw p -> bs hw (nbox p)'))
                all_feats.append(rearrange(feat[i], 'bs nbox hw p f -> bs hw (nbox p) f'))
                all_depths.append(rearrange(depth[i], 'bs nbox hw p -> bs hw (nbox p)'))
                all_semantics.append(rearrange(semantic[i], 'bs nbox hw p -> bs hw (nbox p)'))
            else:
                all_sigmas.append(sigma[i])
                all_feats.append(feat[i])
                all_depths.append(depth[i])
                all_semantics.append(semantic[i])

        all_sigmas = torch.cat(all_sigmas, dim = -1)
        all_feats = torch.cat(all_feats, dim = -2)
        all_depths= torch.cat(all_depths, dim = -1)
        all_semantics= torch.cat(all_semantics, dim = -1)

        composite_samples = OrderedDict({
            'sigma': all_sigmas,
            'feat':all_feats,
            'semantic':all_semantics,
            'depth':all_depths,
        })

        # Only keep points that has sigma value
        # if mode == 'test':
        #     all_depths, all_sigmas, all_feats,all_semantics =  remove_empty_points(depth=all_depths, sigma=all_sigmas, feat=all_feats,semantic = all_semantics)
        composite_samples = self.sort_samples(composite_samples)

        return composite_samples

#--------------------------------Volume Render----------------------------------------
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
    
    def volume_render(self, samples, render_option):
        '''
        depth denote the distance between
        '''
        H, W, batch_size = render_option.H_ray, render_option.W_ray, render_option.batch_size
        # Get Volume Weights
        depth, sigma, feat, semantic = samples['depth'], samples['sigma'], samples['feat'], samples['semantic']
        rays_dir = samples['rays_dir']
        weights = self.calc_volume_weights(depth, rays_dir, sigma)
        feat_map = torch.sum(weights.unsqueeze(-1) * feat, dim=-2)
        # color_map = torch.sum(weights.unsqueeze(-1) * feat[...,:3], dim=-2)

        weights_depth = weights.clone()

        if render_option.render_sky:
                sky_idx = (semantic == 23)
                depth[sky_idx] = render_option.z_far 
        weights_depth = F.normalize(weights_depth, p = 1, dim= -1)
        depth_map = torch.sum(weights_depth.unsqueeze(-1) * depth.unsqueeze(-1), dim=-2)
        
        if render_option.is_debug:
            #! key pixel [59, 150]
            weights_depth_debug = weights_depth.detach().cpu().numpy().reshape((batch_size,H, W, -1))
            a = weights_depth.detach().cpu().numpy()
            # depth_map = torch.sum(weights.unsqueeze(-1) * depth.unsqueeze(-1), dim=-2)
            feat_map_debug = feat_map.detach().cpu().numpy().reshape((batch_size,H, W, -1))
            depth_debug = depth.detach().cpu().numpy().reshape((batch_size,H, W, -1))
            semantic_debug = semantic.detach().cpu().numpy().reshape((batch_size,H, W, -1))
            depth_map_debug = depth_map.detach().cpu().numpy().reshape((batch_size,H, W, -1))
            sigma_debug = sigma.detach().cpu().numpy().reshape((batch_size,H, W, -1))
            weights_debug =  weights.detach().cpu().numpy().reshape((batch_size,H, W, -1))

        depth_map = ((depth_map - 1e-3) / (render_option.z_far  - 1e-3)).clamp(0,1.)
        depth_map = depth_map.permute(0, 2, 1).reshape(
            batch_size, -1, H, W)
        feat_map = feat_map.permute(0, 2, 1).reshape(
            batch_size, -1, H, W)  # B x feat x h x w
        alpha_map = torch.sum(weights, dim = -1).reshape((batch_size, 1, H, W))

        output = OrderedDict({
            'feat_map' : feat_map,
            'depth_map' : depth_map,
            'alpha_map': alpha_map,
            'weights':weights
        })

        return output
    
    def forward(self, samples, render_option , is_compositional = True):
        if is_compositional:
            samples = self.composite_multi_fields(samples, render_option)

        render_results = self.volume_render(samples = samples, render_option = render_option)
        return render_results

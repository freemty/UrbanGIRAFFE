import tools.voxlib as voxlib
import torch

import torch
import numpy as np
import pickle as pkl
from lib.networks.volumetric_rendering import sample_from_3dgrid
import matplotlib.pyplot as plt

def build_rays(H, W, K, c2w = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])):
    assert c2w.shape == (4,4)

    X, Y = np.meshgrid(np.arange(W), np.arange(H))
    XYZ = np.concatenate((X[:, :, None], Y[:, :, None], np.ones_like(X[:, :, None])), axis=-1)
    XYZ = XYZ @ np.linalg.inv(K[:3, :3]).T
    XYZ = XYZ @ c2w[:3, :3].T
    rays_d = XYZ.reshape(-1, 3)
    rays_d = rays_d / np.linalg.norm(rays_d, axis=-1)[:, None]
    rays_o = c2w[:3, 3]
    return np.concatenate((rays_o[None].repeat(len(rays_d), 0), rays_d), axis=-1)



def trans_vec_homo(m, v, is_vec=False, normalize = True):
    r"""3-dimensional Homogeneous matrix and regular vector multiplication
    Convert v to homogeneous vector, perform M-V multiplication, and convert back
    Note that this function does not support autograd.

    Args:
        m (4 x 4 tensor): a homogeneous matrix
        v (3 tensor): a 3-d vector
        vec (bool): if true, v is direction. Otherwise v is point
    """
    if is_vec:
        v = torch.tensor([v[0], v[1], v[2], 0], dtype=v.dtype)
    else:
        v = torch.tensor([v[0], v[1], v[2], 1], dtype=v.dtype)
    v = torch.mv(m, v)
    if not is_vec:
        v = v / v[3]
    elif normalize:
        v = v[:3] / torch.sqrt(v[0] * v[0]+ v[1]* v[1]+ v[2] * v[2])

    v = v[:3]
    return v

def trans_vec_homo_batch(m, v, is_vec=False, normalize = True):
    r"""3-dimensional Homogeneous matrix and regular vector multiplication
    Convert v to homogeneous vector, perform M-V multiplication, and convert back
    Note that this function does not support autograd.

    Args:
        m (4 x 4 tensor): a homogeneous matrix
        v (3 tensor): a 3-d vector
        vec (bool): if true, v is direction. Otherwise v is point
    """
    m = m.to(v.device).to(torch.float32)
    v = v.to(torch.float32)
    batch_size = v.shape[0]
    if is_vec:
        v = torch.cat((v, torch.zeros_like(v[:,0:1])), dim =1)
    else:
        v =torch.cat((v, torch.ones_like(v[:,0:1])), dim =1)
    v = (m @ v.T).T 
    if not is_vec:
        v = v[:] / v[:,3]

    v = v[:,:3]
    return v


def ray_voxel_intersection_sampling(camera_intrinsic, camera_pose, voxel_seamntic, voxel_range = [[-32,32], [-14,2], [0,64]], cam_res = (94,352), max_samples = 12):
    '''
    voxel_seamntic: (H,W,D)
    voxel_range: ((x_start, x_start),(x_start, x_start),(y_start, y_start))
    '''

    voxel_seamntic = torch.flip(voxel_seamntic, dims = [0])
    voxel_features = 0
    voxel_seamntic_debug = voxel_seamntic.cpu().numpy()

    rays_dir_global = torch.tensor(build_rays(H = 94, W = 352, K = camera_intrinsic.cpu().numpy(), c2w = camera_pose.cpu().numpy())[...,3:]).cuda()

    # Step1 prase camera args from intrinsic matrix 
    cam_f = camera_intrinsic[0,0] #Camera focal length (in pixels).
    cam_c = (camera_intrinsic[1,2], camera_intrinsic[0,2]) #Camera optical center (Y,X).

    # Step2 transform from global to local coordinate
    Y_sample_num, X_sample_num, Z_sample_num =  voxel_seamntic.shape[0], voxel_seamntic.shape[1], voxel_seamntic.shape[2]
    X_scale = (voxel_range[0][1] - voxel_range[0][0])/X_sample_num
    Y_scale=  (voxel_range[1][1] - voxel_range[1][0])/Y_sample_num
    Z_scale = (voxel_range[2][1] - voxel_range[2][0])/Z_sample_num
    global2local_mat = torch.inverse(torch.tensor(
        [[0,X_scale,0,voxel_range[0][0]],
        [-Y_scale,0,0,voxel_range[1][1]],
        [0,0,Z_scale,voxel_range[2][0]],
        [0,0,0,1]])
    )
    # a =  trans_vec_homo(m = global2local_mat, v = torch.tensor((-32.,-14,0)), is_vec=False)
    # b =  trans_vec_homo(m = global2local_mat, v = torch.tensor((32.,2,64)), is_vec=False)

    rays_dir_local = trans_vec_homo_batch(m = global2local_mat, v = rays_dir_global, is_vec= True).reshape((94,352,3))

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
    camera_ori_global = torch.tensor([camera_pose[0,3],camera_pose[1,3],camera_pose[2,3],1])

    cam_up_local =  trans_vec_homo(m = global2local_mat, v = camera_up_global, is_vec=True).cuda()
    cam_dir_local = trans_vec_homo(m = global2local_mat, v = camera_dir_global, is_vec=True).cuda()
    cam_ori_local = trans_vec_homo(m = global2local_mat, v = camera_ori_global, is_vec=False).cuda() 


    # Step3 Utilize cancraft's ray_voxel_intersection kernel
    rays_dir_local = rays_dir_local.contiguous()
    # print( b.stride())
    voxel_id, depth2, raydirs = voxlib.ray_voxel_intersection_perspective(
                        voxel_seamntic, 
                        rays_dir_local,
                        cam_ori_local, cam_dir_local, cam_up_local, 
                        cam_f, cam_c, cam_res,
                        max_samples)

    a  = voxel_id.cpu().numpy()
    d = raydirs.cpu().numpy()

    plt.imsave('tmp/aaaa.jpg', a[:,:,0,0])

    '''
    In cuda kernel use flip height
    float ndc_imcoords[2];
    ndc_imcoords[0] = p.cam_c[0] - (float)img_coords[0]; // Flip height
    ndc_imcoords[1] = (float)img_coords[1] - p.cam_c[1];
    '''

    # Step4 Sampleing based on intersection

    b = depth2.cpu().numpy()
    
    voxel_id = voxel_id.squeeze(-1)
    depth = depth2.mean(dim = 0).squeeze(-1)
    # a = torch.argwhere(depth2.mean(dim = 0) > 64 ).cpu().numpy() 
    idx = (voxel_id != 0) * (depth < 64)
    depth_valid =depth[idx].unsqueeze(-1)
    # depth_debug = depth[depth > 64].cpu().numpy()
    rays_dir_local_ = rays_dir_local.reshape((94,352,1,3)).repeat(1,1,max_samples,1)[idx]
    #.permute(0,1,3,2)[voxel_id != 0]
    rays_dir_local_ = rays_dir_local_ / rays_dir_local_[...,-1,None]
    p_loc_local = cam_ori_local + (depth_valid - cam_ori_local[2]) * rays_dir_local_
    p_loc_local_debug = p_loc_local.cpu().numpy()

    p_loc_local_normalized = 2 * (p_loc_local / torch.tensor([64,64,64]).cuda() - 0.5)
    p_semantic = voxel_id[idx]
    feature_grid = torch.meshgrid([torch.linspace(0,X_sample_num,X_sample_num),torch.linspace(0,Y_sample_num,Y_sample_num),torch.linspace(0,Z_sample_num,Z_sample_num)], indexing='xy')

    feature_grid = torch.stack(feature_grid, dim = -1).unsqueeze(0).cuda()
    p_loc_local_normalized = p_loc_local_normalized.unsqueeze(0).cuda()
    sampled_feature = sample_from_3dgrid(grid=feature_grid, coordinates=p_loc_local_normalized, grid_form_return=False, padding_mode= 'border')
    a = (sampled_feature.squeeze(0) -p_loc_local).mean(dim = -1).cpu().numpy()
    b =  np.argwhere(abs(a) > 0.5)

    return 0



if __name__ == '__main__':
    with open('tmp/0000000812.pkl', 'rb') as f:
        voxel_seamntic = torch.tensor(pkl.load(f)).reshape((64,64,64)).cuda().to(torch.int32)

    camera_intrinsic = torch.tensor([
        [138.13856525,   0.        , 170.51236325],
        [  0.        , 138.13856525,  59.69238725],
        [0., 0., 1.]]).cuda()
    camera_pose_global = torch.tensor([
        [1., 0., 0., 0.],
        [ 0. ,  0.99619492,  0.08715318, -1.55],
        [ 0. , -0.08715318,  0.99619492,  0.  ],
        [0., 0., 0., 1.]]).cuda()

    
    ray_voxel_intersection_sampling(
        camera_intrinsic=camera_intrinsic, 
        camera_pose = camera_pose_global,
        voxel_seamntic=voxel_seamntic)

import numpy as np



def convert2defaultvox(kitti_voxel, is_loc = False):
    flag = 0
    if len(kitti_voxel.shape) == 3:
        kitti_voxel = kitti_voxel[:,:,:,None]
        flag = 1
    N2S_mat = np.array(
                    [[0,-1,0,0],
                    [0,0,-1,0],
                    [1,0,0,0],
                    [0,0,0,1]])# Transform point from normal coordinate to shit coordinate
    default_voxel = kitti_voxel.copy()
    W, L, H, C = kitti_voxel.shape
    default_voxel = default_voxel.transpose((2,0,1,3)) #  H W L C
    default_voxel = np.flip(default_voxel, 0) # flip H axis
    default_voxel = np.flip(default_voxel, 1) # flip W axis
    if is_loc:
        default_voxel = default_voxel @  N2S_mat[:3,:3].T
    default_voxel = np.ascontiguousarray(default_voxel)

    if flag == 1:
        default_voxel = default_voxel[...,0]
    return default_voxel


def init_voxel_grid_kitti(p_LHW = np.array([64,64,64])):
    """
    X+ -> forward (L+)
    Y+ -> left (W+)
    Z+ -> up (H+)
    with the increase of index, voxel (right, down, back) -> (left, uo, forward) in real world
    """
    # KITTI-semantic grid in normal coordinate
    H, W, L = 16, 64, 64
    target_scene_size = np.array((64, 64, 16))
    target_voxel_origin = np.array((0, -32, -2))
    point_num_L, point_num_H, point_num_W, = p_LHW
    x_min, x_max = target_voxel_origin[0], target_voxel_origin[0] + target_scene_size[0]
    y_min, y_max = target_voxel_origin[1], target_voxel_origin[1] + target_scene_size[1]
    z_min, z_max = target_voxel_origin[2], target_voxel_origin[2] + target_scene_size[2]
    bounds = np.array((x_max, x_min, y_max, y_min, z_max, z_min))
    point_num_X, point_num_Y, point_num_Z, = point_num_L, point_num_W, point_num_H
    vertices_gridx,  vertices_gridy, vertices_gridz= np.meshgrid(np.linspace(x_min, x_max, point_num_X), np.linspace(y_min, y_max, point_num_Y), np.linspace(z_min, z_max, point_num_Z), indexing='xy') #! [Y, X, Z, 3] = [W, L, H, 3] 

    stuff_loc_gird = np.concatenate((vertices_gridx[:,:,:,None], vertices_gridy[:,:,:,None], vertices_gridz[:,:,:,None]), axis = -1)
    return stuff_loc_gird #! shape [W, L, H, 3]

def init_voxel_grid_default(p_LHW = np.array([64,64,64])):
    """
    default coordinate which yyb use
    X+ -> right ()
    Y+ -> down ()
    Z+ -> forward ()
    # with the increase of index, voxel (right, up, forward) -> (left, uo, forward) in real world
    return with size (H, W, L, 3) = (Y_b, X_b, Z_b, 3)
    """
    target_scene_size = np.array((64, 16, 64))
    target_voxel_origin = np.array((-32, -14, 0))
    point_num_L, point_num_H, point_num_W, = p_LHW
    x_min, x_max = target_voxel_origin[0], target_voxel_origin[0] + target_scene_size[0]
    y_min, y_max = target_voxel_origin[1], target_voxel_origin[1] + target_scene_size[1]
    z_min, z_max = target_voxel_origin[2], target_voxel_origin[2] + target_scene_size[2]
    bounds = np.array((x_max, x_min, y_max, y_min, z_max, z_min))
    point_num_X, point_num_Y, point_num_Z, = point_num_W, point_num_H, point_num_L
    vertices_gridx,  vertices_gridy, vertices_gridz= np.meshgrid(np.linspace(x_min, x_max, point_num_X), np.linspace(y_min, y_max, point_num_Y), np.linspace(z_min, z_max, point_num_Z), indexing='xy') #! [Y_shit, X_shit, Z_shit, 3]

    stuff_loc_gird = np.concatenate((vertices_gridx[:,:,:,None], vertices_gridy[:,:,:,None], vertices_gridz[:,:,:,None]), axis = -1)
    return stuff_loc_gird  #! [Y_shit, X_shit, Z_shit, 3] = [H, W, L, 3] 



N2S_mat = np.array(
                    [[0,-1,0,0],
                    [0,0,-1,0],
                    [1,0,0,0],
                    [0,0,0,1]])# Transform point from normal coordinate to shit coordinate

p_LHW = (61,62,63)

# L H W
loc_grid_normal = init_voxel_grid_kitti(p_LHW) # (WLH 3)
loc_grid_default = init_voxel_grid_default(p_LHW) # (HWL 3)

if __name__ == '__main__':

    if False:
        loc_grid_normal_ = loc_grid_normal.transpose((1,2,0,3))
        loc_grid_default_ = loc_grid_default.transpose((2,0,1,3))
        loc_grid_default_ = loc_grid_default_.reshape((-1,3))
        # LHW
        loc_grid_normal_ = np.flip(loc_grid_normal_, axis = 1)
        loc_grid_normal_ = np.flip(loc_grid_normal_, axis = 2)
        loc_grid_normal_ = loc_grid_normal_.reshape((-1,3))
        loc_grid_normal_ = (N2S_mat[:3,:3] @ loc_grid_normal_.T).T
        error = np.sum(loc_grid_normal_ - loc_grid_default_, axis=-1)
    else:
        default_grid_convert = convert2defaultvox(loc_grid_normal)
        error = default_grid_convert - loc_grid_default
    assert np.max(error) < 1e-3
    print('done')
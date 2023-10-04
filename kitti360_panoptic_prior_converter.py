
import os 
# Print the current working directory
print("Current working directory: {0}".format(os.getcwd()))
# # Change the current working directory
# os.chdir('../')
# # Print the current working directory
# print("Current working directory: {0}".format(os.getcwd()))
from collections import OrderedDict
import pickle as pkl 
import numpy as np
from  math import sin, cos
import torch
import torch.nn.functional as F
from tools.kitti360Scripts.helpers.labels import name2label

stuff_name_list = {
    # Major
    'vegetation',
    'terrain',
    'ground',
    'road', 
    'sidewalk',
    'parking',
    'rail track',
    'building',
    'gate',
    'garage', 
    'bridge',
    'tunnel',
    # Vehicle
    'wall',
    'car',
    'truck',
    'train','caravan',
    'bus',
    # Minor
    'trailer',
    'fence',
    'guard rail',
    'trash bin',
    'box',
    'lamp',
    'smallpole',
    'polegroup',
    'stop',
    'pole',
    'traffic sign',
    'traffic light'
}


pi = 3.1415

# Transform utils

def create_R(rotate = (0,0,0), scale = (1,1,1)):
    """ Build R matrix from rotate(eular angle) and scale(xyz)
    Args:
        rotate: eular angle (alpha, beta, gamma)
        scale: (s_x, s_y, s_z)
    Return:
        R
    """
    alpha, beta, gamma = rotate[0], rotate[1], rotate[2]

    rx_mat = np.array([
        [1,0,0],
        [0,cos(alpha),-sin(alpha)],
        [0,sin(alpha),cos(alpha)],   
    ])
    ry_mat = np.array([
        [cos(beta),0,sin(beta)],
        [0,1,0],
        [-sin(beta),0,cos(beta)],
    ])

    rz_mat = np.array([
        [cos(gamma),-sin(gamma),0],
        [sin(gamma),cos(gamma),0],
        [0,0,1],
    ])

    scale_mat = np.array([
        [scale[0],0,0],
        [0,scale[1],0],
        [0,0,scale[2]],
    ])

    R = np.matmul(rz_mat, ry_mat).dot(rx_mat).dot(scale_mat)

    return R


def create_c2w(cam_R_world, cam_T_world, cam_type = 'opencv'):
    
    c2w = np.eye(4)
    c2w[:3,:3] = cam_R_world
    # cam_tr_world[:3,:3] = cam_R_world
    # cam_tr_world[:3,3] = cam_T_world
    if cam_type == 'opencv':
        # rectmat: Transfrom points from camera coodriante to world coordinate (using world basis to describe cmaera basis)
        # > https://zhuanlan.zhihu.com/p/404773542
        # > https://www.zhihu.com/question/407150749
        rectmat = np.array([
            [0,0,1,0],
            [-1,0,0,0],
            [0,-1,0,0],
            [0,0,0,1]])
    elif cam_type == 'opengl':
        rectmat = np.array([
            [0,0,-1,0],
            [-1,0,0,0],
            [0,1,0,0],
            [0,0,0,1]
        ])
    else:
        raise TypeError
    
    c2w = c2w @ rectmat
    c2w[:3,3] += cam_T_world

    return c2w


def convert_legacy_layout(layout_path):
    with open(layout_path, 'rb') as fp:
        layout_raw = pkl.load(fp)
    layout = OrderedDict()
    
    opencv2world = np.array([
            [0,0,1,0],
            [-1,0,0,0],
            [0,-1,0,0],
            [0,0,0,1]])
    for globalId,obj in layout_raw.items():
        # skip dynamic objects

        tr = np.eye(4)
        tr[:3,:3], tr[:3,3] = obj.R, obj.T
        tr = tr @ opencv2world.T

        vertices = np.concatenate((obj.vertices, np.ones_like(obj.vertices[:,0:1])), axis=-1)
        vertices = vertices @ opencv2world.T

        layout[globalId] = OrderedDict({
            'R' : tr[:3,:3],
            'T' : tr[:3,3],
            'vertices' : vertices[:,:3],
            'faces' : obj.faces,
            'name' : obj.name,
            'color' : np.array(name2label[obj.name].color / 255.),
            'semanticId' : obj.semanticId,
            'instanceId' : obj.instanceId,
            'annotationId' : obj.annotationId})
            
    return layout

def convert_legacy_stuff_semantic(semantic_voxel_path,semantic_list):
    '''
    Load old version seamntic voxel
    '''
    with open(semantic_voxel_path, 'rb') as fp:
        stuff_semantic_idx= pkl.load(fp)

    point_num = (scene_size / vox_size).astype(int) # X, Y, Z
    H, W, D = 64, 64, 64
    H_o, W_o, D_o = point_num[2], point_num[1], point_num[0]
    stuff_semantic = np.zeros(H * W * D)
    for s in semantic_list:
        if stuff_semantic_idx[s].shape[0] == 0:
            continue
        stuff_semantic[stuff_semantic_idx[s]] = name2label[s].id
        
    semantic_voxel = stuff_semantic.reshape(H, W, D)
    semantic_voxel = semantic_voxel.transpose(1, 2, 0) # W D H
    semantic_voxel = np.flip(semantic_voxel, axis=0)
    semantic_voxel = np.flip(semantic_voxel, axis=2)
    semantic_voxel = np.ascontiguousarray(semantic_voxel)

    semantic_voxel = F.interpolate(torch.tensor(semantic_voxel, dtype=torch.uint8)[None,None,:,:,:], (W_o, D_o, H_o), mode = 'nearest')[0,0].numpy()


    vertices_gridx,  vertices_gridy, vertices_gridz= np.meshgrid(np.linspace(0, scene_size[0], point_num[0]), np.linspace(0, scene_size[1], point_num[1]), np.linspace(0, scene_size[2], point_num[2]), indexing='xy') 
    loc_voxel = np.concatenate((vertices_gridx[:,:,:,None], vertices_gridy[:,:,:,None], vertices_gridz[:,:,:,None]), axis = -1)
    loc_voxel += vox_origin[None,None,None]

    return semantic_voxel, loc_voxel



if __name__ == '__main__':
    scene_size = np.array((64., 64, 16))
    vox_size = np.array((0.25, 0.25,0.25))
    vox_origin = np.array([0, -32, -2])
    seq = 0
    sequence = ''

    H, W = 376, 1408
    K = np.array([[552.554261,   0.      , 682.049453],
                    [  0.      , 552.554261, 238.769549],
                    [0., 0., 1.]])
    cam_T = np.array((0,0,1.55))
    cam_R = create_R((0,5 / 180 * pi,0,))
    c2w = create_c2w(cam_R_world=cam_R, cam_T_world=cam_T, cam_type='opencv')
    c2w_GL = create_c2w(cam_R_world=cam_R, cam_T_world=cam_T, cam_type='opengl')
    w2c = np.linalg.inv(c2w)


    # Load Instance Layout 
    data_root = 'data/kitti-360'
    frame_id = 7337
    legacy_layout_path = os.path.join(data_root, 'layout', '2013_05_28_drive_0000_sync', '%010d.pkl'%frame_id)
    legacy_semantic_voxel_path = os.path.join(data_root, 'voxel', '2013_05_28_drive_0000_sync', '%010d.pkl'%frame_id)


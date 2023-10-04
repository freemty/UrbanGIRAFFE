# import torch
import numpy as np
import open3d as o3d
import trimesh
import pickle as pkl
import time
import os
import  re
import json


# AABB
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



import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--seq_num", default=0, type=int)
args = parser.parse_args()
seq_num = args.seq_num
N = 1000
data_root = '/data/ybyang/clevrtex_train/%d'%seq_num
frame_num =  os.listdir(os.path.join(data_root,'Depth'))
voxel_dir = os.path.join(data_root,'Voxel')
if not os.path.exists(voxel_dir):
    os.makedirs(voxel_dir)
for i in range(len(frame_num)):
    frame_id = i
    layout_path = os.path.join(data_root, 'Metadata', 'CLEVRTEX_train_%06d.json'%(frame_id + N* seq_num))
    mesh_path = os.path.join(data_root, 'Obj', 'CLEVRTEX_train_%06d.obj'%(frame_id + N* seq_num) )
    rgb_path = os.path.join(data_root, 'RGB', 'CLEVRTEX_train_%06d.png'%(frame_id + N* seq_num) )
    mask_path = os.path.join(data_root, 'Mask', 'CLEVRTEX_train_%06d_mask.png'%(frame_id + N* seq_num) )
    depth_path = os.path.join(data_root, 'Depth', 'CLEVRTEX_train_%06d_depth_0001.png'%(frame_id + N* seq_num) )
    voxel_path = os.path.join(data_root, 'Voxel', 'CLEVRTEX_train_%06d.pkl'%(frame_id + N* seq_num) )

    if os.path.exists(voxel_path):
        continue

    max_x,min_x = 8,-8
    max_y,min_y = 4,0
    max_z,min_z = 8,-8


    # pcd = trimesh.load_path(pcd_path)
    scene = trimesh.load_mesh(mesh_path)
    scene_mesh = {i: scene.geometry[i] for i in scene.geometry}
    print(scene_mesh)
    # print(np.asarray(pcd.points))
    point_num_Y, point_num_X, point_num_Z, =64, 64, 64
    vertices_gridx, vertices_gridy, vertices_gridz= np.meshgrid(np.linspace(min_x, max_x, point_num_X), np.linspace(min_y, max_y, point_num_Y), np.linspace(min_z, max_z, point_num_Z))
    vertices_grid_pts = np.concatenate((vertices_gridx[:,:,:,None], vertices_gridy[:,:,:,None], vertices_gridz[:,:,:,None]), axis = -1).reshape((-1,3))

    semantic_num_list = {'ground' : 0,'wall' : 0, 'object': 0}
    name2id =  {'ground' : 0, 'wall' : 1,'object': 2}
    name2color = {'ground' : (128, 64,128), 'wall' : (102,102,156),'object': (70, 70, 70), 'metal': (  0,  0,142), 'cylinder': (107,142, 35), 'ball': (0, 60,100)}
    vertices_seamntic,vertices_instance  = {}, {}
    for name in name2id:
            vertices_seamntic[name] = []
            vertices_instance[name] = []

    meshs = []
    for name, obj in scene_mesh.items():
        meshs.append(obj.as_open3d)
        t = time.time()
        if name in ['Rubber', 'Rubber_2']:
            obj.name = 'wall'
        elif re.search('Metal',name) != None:
            obj.name = 'object'
        elif re.search('Rubber', name) != None:
            obj.name = 'object'
        elif re.search('cylinder', name) != None:
            obj.name = 'object'
        elif re.search('TabulaRasa', name) != None:
            obj.name = 'ground'
            s = np.argwhere(vertices_grid_pts[:,1] <= 0.05)
            vertices_seamntic[obj.name] += list(s)
            vertices_instance[obj.name] += [0] * len(vertices_seamntic[obj.name])
            continue
        else:
            continue
        obj.instanceId = semantic_num_list[obj.name]
        max_xyz = np.max(obj.vertices, axis = 0)
        min_xyz = np.min(obj.vertices, axis = 0)
        bounds = np.stack([min_xyz, max_xyz], axis=0)
        bbx_mask = get_near_far(bounds, ray_o=vertices_grid_pts, ray_d = np.ones_like(vertices_grid_pts), return_mask = True)
        a = np.sum(bbx_mask)
        if np.sum(bbx_mask) > 0:
            candidate_idx = np.argwhere(bbx_mask == True).reshape(-1)
            candidate_vertices = vertices_grid_pts[candidate_idx]
            obj_mesh_tri = trimesh.Trimesh(vertices=obj.vertices, faces=obj.faces)
            _, index_rays, _ = obj_mesh_tri.ray.intersects_location(ray_origins=candidate_vertices, ray_directions=np.ones_like(candidate_vertices)) 
            intersect_rays, intersect_counts = np.unique(index_rays, return_counts=True)
            valid_idx = candidate_idx[intersect_rays[intersect_counts %2 != 0]]
            vertices_seamntic[obj.name] += list(valid_idx)
            vertices_instance[obj.name] += [obj.instanceId] * len(valid_idx)
                # vertices_seamntic_grid[valid_idx] = name2label[obj.name].id 
        semantic_num_list[obj.name] += 1


    for k in vertices_seamntic:
        vertices_seamntic[k] = np.array(vertices_seamntic[k])
        vertices_instance[k] = np.array(vertices_instance[k])
    voxel = {
        'semantic':vertices_seamntic,
        'instance':vertices_instance,
        '(H,W,L)':(point_num_Y, point_num_X, point_num_Z),
        'X_range':(max_x,min_x ),
        'Y_range':(max_y,min_y ),
        'Z_range':(max_z,min_z ),

    }

    with open(voxel_path,'wb+') as f:
        pkl.dump(voxel, f)

    print('%d Done'%i)

import numpy as np
import pickle as pkl
import os
import torch
import torch.nn.functional as F
import open3d as o3d
from tools.kitti360Scripts.helpers.labels import labels, name2label

import matplotlib.pyplot as plt

learning_map_inv={
# 0: "unlabeled", 
#    1: "car",
#   2: "bicycle",
#   3:  "motorcycle",
#   4: "truck",
#   5:  "other-vehicle",
#    6: "person",
#   7: "bicyclist",
#   8: "motorcyclist",
  9:  "road",
  10: "parking",
  11: "sidewalk",
  12: "ground",
  13: "building",
  14: "fence",
  15: "vegetation",
#    16:  "trunk",
  17: "terrain",
#    18:"pole",
#    19:"traffic-sign"
}

learning_map = {learning_map_inv[k]:k for k in learning_map_inv}
# name2semantic_kitti = {id2name_semantic_kitti : k for k in id2name_semantic_kitti }

stuff_semantic_list_render =  [
    'vegetation',
    'terrain','ground','road', 'sidewalk','parking',
    'building',
     'fence',
    ]

def pad_col(voxel, u, v):
    u_, v_ = u, v
    max_n = 0
    for x in [max(u-1, 0), u, min(u+1,63)]:
        for y in [max(v-1,0), v, min(v+1,63)]:
            if sum(voxel[x, y]) > max_n:
                max_n = sum(voxel[x, y]) 
                u_, v_ = x, y
    col_ = voxel[u_,v_]
    return col_


def higher_stuff(voxel, h_threshold1 = 10,  h_threshold2 = 8):
    target_stuff = np.array([name2label['vegetation'].id, name2label['building'].id])
    Y_num , X_num, Z_num = voxel.shape[0], voxel.shape[1], voxel.shape[2]
    for u in range(X_num):
        for v in range(Y_num):
            col_i = voxel[u, v]
            if sum(col_i) == 0:
                voxel[u, v] = pad_col(voxel, u,v)
            if len(np.intersect1d(col_i, target_stuff)) == 0 :
                continue
            z = col_i != 0
            max_h = max(np.argwhere(col_i != 0))[0]
            if max_h >= h_threshold2:
                voxel[u, v][max_h : max_h * 3] = col_i[max_h]
            # elif max_h >= h_threshold1:
            #     voxel[u, v][max_h : max_h * 2] = col_i[max_h]
    return voxel

def kitti2yybvox_(raw_voxel):
    '''
    transform voxel world from sematic kitti to urban giraffe format
    '''

    raw_scene_size = np.array((51.2, 51.2, 6.4))
    raw_vox_origin = np.array([0, -25.6, -2]) # x, y, z
    raw_voxel_size = np.array((0.2,0.2,0.2))  # 0.2m
    raw_res = raw_scene_size / raw_voxel_size

    target_scene_size = np.array((64, 64, 16))
    target_voxel_size = np.array((1, 1, 0.25)) # X, Y,Z
    target_voxel_origin = np.array([0, -32, -2])
    target_res = raw_scene_size / target_voxel_size


    stuff_semantic = raw_voxel
        # x_max, x_min, y_max, y_min, z_max, z_min = 32, -32, 2, -14, 64, 0
    
    x_min, x_max = target_voxel_origin[0], target_voxel_origin[0] + target_scene_size[0]
    y_min, y_max = target_voxel_origin[1], target_voxel_origin[1] + target_scene_size[1]
    z_min, z_max = target_voxel_origin[2], target_voxel_origin[2] + target_scene_size[2]
    bounds = np.array((x_max, x_min, y_max, y_min, z_max, z_min))
    point_num_X, point_num_Y, point_num_Z, = 64, 64, 64
    vertices_gridx,  vertices_gridy, vertices_gridz= np.meshgrid(np.linspace(x_min, x_max, point_num_X), np.linspace(y_min, y_max, point_num_Y), np.linspace(z_min, z_max, point_num_Z), indexing='xy') 

    stuff_loc_gird = np.concatenate((vertices_gridx[:,:,:,None], vertices_gridy[:,:,:,None], vertices_gridz[:,:,:,None]), axis = -1).transpose((1,0,2,3)) 
    #! shape [P_X, P_Y, P_Z, 3]
    #  H

    # c2w = c2w_raw
    stuff_semantic = stuff_semantic.astype(np.uint8)
    stuff_semantic = F.interpolate(torch.tensor(stuff_semantic, dtype=torch.uint8)[None,None,:,:,:], (52, 52, 26), mode = 'nearest')[0,0].numpy()

    t = np.zeros((64,64,64))
    
    # mirror Padding X, Y axis to target size
    stuff_semantic = np.pad(stuff_semantic,((0,12),(6,6),(0,0)),'edge')
    t[:,:,0:26] = stuff_semantic 
    # a = np.linalg.inv(raw_data['T_velo_2_cam'])
    stuff_semantic = np.zeros((64,64,64))

    for i in learning_map_inv:
        stuff_semantic[t == i] = name2label[learning_map_inv[i]].id

    stuff_semantic = higher_stuff(stuff_semantic)

    return stuff_semantic, stuff_loc_gird, bounds



def prepare_seq(seq_id):
    seq_num = '%02d'%seq_id
    data_root = '/data/ybyang/Semantic-kitti'
    image_dir = os.path.join(data_root, 'dataset', 'sequences', seq_num, 'image_2')
    voxel_dir = os.path.join(data_root, 'monoscene_pred', seq_num)

    frame_id = [int(n[:-4]) for n in os.listdir(voxel_dir)]
    frame_id.sort()
    target_dir = '/data/ybyang/semantic-kitti'
    # if not os.path.exists(target_dir):
    # os.makedirs(target_dir)
    os.makedirs(os.path.join(target_dir,'data_3d_voxel', seq_num), exist_ok= True)
    # os.makedirs(os.path.join(target_dir,'data_2d_raw', seq_num,'image_2'), exist_ok= True)
    os.makedirs(os.path.join(target_dir,'data_2d_raw', seq_num, 'kitti', 'testing','image_2'  ), exist_ok= True)
    for idx in frame_id:
        voxel_path = os.path.join(voxel_dir, '%06d.pkl'%idx)
        img_path = os.path.join(image_dir, '%06d.png'%idx)
        voxel_target = os.path.join(target_dir,'data_3d_voxel', seq_num , '%06d.pkl'%idx)
        img_target = os.path.join(target_dir,'data_2d_raw', seq_num,'kitti', 'testing', 'image_2', '%06d.png'%idx)
        with open(voxel_path, 'rb+') as f:
            raw_data = pkl.load(f)

        raw_voxel = raw_data['y_pred']
        stuff_semantic = raw_data['y_pred']
        valid_semantic = np.array(list(learning_map_inv.keys()))
        stuff_semantic[np.isin(raw_data['y_pred'], valid_semantic) == 0] = 0
        # stuff_semantic = raw_data['y_pred']
        K = raw_data['cam_k']
        c2w_raw = np.linalg.inv(raw_data['T_velo_2_cam'])
        c2w = c2w_raw
        stuff_semantic, stuff_loc_gird, bounds = kitti2yybvox_(raw_voxel)    

        with open(voxel_target, 'wb+') as fp:
            pkl.dump({"HW":(370,1226), "K":K, 
                    "c2w": c2w , 
                    'T_velo_2_cam':raw_data['T_velo_2_cam'],
                    "stuff_semantic_voxel": stuff_semantic,
                    "loc_voxel": stuff_loc_gird,},  fp)
            rgb_img = plt.imread(img_path)
            plt.imsave(img_target,rgb_img)

        print('Start frame%06d'%idx)
        visible_3d = False
        if False:
            geo_group = []
            geo_group += vis_voxel_world_o3d(vertices_grid_pts= stuff_loc_gird, vertices_seamntic= stuff_semantic, stuff_semantic_list =stuff_semantic_list_render)
            geo_group += vis_world_bounds_o3d(bounds=bounds)
            # geo_group += vis_bbox_layout_o3d(bbox_tr=bbox_tr, bbox_semantic=bbox_semantic)
            geo_group += vis_camera_o3d(instrinsic=K, extrinsic=np.linalg.inv(c2w), z_revrse=False)
            vis = costum_visualizer_o3d(geo_group=geo_group, set_camera=True, instrinsic=K, extrinsic=np.linalg.inv(c2w), visible = visible_3d)
            # vis.capture_screen_image(os.path.join('tmp', '%06d'%idx + '.png'))
            project_img = np.asarray(vis.capture_screen_float_buffer())



            rgb_img = plt.imread(img_path)
            H, W = project_img.shape[0], project_img.shape[1]
            img = (project_img[:H,:W] + rgb_img[:H, :W]).clip(0,1)
            if visible_3d:
                    vis.run()
            else:
                plt.imshow(img)
                plt.imsave('tmp/vis/%06d.png'%idx,img)

            del vis

            print('a')





if __name__ == "__main__":
    seq_list = [3,4]
    for seq_num in seq_list:
        prepare_seq(seq_num)

# python gan_data_prepare.py --seq_num 3 --frame_start 800 --frame_num 100 > 8.log 2>&1&
# python gan_data_prepare.py --seq_num 3 --frame_start 900 --frame_num 100 > 9.log 2>&1&
# python gan_data_prepare.py --seq_num 3 --frame_start 600 --frame_num 100 > 6.log 2>&1&
# python gan_data_prepare.py --seq_num 3 --frame_start 500 --frame_num 100 > 5.log 2>&1&
# python gan_data_prepare.py --seq_num 3 --frame_start 400 --frame_num 100 > 4.log 2>&1&
# python gan_data_prepare.py --seq_num 3 --frame_start 300 --frame_num 100 > 3.log 2>&1&
# python gan_data_prepare.py --seq_num 3 --frame_start 200 --frame_num 100 > 2.log 2>&1&
# python gan_data_prepare.py --seq_num 3 --frame_start 100 --frame_num 100 > 1.log 2>&1&
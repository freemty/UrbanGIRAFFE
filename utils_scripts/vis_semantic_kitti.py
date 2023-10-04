import numpy as np
import pickle as pkl
import os
import torch
import torch.nn.functional as F
import open3d as o3d
from tools.kitti360Scripts.helpers.labels import name2label
from utils_scripts.vis_cam import get_camera_frustum, frustums2lineset

def costum_visualizer_o3d(geo_group, set_camera = True,  instrinsic =None, extrinsic = None):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1202, height=366, visible= False)
    for g in geo_group:
        vis.add_geometry(g)
    if True:
        instrinsic[0,2] = int(instrinsic[0,2]) - 0.5
        instrinsic[1,2] = int(instrinsic[1,2]) - 0.5
        ctr = vis.get_view_control()
        cam = ctr.convert_to_pinhole_camera_parameters()
        cam.extrinsic = extrinsic
        cam.intrinsic.intrinsic_matrix = instrinsic
        ctr.convert_from_pinhole_camera_parameters(cam)

    rd = vis.get_render_option()
    rd.light_on = False
    rd.background_color = np.array([1,1,1])
    vis.poll_events()
    vis.update_renderer()
    return vis

def vis_world_bounds_o3d(bounds, vis = False):
    x_max, x_min, y_max, y_min, z_max, z_min = bounds.ravel()
    mark_group = []
    mark_group += [o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[x_max, 0, z_max])]
    mark_group += [o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[x_max, y_max,z_min])]
    mark_group += [o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[x_min, y_max,z_max])]
    mark_group += [o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[x_min, y_max,z_min])]
    mark_group += [o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[x_max, y_max, z_max])]
    mark_group += [o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[x_max, y_min,z_min])]
    mark_group += [o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[x_min, y_min,z_max])]
    mark_group += [o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[x_min, y_min,z_min])]
    mark_group += [o3d.geometry.TriangleMesh.create_coordinate_frame(size=3, origin=[0., 0,0])]

    if vis:
        o3d.visualization.draw_geometries(mark_group)
    return mark_group

def vis_voxel_world_o3d(vertices_seamntic,vertices_grid_pts, stuff_semantic_list = ['ground','wall', 'object'], vis = False):
    pt_group = []
    # for k in range(50):
    #     valid_idx = (vertices_seamntic ==  k)
    #     print('%d : %d'%(k, valid_idx.sum()))
    
    vertices_seamntic,vertices_grid_pts = vertices_seamntic.reshape(-1), vertices_grid_pts.reshape(-1,3)
    for k in  stuff_semantic_list:
        # if k in ['motorcycle','trailer','sky']:
        #     continue
        valid_idx = (vertices_seamntic ==  name2label[k].id)
        pt= o3d.geometry.PointCloud()
        print(k + '%d'%valid_idx.sum())
        if valid_idx.shape[0] != 0:
            pt.points = o3d.utility.Vector3dVector(vertices_grid_pts[valid_idx == 1])
            pt_color= np.array( name2label[k].color)/255.0
            pt.paint_uniform_color(pt_color)
            pt_group.append(pt)
    if vis:
        o3d.visualization.draw_geometries(pt_group)
    return pt_group


def vis_camera_o3d(instrinsic, extrinsic, img_size = (256,256),z_revrse = True, vis = False):
    frustums = []
    camera_size = 1
    K = np.eye(4)
    K[:3,:3] = instrinsic
    W2C = extrinsic
    C2W = np.linalg.inv(W2C)
    # img_size = (256,256)
    frustums.append(get_camera_frustum(img_size, K, W2C, frustum_length=camera_size, color=(1,1,1.),z_reverse = z_revrse))
    # cnt += 1
    camera_gropu = [frustums2lineset(frustums)]
    if vis:
        o3d.visualization.draw_geometries(camera_gropu)
    return camera_gropu



learning_map_inv={
# 0: "unlabeled", 
#    1: "car",
#   2: "bicycle",
#   3:  "motorcycle",
#   4: "truck",
#   5:  "other-vehicle",
#   6: "person",
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
    'vegetation','terrain','ground','road', 'sidewalk','parking','rail track','building','gate','garage', 'bridge','tunnel','wall',
    # 'truck','train','caravan','bus','trailer',
    'fence','guard rail','trash bin','box','lamp','smallpole','polegroup','stop','pole','traffic sign','traffic light']

voxel_dir = os.listdir('/data/ybyang/semantic-kitti')
voxel_dir.sort()
if not os.path.exists('kittivoxel4render'):
    os.makedirs('kittivoxel4render')
for d in voxel_dir[:100] :
    voxel_path = os.path.join('/data/ybyang/semantic-kitti',d )
    with open(voxel_path, 'rb+') as f:
        raw_data = pkl.load(f)


    # x_max, x_min, y_max, y_min, z_max, z_min = 32, -32, 2, -14, 64, 0
    x_max, x_min, y_max, y_min, z_max, z_min = 32, -32, 2, -14,64,0
    bounds = np.array((x_max, x_min, y_max, y_min, z_max, z_min))
    point_num_H, point_num_W, point_num_L, = 64, 64, 64
    vertices_gridx,  vertices_gridy, vertices_gridz= np.meshgrid(np.linspace(x_min, x_max, point_num_W), np.linspace(y_min, y_max, point_num_H), np.linspace(z_min, z_max, point_num_L), indexing='xy') 

    stuff_semantic_list_render =  [
        'vegetation','terrain','ground','road', 'sidewalk','parking','rail track','building','gate','garage', 'bridge','tunnel','wall',
        # 'car',
        # 'truck','train','caravan','bus','trailer',
        'fence','guard rail','trash bin','box','lamp','smallpole','polegroup','stop','pole','traffic sign','traffic light']

    '''
    the indexing mode, either “xy” or “ij”, defaults to “ij”. See warning for future changes.

    If “xy” is selected, the first dimension corresponds to the cardinality of the second input and the second dimension corresponds to the cardinality of the first input.

    If “ij” is selected, the dimensions are in the same order as the cardinality of the inputs.
    '''
    stuff_loc_gird = np.concatenate((vertices_gridx[:,:,:,None], vertices_gridy[:,:,:,None], vertices_gridz[:,:,:,None]), axis = -1)#! shape [H, W, L, 3]

    stuff_semantic = raw_data['y_pred'].transpose((2,1,0))
    # stuff_semantic = raw_data['y_pred']
    K = raw_data['cam_k']
    c2w_raw = np.linalg.inv(raw_data['T_velo_2_cam'])
    c2w_raw[:3,1] = -c2w_raw[:3,1]
    c2w_raw[:3,2] = -c2w_raw[:3,2]
    # c2w_raw[:3,0] = -c2w_raw[:3,0]
    c2w = np.array([[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]]) @ c2w_raw
    stuff_semantic = stuff_semantic.astype(np.uint8)
    stuff_semantic = F.interpolate(torch.tensor(stuff_semantic, dtype=torch.uint8)[None,None,:,:,:], (26, 52, 52), mode = 'nearest')[0,0].numpy()

    t = np.zeros((64,64,64))
    t[:26,6:58,:52] = stuff_semantic
    # a = np.linalg.inv(raw_data['T_velo_2_cam'])
    stuff_semantic = np.zeros((64,64,64))
    for i in learning_map_inv:
        stuff_semantic[t == i] = name2label[learning_map_inv[i]].id
    stuff_semantic = np.flip(stuff_semantic, axis=0)

    with open(os.path.join('kittivoxel4render', d), 'wb+') as fp:
        pkl.dump({"HW":(370,1220), "K":K, 
                  "zhenzhenyy": c2w , 
                  'T_velo_2_cam':raw_data['T_velo_2_cam'],"stuff_semantic_voxel": stuff_semantic},  fp)
    # for clver_scene in dataloader:
    # clver_scene = {k : clver_scene[k].cpu().numpy() for k in clver_scene}
    # intrinsic, camera_pose = clver_scene['camera_mat'][0], clver_scene['world_mat'][0]
    # stuff_semantic, stuff_loc = clver_scene['stuff_semantic_grid'][0],  clver_scene['stuff_loc_grid'][0]
    # bbox_tr, bbox_semantic = clver_scene['bbox_tr'][0], clver_scene['bbox_semantic'][0]
    if True:
        geo_group = []
        geo_group += vis_voxel_world_o3d(vertices_grid_pts= stuff_loc_gird, vertices_seamntic= stuff_semantic, stuff_semantic_list =stuff_semantic_list_render)
        geo_group += vis_world_bounds_o3d(bounds=bounds)
        # geo_group += vis_bbox_layout_o3d(bbox_tr=bbox_tr, bbox_semantic=bbox_semantic)
        geo_group += vis_camera_o3d(instrinsic=K, extrinsic=np.linalg.inv(c2w), z_revrse=False)
        vis = costum_visualizer_o3d(geo_group=geo_group, instrinsic=K, extrinsic=np.linalg.inv(c2w))
            
        vis.run()
        del vis

    print('Done')

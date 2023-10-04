import time
import json
import open3d as o3d
from math import sqrt, sin, cos
import numpy as np
import os
import matplotlib.pyplot as plt
import trimesh
import imageio
import re
import cv2
import open3d
import copy
from tools.kitti360Scripts.helpers.annotation import Annotation2D, global2local, Annotation2DInstance
from tools.kitti360Scripts.helpers.labels import name2label, id2label
from lib.utils.img_utils import save_tensor_img
from tools.kitti360Scripts.viewer.kitti360Viewer3D import Kitti360Viewer3D
import pickle as pkl
import trimesh
from lib.utils.transform_utils import create_R, RT2tr, tr2RT
from mpl_toolkits import mplot3d


pi = 3.1415

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


def draw_camera_frustum(H, W, K,c2w, depth = 32):
    '''
    return 
    '''
    center_x, center_y =  K[0,2], K[1,2]
    origin = np.array([0,0,1])
    frustum_rays = np.array([[0,center_y,1],[W/3,center_y,1],[W*2/3,center_y,1],[W,center_y,1]])
    frustum_rays = frustum_rays @np.linalg.inv(K[:3, :3]).T 
    frustum_rays = frustum_rays@ c2w[:3, :3].T
    origin = c2w[:3, 3]
    frustum_rays =  frustum_rays/ np.linalg.norm(frustum_rays, axis=-1)[:, None]
    frustum_vertices = depth * frustum_rays + origin
    
    camera_frustum = np.concatenate((origin.reshape(1,3), frustum_vertices), axis = 0)

    return camera_frustum


def cross(p1, p2, p):
    '''
    calculate (p1p2) X (p1p)
    '''
    return (p2[0] - p1[0]) * (p[1] - p1[1]) - (p[0] - p1[0]) * (p2[1] - p1[1])


def ifInMatrix(P, matrix_vertices):
    '''
    (AB X AE ) * (CD X CE)  >= 0 && (DA X DE ) * (BC X BE) >= 0 。
    return 
    '''
    assert matrix_vertices.shape == (4,2)
    A, B, C, D = matrix_vertices[0], matrix_vertices[1], matrix_vertices[2], matrix_vertices[3]
        
    a = cross(A, B, P) * cross(C, D, P) >= 0
    b = cross(D, A, P) * cross(B, C, P) >= 0

    if_in_matrix = (a and b)

    return if_in_matrix

def ifAllInMatrix(p_list, matrix_vertices = np.array(((-32,0),(32,0),(32,64),(-32,64)))):
    if_all_in_matrix = True 
    assert len(p_list.shape) == 2 and p_list.shape[1] == 2
    for i in range(p_list.shape[0]): 
        if_all_in_matrix  *= ifInMatrix(p_list[i], matrix_vertices)

    return if_all_in_matrix


def assignColor(globalIds, gtType='semantic'):
    if not isinstance(globalIds, (np.ndarray, np.generic)):
        globalIds = np.array(globalIds)[None]
    color = np.zeros((globalIds.size, 3))
    for uid in np.unique(globalIds):
        semanticId, instanceId = global2local(uid)
        if gtType=='semantic':
            color[globalIds==uid] = id2label[semanticId].color
        else:
            color[globalIds==uid] = (96,96,96) # stuff objects in instance mode
    color = color.astype(np.float)/255.0
    return color

def loadBoundingBoxes(annotations):
    bboxes = []
    bboxes_globalId = []
    bboxes_semanticId = []
    
    for globalId,obj in annotations:
        # skip dynamic objects
        lines=np.array(obj.lines)
        vertices=obj.vertices
        faces=obj.faces
        mesh = open3d.geometry.TriangleMesh()
        mesh.vertices = open3d.utility.Vector3dVector(obj.vertices)
        mesh.triangles = open3d.utility.Vector3iVector(obj.faces)
        color = assignColor(globalId, 'semantic')
        semanticId, instanceId = global2local(globalId)
        mesh.paint_uniform_color(color.flatten())
        mesh.compute_vertex_normals()
        bboxes.append( mesh )
        bboxes_semanticId.append(semanticId)
        bboxes_globalId.append(globalId)
    return bboxes

def readVariable(fid, name, M, N):
    # rewind
    fid.seek(0, 0)
    # search for variable identifier
    line = 1
    success = 0
    while line:
        line = fid.readline()
        if line.startswith(name):
            success = 1
            break
    # return if variable identifier not found
    if success == 0:
        return None
    # fill matrix
    line = line.replace('%s:' % name, '')
    line = line.split()
    assert (len(line) == M * N)
    line = [float(x) for x in line]
    mat = np.array(line).reshape(M, N)
    return mat


def loadCalibrationCameraToPose(filename):
    # open file
    fid = open(filename, 'r')
    # read variables
    Tr = {}
    cameras = ['image_00', 'image_01', 'image_02', 'image_03']
    lastrow = np.array([0, 0, 0, 1]).reshape(1, 4)
    for camera in cameras:
        Tr[camera] = np.concatenate((readVariable(fid, camera, 3, 4), lastrow))
    # close file
    fid.close()
    return Tr


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


class Dataset:
    def __init__(self,  data_root, seq_num, frame_start = -1, frame_num = -1, make_layout = True, make_road_semantic = False, make_semantic_volume = True, resume = False, ratio = 0.25):
        super(Dataset, self).__init__()
        # path and initialization
        sequence = os.path.join('2013_05_28_drive_' + '%04d'%seq_num + '_sync')
        self.sequence = sequence
        self.img_root = os.path.join(data_root,'data_2d_raw',  sequence)
        self.semantic_root = os.path.join(data_root, 'data_2d_semantics','train', sequence)
        self.bbx_root = os.path.join(data_root, 'data_3d_bboxes')

        # load intrinsics
        calib_dir = os.path.join(data_root, 'calibration')
        self.intrinsic_file = os.path.join(calib_dir, 'perspective.txt')
        self.load_intrinsic(self.intrinsic_file)

        self.K_00[:2] = self.K_00[:2] * ratio
        self.intrinsic = self.K_00[:, :-1]

        # load camera poses
        cam2world_root = os.path.join(data_root, 'data_poses', sequence, 'cam0_to_world.txt')
        self.cam2world_dict_00 = {}
        self.pose_file = os.path.join(data_root, 'data_poses', sequence, 'poses.txt')
        poses = np.loadtxt(self.pose_file)
        frames = poses[:, 0].astype(int)
        poses = np.reshape(poses[:, 1:], [-1, 3, 4])
        fileCameraToPose = os.path.join(calib_dir, 'calib_cam_to_pose.txt')
        self.camToPose = loadCalibrationCameraToPose(fileCameraToPose)['image_01']
        for line in open(cam2world_root, 'r').readlines():
            value = list(map(float, line.strip().split(" ")))
            self.cam2world_dict_00[value[0]] = np.array(value[1:]).reshape(4, 4)

        # load image_ids
        # For not every raw 2D image has camera pose, some selected is requires
        if frame_start == -1:
            start_idx = frames[0]
        else:
            start_idx = frame_start
        if frame_num == -1:
            frame_num = len(frames)- start_idx
    
        self.image_ids = frames[start_idx : start_idx + frame_num]

        # load images
        self.rgb_img_dict = {}
        for idx in self.image_ids:
            frame_name = '%010d' % idx
            image_file_00 = os.path.join(self.img_root, 'image_00/data_rect/%s.png' % frame_name)
            if not os.path.isfile(image_file_00):
                continue
                #raise RuntimeError('%s does not exist!' % image_file_00)
            self.rgb_img_dict[idx] = image_file_00

        # load semantic 
        N = 10000
        obj = Annotation2D(N = 1000)
        self.seamntic_img_dict = {}
        self.visible_instance_dict = {}
        visible_instance_dict_path = os.path.join(self.semantic_root, 'image_00', 'visible_instance_dict.pkl') 
        if not os.path.isfile(visible_instance_dict_path):
            for img_idx in self.image_ids:
                #img_idx = 4399
                semantic_path = os.path.join(self.semantic_root, 'image_00','instance', '%010d.png'%int(img_idx))
                if not os.path.isfile(semantic_path ):
                    continue
                obj.loadInstance(semantic_path, toImg=False)
                S = obj.semanticId.astype(np.long)
                I = obj.instanceId.astype(np.long)
                G = N * S + I
                self.visible_instance_dict[img_idx] = {k : len(G[G == k]) for k in list(np.unique(G)) }
                #self.seamntic_img_dict[img_idx] = G
            pkl.dump(self.visible_instance_dict, open(visible_instance_dict_path, 'wb+'))
        else:
            self.visible_instance_dict = pkl.load(open(visible_instance_dict_path, 'rb+')) 
        # not all 2d rgb image has corrpondong semantic image, thus those frame without semantic should be remove
        self.image_ids = [int(i) for i in self.image_ids if i in self.visible_instance_dict.keys() ]
        self.rgb_img_dict = {i : self.rgb_img_dict[i]  for i in  self.image_ids}

        if make_layout:
            # Load annotation3D(instance only) and transfer instances's bbx to local coordinate
            layout_root = os.path.join(data_root, 'layout', self.sequence)
            if not os.path.exists(layout_root):
                os.makedirs(layout_root)
            v = Kitti360Viewer3D(path=data_root, seq=seq_num, load_full=False)
            ins_annotation3D  = v.annotation3D


            trajectory_fram_num = 50
            
            trajectory = {}
            for i, (idx, vis_ins) in enumerate(self.visible_instance_dict.items()):
                layout_filepath = os.path.join(layout_root, '{:010d}.pkl'.format(idx))
                if os.path.exists(layout_filepath) == True and not resume:
                    continue
                #     os.system('rm {}'.format(fileroot))
                
                c2w = self.cam2world_dict_00[idx]
                rect_mat =  RT2tr(create_R((5 * pi / 180 , 0 , 0 ),(1,1,1)), (0,1.6,0))
                c2w = c2w @ rect_mat     
                w2c = np.linalg.inv(c2w)
                trajectory[idx] = {}
                for j in range(trajectory_fram_num):
                    idx_t = self.image_ids[min(i + j,len(self.image_ids) - 1)]
                    pose_i = w2c @ self.cam2world_dict_00[idx_t]
                    # build_rays(H = 94, W = 352, K = self.intrinsic, c2w=pose_i)
                    frustum_i =  draw_camera_frustum(H = 94, W = 352, K = self.intrinsic, c2w=pose_i, depth=32)
                    frustum_i_xz = np.concatenate((frustum_i[:,0,None],frustum_i[:,2,None]), axis = -1)
                    frustum_in_matrix =  ifAllInMatrix(frustum_i_xz)
                    if frustum_in_matrix:
                        trajectory[idx][idx_t] = pose_i

                    # print('frame%02d(x:%02f,y:%02f, z:%02f) in traj %d'%(j, pose_i[0,3],pose_i[1,3],pose_i[2,3], frustum_in_matrix))
                R_cam, T_cam = c2w[0:3,0:3], c2w[0:3,3]
                ins_dict = {i : ins_annotation3D.objects[i] for i in vis_ins}
                ins_local = {}
                for globalId, obj in ins_dict.items():
                    if obj == {}:
                        continue
                    if len(obj.keys()) > 1:
                        continue
                    points = obj[-1].vertices
                    o = copy.deepcopy(obj[-1])
                    o.vertices = np.matmul(R_cam.T, (points - T_cam).T).T

                    tr_w = RT2tr(o.R, o.T)
                    tr_c = w2c @ tr_w
                    o.R, o.T = tr2RT(tr_c)
                    o.pixel_num = vis_ins[globalId]

                    ins_local[globalId] = o
                with open(layout_filepath,'wb+') as f:
                    pkl.dump(ins_local, f)
            # save valid frame list
            with open(os.path.join(layout_root, 'valid_frames.json'), 'w') as f:
                json.dump(self.image_ids, f)
            with open(os.path.join(layout_root, 'trajectory.pkl'), 'wb+') as f:
                pkl.dump(trajectory, f)

        if make_road_semantic:
            # # make road semantic_plane
            road_seamntic_root = os.path.join(data_root, 'road_semantic', self.sequence)
            if not os.path.exists(road_seamntic_root):
                os.makedirs(road_seamntic_root)
            road_semantic_list = ['ground', 'road', 'sidewalk', 'parking','rail track', 'terrain']
            road_semanticId_list = [name2label[i].id for i in road_semantic_list]
            v = Kitti360Viewer3D(path='datasets/KITTI-360', seq=seq_num, load_full=True)
            v.loadBoundingBoxes()
            annotation3D  = v.annotation3D
            windows_unique = np.unique(np.array(v.bboxes_window), axis=0)
            for idx in self.image_ids:
                # Check if already exist
                road_semantic_filepath = os.path.join(road_seamntic_root, '{:010d}.png'.format(idx))
                if os.path.exists(road_semantic_filepath) == True:
                    continue
                # Get the window of target frame
                for w_id , w in enumerate(windows_unique):
                    if w[0] <= idx and w[1] >= idx:
                        window = w
                        # Load 2 window at the same time
                        if w_id != (windows_unique.shape[0] -1):
                            window_next = windows_unique[w_id + 1]
                        else:
                            window_next = w
                        break
                road_bbxes = [v.bboxes[i] for i in range(len(v.bboxes)) if (v.bboxes_window[i][0]==window[0] or v.bboxes_window[i][0]==window_next[0])and v.bboxes_semanticId[i] in road_semanticId_list]
                # Show birdeveview image
                bv_cam_intrinsic, bv_cam_extrinsic = get_o3d_birdeyeview_cam(c2w = self.cam2world_dict_00[idx])
                raw_semantic_plane = shot_birdeyeview_image(road_bbxes, bv_cam_extrinsic, bv_cam_intrinsic, o3d_visable= False)
                # Rgb shot to semantic plane
                clist = np.zeros((len(road_semantic_list)+1,128,128,3))
                slist = [0]
                for i, sem in enumerate(road_semantic_list):
                    clist[i+1,...] = np.repeat(np.array(name2label[sem].color).reshape((1,3)), 128 * 128, axis = 0).reshape(128,128,3)
                    slist.append(name2label[sem].id)
                t = np.repeat(raw_semantic_plane.reshape((1,128,128,3)), len(road_semantic_list)+1, axis=0)
                t = np.sum(np.abs(t - clist), axis=-1).transpose((1,2,0)).argmin(axis = -1)
                t= t.astype('u1').T
                road_semantic_plane = t.copy()
                for i, s in enumerate(slist):
                    #! 给矩阵按照索引重新赋值前要先copy一份
                    #road_semantic_plane[road_semantic_plane == i] = s
                    road_semantic_plane[t == i] = s
                # Visualize semantic plane
                if False:
                    plt.subplot(2, 1, 1)
                    plt.imshow(raw_semantic_plane)
                    plt.subplot(2, 1, 2)
                    plt.imshow(road_semantic_plane)
                    plt.show()
                # plt.imshow(road_semantic_plane, cmap='gray')
                imageio.imsave(road_semantic_filepath, road_semantic_plane)
                #!plt.imsave(road_semantic_filepath, road_semantic_plane, cmap='gray')
                # plt.show()

        if make_semantic_volume:
            H, W, L = 16, 64, 64
            point_num_H, point_num_W, point_num_L, = 64,64,64           # # make road semantic_plane
            uncountable_semantic_root = os.path.join(data_root, 'semantic_voxel', self.sequence, '(H:%d:%d,W%d:%d,L%d:%d)'%(H,point_num_H, W,point_num_W, L,point_num_L))
            uncountable_instance_root = os.path.join(data_root, 'semantic_voxel_insId', self.sequence, '(H:%d:%d,W%d:%d,L%d:%d)'%(H,point_num_H, W,point_num_W, L,point_num_L))
            if not os.path.exists(uncountable_semantic_root):
                os.makedirs(uncountable_semantic_root)
            # uncountable_semantic_list = ['vegetation','wall' ,'fence','ground', 'road', 'sidewalk', 'parking','rail track', 'terrain']

            uncountable_semantic_list = [k for k in name2label if (re.search('unknown', k) == None and re.search('dynamic', k) == None)]
            uncountable_semanticId_list = [name2label[k].id for k in uncountable_semantic_list]

            v = Kitti360Viewer3D(path=data_root, seq=seq_num, load_full=True)
            v.loadBoundingBoxes()
            annotation3D  = v.annotation3D
            windows_unique = np.unique(np.array(v.bboxes_window), axis=0)
            for idx in self.image_ids:
                # Check if already exist

                uncountable_semantic_filepath = os.path.join(uncountable_semantic_root,'{:010d}.pkl'.format(idx))
                uncountable_instance_filepath = os.path.join(uncountable_semantic_root,'{:010d}_insId.pkl'.format(idx))
                if os.path.exists(uncountable_semantic_filepath)and os.path.exists(uncountable_instance_filepath) == True  and not resume:
                    continue
                # Get the window of target frame
                for w_id , w in enumerate(windows_unique):
                    if w[0] <= idx and w[1] >= idx:
                        window = w
                        # Load 2 window at the same time
                        if w_id != (windows_unique.shape[0] -1):
                            window_next = windows_unique[w_id + 1]
                        else:
                            window_next = w
                        break
                if True:
                    uncountable_annotations = [list(v.annotation3D.objects[v.bboxes_globalId[i]].values())[0] for i in range(len(v.bboxes)) if ((v.bboxes_window[i][0]==window[0] or v.bboxes_window[i][0]==window_next[0])  and v.bboxes_semanticId[i] in uncountable_semanticId_list)]
                else:
                    uncountable_annotations = [list(v.annotation3D.objects[v.bboxes_globalId[i]].values())[0] for i in range(len(v.bboxes)) if v.bboxes_semanticId[i] in uncountable_semanticId_list]
    
    
                # H, W, L = 32, 64, 128
                # point_num_H, point_num_W, point_num_L, = 32, 64, 128
                vertices_gridx,  vertices_gridy, vertices_gridz= np.meshgrid(np.linspace(-W/2, W/2, point_num_W), np.linspace(2-H, 2, point_num_H), np.linspace(0, L, point_num_L))
                vertices_grid = np.concatenate((vertices_gridx[:,:,:,None], vertices_gridy[:,:,:,None], vertices_gridz[:,:,:,None]), axis = -1).reshape((-1,3))

                c2w = self.cam2world_dict_00[idx]

                rect_mat =  RT2tr(create_R((5 * pi / 180 , 0 , 0 ),(1,1,1)), (0,1.6,0))

                rect_mat = c2w @ rect_mat
                
                vertices_grid_world = np.matmul(rect_mat[:3,:3], vertices_grid.T).T + rect_mat[:3,3]

                vertices_seamntic, vertices_instance = {}, {}
                # vertices_seamntic_grid = np.zeros(vertices_grid.shape[0])
                for name in uncountable_semantic_list:
                     vertices_seamntic[name] = []
                     vertices_instance[name] = []

                t = time.time()
                n = 0
                # uncountable_semantic_list = ['vegetation','terrain','ground','road','parking','sidewalk','rail track','wall','fence']
                # uncountable_semantic_list = [id2label[k].name for k in  id2label]
                for obj in uncountable_annotations:
                    max_xyz = np.max(obj.vertices, axis = 0)
                    min_xyz = np.min(obj.vertices, axis = 0)
                    bounds = np.stack([min_xyz, max_xyz], axis=0)
                    bbx_mask = get_near_far(bounds, ray_o=vertices_grid_world, ray_d = np.ones_like(vertices_grid_world), return_mask = True)
                    if np.sum(bbx_mask) > 0:
                        candidate_idx = np.argwhere(bbx_mask == True).reshape(-1)
                        candidate_vertices = vertices_grid_world[candidate_idx]
                        obj_mesh_tri = trimesh.Trimesh(vertices=obj.vertices, faces=obj.faces)
                        _, index_rays, _ = obj_mesh_tri.ray.intersects_location(ray_origins=candidate_vertices, ray_directions=np.ones_like(candidate_vertices))
                        intersect_rays, intersect_counts = np.unique(index_rays, return_counts=True)
                        valid_idx = candidate_idx[intersect_rays[intersect_counts % 2 != 0]]
                        vertices_seamntic[obj.name] += list(valid_idx)
                        vertices_instance[obj.name] += [obj.instanceId] * len(valid_idx)
                        # vertices_seamntic_grid[valid_idx] = name2label[obj.name].id 
                t = time.time() - t

                for k in vertices_seamntic:
                    vertices_seamntic[k] = np.array(vertices_seamntic[k])
                    vertices_instance[k] = np.array(vertices_instance[k])

                voexl_world = {'semantic': vertices_seamntic, 'instance':vertices_instance}

                with open(uncountable_semantic_filepath, 'wb+') as f:
                    # np.savez()
                    pkl.dump(voexl_world['semantic'], f)
                with open(uncountable_instance_filepath, 'wb+') as f:
                    # np.savez()
                    pkl.dump(voexl_world['instance'], f)
                # with open('vertices_seamntic', 'wb+') as f:
                #     pkl.dump(vertices_seamntic, f)
                print('seq%04d frame%010d \'s vertices grid done! cost%d scend'%(seq_num, idx, t))
                

                if False:
                    fig = plt.figure()
                    ax = plt.axes(projection='3d')
                    a = vertices_seamntic[obj.name] == 0
                    b = vertices_seamntic[obj.name] == 1
                    #ax.scatter3D(vertices_grid[a,0], vertices_grid[a,1], vertices_grid[a,2])
                    ax.scatter3D(vertices_grid[b,0], vertices_grid[b,1], vertices_grid[b,2])
                if False:
                    pt_group = []
                    for k in vertices_seamntic.keys():
                        valid_idx = np.argwhere(vertices_seamntic[k] == 1).reshape(-1)
                        pt= o3d.geometry.PointCloud()
                        pt.points = o3d.utility.Vector3dVector(vertices_grid_world[valid_idx])
                        pt_color= np.array(name2label[k].color)/255.0
                        pt.paint_uniform_color(pt_color)
                        pt_group.append(pt)
                    #o3d.visualization.draw_geometries([pt])

                    bbxes = [v.bboxes[i] for i in range(len(v.bboxes)) if (v.bboxes_window[i][0]==window[0] and v.bboxes_semanticId[i] not in uncountable_semanticId_list)]
                    #and v.bboxes_semanticId[i] not in uncountable_semanticId_list]

                    vis = open3d.visualization.Visualizer()
                    vis.create_window(width=1000, height=1000, visible= True)
                    for pt in pt_group:
                        vis.add_geometry(pt)
                    for b in bbxes:
                        vis.add_geometry(b)

                    ctr = vis.get_view_control()
                    cam = ctr.convert_to_pinhole_camera_parameters()
                    cam_extrinsic = np.linalg.inv(c2w)
                    cam.extrinsic = cam_extrinsic
                    ctr.convert_from_pinhole_camera_parameters(cam)

                    rd = vis.get_render_option()
                    rd.light_on = False
                    rd.background_color = np.array([0,0,0])

                    vis.update_geometry(b)
                    vis.poll_events()
                    vis.update_renderer()
                    vis.run()

        
        print(sequence + 'done!')

    def load_intrinsic(self, intrinsic_file):
        with open(intrinsic_file) as f:
            intrinsics = f.read().splitlines()
        for line in intrinsics:
            line = line.split(' ')
            if line[0] == 'P_rect_00:':
                K = [float(x) for x in line[1:]]
                K = np.reshape(K, [3, 4])
                self.K_00 = K
            elif line[0] == 'P_rect_01:':
                K = [float(x) for x in line[1:]]
                K = np.reshape(K, [3, 4])
                intrinsic_loaded = True
                self.K_01 = K
            elif line[0] == 'R_rect_01:':
                R_rect = np.eye(4)
                R_rect[:3, :3] = np.array([float(x) for x in line[1:]]).reshape(3, 3)
            elif line[0] == "S_rect_01:":
                width = int(float(line[1]))
                height = int(float(line[2]))
        assert (intrinsic_loaded == True)
        assert (width > 0 and height > 0)
        self.width, self.height = width, height
        self.R_rect = R_rect

    def __getitem__(self, index):
        return 0

    def __len__(self):
        return len(self.metas)


def shot_birdeyeview_image(bbxes, bv_cam_extrinsic,bv_cam_intrinsic, image_size = (128,128), road_size = (100,100) , o3d_visable = False):

    shot_width, shot_height=1000, 1000
    vis = open3d.visualization.Visualizer()
    vis.create_window(width=shot_width, height=shot_height, visible= o3d_visable)
    for b in bbxes:
        vis.add_geometry(b)
    ctr = vis.get_view_control()
    
    bv_cam = ctr.convert_to_pinhole_camera_parameters()

    bv_cam.extrinsic = bv_cam_extrinsic
    bv_cam.intrinsic.intrinsic_matrix = bv_cam_intrinsic
    ctr.convert_from_pinhole_camera_parameters(bv_cam)

    rd = vis.get_render_option()
    rd.light_on = False
    rd.background_color = np.array([0,0,0])

    vis.update_geometry(b)
    vis.poll_events()
    vis.update_renderer()

    bv_img = vis.capture_screen_float_buffer()
    if o3d_visable:
        vis.run()
    del vis, ctr, rd
    #semantic_plane = vis.capture_screen
    bv_img = (255.0 * np.asarray(bv_img)).astype(np.uint8)
    # a = np.unique(raw_img.reshape((-1,3)), axis = 0)
    bv_img = cv2.resize(bv_img, (128, 128), interpolation=cv2.INTER_NEAREST)
    return bv_img

def get_o3d_birdeyeview_cam(c2w, bv_u=1000, bv_v= 1000,W_road = 64, L_road = 64):
    #W_road, L_road = 64, 64
    pi = 3.1415
    theta = pi * 5/180
    road_ltitude = 1.6 # 

    # Computr birdeyeview camera extrinsic (suppose fov = 60)
    x_bv_cam = 0
    y_bv_cam = road_ltitude - L_road * sin(theta) * 0.5 - sqrt(3)* 0.5 * L_road * cos(theta)
    z_bv_cam = L_road * cos(theta) * 0.5 - sqrt(3) * L_road * sin(theta) * 0.5
    bv_cam_pose = np.zeros((4,4))
    bv_cam_pose[3,3] = 1
    bv_cam_pose[:3,:3] = create_R((-85 * pi / 180 , 0 , 0 ),(1,1,1)) 
    bv_cam_pose[:3,3] = np.array((x_bv_cam, y_bv_cam, z_bv_cam))

    bv_cam_extrinsic = np.linalg.inv(bv_cam_pose) @ np.linalg.inv(c2w)


    fov = 60
    focal = 1. / np.tan(0.5 * fov * np.pi/180.)
    bv_cam_intrinsic = np.array(
    [[focal* bv_u/2 ,0.0,bv_u/2-0.5],
    [0.0,focal*  bv_v/2,bv_u/2-0.5],
    [0.0,0.0,1.0]])

    return bv_cam_intrinsic, bv_cam_extrinsic


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_num", default=3, type=int)
    parser.add_argument("--frame_start", default=-1, type=int)
    parser.add_argument("--frame_num", default= -1, type=int)
    args = parser.parse_args()
    frame_start = -1
    frame_num = -1
    data_root = '/data/ybyang/KITTI-360/'

    # render_road = False
    # seq_list = [0,2,3,4,5,6,7,9,10]
    seq_list = [args.seq_num]
    frame_start = args.frame_start
    frame_num = args.frame_num
    for seq_num in seq_list:
        road_semantic = Dataset(    data_root=data_root,
                                    seq_num=seq_num, 
                                    frame_start = frame_start,
                                    frame_num = frame_num,
                                    make_layout = False, 
                                    make_road_semantic = False, 
                                    make_semantic_volume = True,
                                    resume = False)


from __future__ import annotations
from code import interact
from collections import defaultdict
from math import sqrt, sin, cos, tan
from re import X
import numpy as np
import os
import matplotlib.pyplot as plt
import trimesh
from lib.config import cfg
import imageio
import torch
import cv2
import open3d
from tools.kitti360Scripts.helpers.annotation import Annotation2D, global2local
from tools.kitti360Scripts.helpers.labels import name2label, id2label
from lib.utils.img_utils import save_tensor_img
from tools.vis_cam import get_camera_frustum, frustums2lineset
from tools.kitti360Scripts.viewer.kitti360Viewer3D import Kitti360Viewer3D
import copy
import mpl_toolkits.mplot3d as p3d
import pickle as pkl
import trimesh
from lib.utils.transform_utils import create_R



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


# AABB
def get_near_far(bounds, ray_o, ray_d):
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
    if mask_at_box.sum() > 0:
        return True
    else:
        return False

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

class Dataset:
    def __init__(self, cam2world_root, data_root, sequence):
        super(Dataset, self).__init__()
        # path and initialization
        self.split = split
        self.sequence = sequence
        self.start = cfg.start

        self.img_root = os.path.join(data_root,'data_2d_raw',  sequence)
        self.semantic_root = os.path.join(data_root, 'data_2d_semantics','train', sequence)
        self.bbx_root = os.path.join(data_root, 'data_3d_bboxes')

        # load intrinsics
        calib_dir = os.path.join(data_root, 'calibration')
        self.intrinsic_file = os.path.join(calib_dir, 'perspective.txt')
        self.load_intrinsic(self.intrinsic_file)
        ratio = 1
        self.H = int(self.height * cfg.ratio)
        self.W = int(self.width  * cfg.ratio)
        self.K_00[:2] = self.K_00[:2] * cfg.ratio
        self.K_01[:2] = self.K_01[:2] * cfg.ratio
        self.intrinsic_00 = self.K_00[:, :-1]
        self.intrinsic_01 = self.K_01[:, :-1]
 
        # load cam2world poses
        self.cam2world_dict_00 = {}
        self.cam2world_dict_01 = {}
        self.pose_file = os.path.join(data_root, 'data_poses', sequence, 'poses.txt')
        poses = np.loadtxt(self.pose_file)
        frames = poses[:, 0].astype(int)
        poses = np.reshape(poses[:, 1:], [-1, 3, 4])
        fileCameraToPose = os.path.join(calib_dir, 'calib_cam_to_pose.txt')
        self.camToPose = loadCalibrationCameraToPose(fileCameraToPose)['image_01']
        for line in open(cam2world_root, 'r').readlines():
            value = list(map(float, line.strip().split(" ")))
            self.cam2world_dict_00[value[0]] = np.array(value[1:]).reshape(4, 4)
        for frame, pose in zip(frames, poses):
            pose = np.concatenate((pose, np.array([0., 0., 0.,1.]).reshape(1, 4)))
            self.cam2world_dict_01[frame] = np.matmul(np.matmul(pose, self.camToPose), np.linalg.inv(self.R_rect))
        self.translation = np.array(cfg.center_pose)

        # load image_ids
        # For not every raw 2D image has camera pose, some selected is requires
        if self.start in frames:
            self.start = self.start
        else:
            for i, _ in enumerate(frames):
                if frames[i] < self.start and frames[i+1] > self.start:
                    self.start = frames[i+1]
                    break 
        start_idx = int(np.argwhere(frames == self.start))
        train_ids = frames[start_idx : start_idx + cfg.train_frames]
        test_ids = frames[start_idx: start_idx + cfg.train_frames]
        if split == 'train':
            self.image_ids = train_ids
        elif split == 'val':
            self.image_ids = test_ids

        # load images
        self.visible_id_root = os.path.join(data_root, 'visible_id', sequence)
        self.images_list_00 = {}
        self.images_list_01 = {}
        for idx in self.image_ids:
            frame_name = '%010d' % idx
            image_file_00 = os.path.join(self.img_root, 'image_00/data_rect/%s.png' % frame_name)
            image_file_01 = os.path.join(self.img_root, 'image_01/data_rect/%s.png' % frame_name)
            if not os.path.isfile(image_file_00):
                raise RuntimeError('%s does not exist!' % image_file_00)
            self.images_list_00[idx] = image_file_00
            self.images_list_01[idx] = image_file_01

        # Load annotation3D
        v = Kitti360Viewer3D(path='datasets/KITTI-360', seq=0, load_full=True)
        self.annotation3D  = v.annotation3D
        v.loadBoundingBoxes()
        self.bbx_static = {}
        self.bbx_static_globalId = []

        for globalId in self.annotation3D.objects.keys():
            if len(self.annotation3D.objects[globalId].keys()) == 1:
                if -1 in self.annotation3D.objects[globalId].keys():
                    self.bbx_static[globalId] = self.annotation3D.objects[globalId][-1]
                    self.bbx_static_globalId.append(globalId)
        self.bbx_static_globalIds = np.array(self.bbx_static_globalId)

        # load semantic mask
        self.bbx_globalIds = {}
        print('Will render semantics: %s'% road_semantic_list)
        self.render_semantic = road_semantic_list
        self.render_semantic_ids = np.array([name2label[name].id for name in self.render_semantic])
        self.render_bbx_globalIds = defaultdict(dict)
        obj = Annotation2D()

        self.occupancy_mask = {}
        for img_idx in self.image_ids:
        #     # Load render bbx ids
        #     filename = os.path.join(self.visible_id_root, '{:010d}.txt'.format(idx))
        #     with open(filename, "r") as f:
        #         data = f.read().splitlines()
        #         visible_bbx_annotationIds = np.array(list(map(int, data)))
        #         visible_bbx_globalIds = np.unique([v.ann2global[i] for i in visible_bbx_annotationIds])
            visible_bbx_globalIds = np.unique([v.bboxes_globalId[i] for i in range(len(v.bboxes)) if v.bboxes_window[i][0] < img_idx and v.bboxes_window[i][1] >= img_idx ])
            # for i in annotationId:
            #     if i in self.bbx_static_annotationId and self.bbx_static[i].semanticId in  self.render_semantic_ids:
            #         self.render_bbx_ids.append(i)
            imgPath = os.path.join(self.semantic_root, 'image_00','instance', '%010d.png'%int(img_idx))
            obj.loadInstance(imgPath, toImg=False)
            S = obj.semanticId.astype(np.long)
            I = obj.instanceId.astype(np.long)
            G = 10000 * S + I
            # visible_bbx_globalIds = np.intersect1d(np.unique(G), self.bbx_static_globalIds)
            raw_bbx_semanticids, _ = global2local(visible_bbx_globalIds)
            render_index = np.argwhere(np.in1d(raw_bbx_semanticids , self.render_semantic_ids) == True).reshape(-1)
            render_bbx_globalIds =  visible_bbx_globalIds[render_index]
            self.render_bbx_globalIds[img_idx] = render_bbx_globalIds

            mask = np.where(np.isin(S, self.render_semantic_ids), np.ones_like(S), np.zeros_like(S))
            mask = np.tile(mask, (3,1,1)).transpose(1,2,0)
            self.occupancy_mask[img_idx] = mask


            a = [v.bboxes[i] for i in range(len(v.bboxes)) if v.bboxes_globalId[i] in render_bbx_globalIds]
            #open3d.visualization.draw_geometries(a)

        # Build metas
        del v
        self.build_metas(self.cam2world_dict_00, self.cam2world_dict_01, self.images_list_00, self.images_list_01)

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

    def build_metas(self, cam2world_dict_00, cam2world_dict_01, images_list_00, images_list_01, intersection_dict_00 = None, intersection_dict_01 = None):
        input_tuples = []
        H, W = self.H, self.W
        for i, idx in enumerate(self.image_ids):
            pose = cam2world_dict_00[idx]
            #pose[:3, 3] = pose[:3, 3] - self.translation
            image_path = images_list_00[idx]

            image = imageio.imread(image_path)
            masked_image = np.where(self.occupancy_mask[idx] == 0, np.zeros_like(image), image)

            # masked_gt_path = 'out/tmp/gt_masked'
            # if not os.path.exists(masked_gt_path):
            #     os.makedirs(masked_gt_path)
            # im = Image.fromarray(masked_image)
            # im.save(fp = os.path.join(masked_gt_path, '%010d.jpg'%idx))

            image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)
            masked_image = cv2.resize(masked_image, (self.W, self.H), interpolation=cv2.INTER_AREA)
            occupancy_mask = self.occupancy_mask[idx][...,0].astype('uint8')
            occupancy_mask = cv2.resize(occupancy_mask, (self.W, self.H), interpolation=cv2.INTER_AREA)
            image = (image.astype(np.float32) / 255.)
            masked_image = (masked_image.astype(np.float32) / 255.)
            occupancy_mask = np.expand_dims(occupancy_mask.astype(np.float32), axis = -1)

            world_mat = pose
            camera_mat = self.intrinsic_00
            bbx_globalIds = self.render_bbx_globalIds[idx]
            # bbx_tr = self.bbx_trs[idx]
            #bbx_semantic_id = self.bbx_semantic_ids[idx]
            #bbx_intersection = 0
            # bbx_mask = self.bbx_masks[idx]
            stereo_num = 0
            input_tuples.append((idx, H, W, image, masked_image, occupancy_mask,  camera_mat, world_mat, bbx_globalIds, stereo_num))
        print('load meta_00 done')
        self.metas = input_tuples

    def __getitem__(self, index):
        idx, H, W , image, masked_image,occupancy_mask, camera_mat, world_mat, bbx_globalIds,  stereo_num = self.metas[index]
        ret = {
            'frame_id': idx,
            'H': H,
            'W': W,
            'image': image.transpose(2,0,1), 
            'masked_image': masked_image.transpose(2,0,1),
            'occupancy_mask': occupancy_mask.transpose(2,0,1),
            'camera_mat': camera_mat.astype(np.float32),
            'world_mat': world_mat.astype(np.float32),
            'bbx_globalIds': bbx_globalIds,           
        }
        return ret

    def __len__(self):
        return len(self.metas)


def build_rays_world(K, c2w, H, W):
    X, Y = np.meshgrid(np.arange(W), np.arange(H))
    XYZ = np.concatenate((X[:, :, None], Y[:, :, None], np.ones_like(X[:, :, None])), axis=-1)
    XYZ = XYZ @ np.linalg.inv(K[:3, :3]).T
    XYZ = XYZ @ c2w[:3, :3].T
    rays_d = XYZ.reshape(-1, 3)
    rays_d = rays_d / np.linalg.norm(rays_d, axis=-1)[:, None]
    rays_o = c2w[:3, 3]
    return np.concatenate((rays_o[None].repeat(len(rays_d), 0), rays_d), axis=-1)

def build_rays_camera(K, H, W):
    X, Y = np.meshgrid(np.arange(W), np.arange(H))
    XYZ = np.concatenate((X[:, :, None], Y[:, :, None], np.ones_like(X[:, :, None])), axis=-1)
    XYZ = XYZ @ np.linalg.inv(K[:3, :3]).T
    rays_d = XYZ.reshape(-1, 3)
    #rays_d = rays_d / np.linalg.norm(rays_d, axis=-1)[:, None]
    rays_o = np.zeros(3)
    a = rays_d.reshape((94, 352,3))
    return np.concatenate((rays_o[None].repeat(len(rays_d), 0), rays_d), axis=-1)


if __name__ == "__main__":
    frame_start = cfg.recenter_start_frame
    frame_num = cfg.recenter_frames
    data_root = 'datasets/KITTI-360'
    gt_static_frames_root = os.path.join(data_root, 'static_frames.txt')
    #visible_id_root = os.path.join(data_root, 'visible_id')
    split = 'train'
    sequence = '0000'
    sequence = os.path.join('2013_05_28_drive_' + sequence + '_sync')
    cam2world_root = os.path.join(data_root, 'data_poses', sequence, 'cam0_to_world.txt')
    print('{0} : {1}'.format(sequence, frame_start))

    render_road = False

    if render_road:
        road_semantic_list = ['ground', 'road', 'sidewalk', 'parking','rail track', 'terrain']
        road_semantic_path = os.path.join(data_root, 'road_semantic')
    else:
        road_semantic_list = ['vegetation','wall' ,'fence','traffic light','traffic sign','pole','polegroup', 'guard rail']
        road_semantic_path = os.path.join(data_root, 'uncountable_semantic')


    
    fig_path = 'tmp/road/birdeyeview'
    if not os.path.exists(road_semantic_path):
        os.makedirs(road_semantic_path)
    road_semantic = Dataset(cam2world_root=cam2world_root,
                                data_root=data_root,
                                sequence=sequence)
    train_loader = torch.utils.data.DataLoader(road_semantic, batch_size=1, shuffle=False, num_workers=0)

    v = Kitti360Viewer3D(path='datasets/KITTI-360', seq=0, load_full=True)
    annotation3D  = v.annotation3D

    camera_dict = {}
    bbx_globalIds_dict = defaultdict(dict)
    annotations_dict = defaultdict(dict)
    road_semantic_dict = defaultdict(dict)
    for i, data in enumerate(train_loader):
        image, masked_image = data['image'][0], data['masked_image'][0]
        camera_mat, world_mat = data['camera_mat'][0].numpy(), data['world_mat'][0].numpy()
        frame_id = int(data['frame_id'][0].numpy())
        bbx_globalIds = data['bbx_globalIds'][0].numpy().tolist()
        bbx_globalIds_dict[frame_id] = bbx_globalIds

        K = np.zeros((4,4))
        K[3,3] = 1
        K[:-1,:-1] = camera_mat
        w2c = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        W, H = data['W'][0].numpy(), data['H'][0].numpy()
        annotations = []
        a = [v.bboxes[i] for i in range(len(v.bboxes)) if v.bboxes_globalId[i] in bbx_globalIds]
        for globalId, obj in annotation3D.objects.items():
            if globalId in bbx_globalIds:

                points = obj[-1].vertices
                R = world_mat[:3,:3]
                T = world_mat[0:3,3]
                o = copy.deepcopy(obj[-1])
                o.vertices = np.matmul(R.T, (points - T).T).T

                annotations.append((globalId, o))

        bbxes = loadBoundingBoxes(annotations)

        # fov set as  pi/2
        pi = 3.1415
        w_bw = l_bw = 100
        focal = sqrt(2) * l_bw /2
        road_ltitude = 1.6
        l_offest = 0
        vis_3d = True
        theta = pi * 5/180
        # visulize camera
        if vis_3d:
            #bbxes.append(open3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[0., 0., 0.]))
            frustum = [get_camera_frustum((W, H), K, w2c, frustum_length=10, color=(255,0,0))]
            camera = frustums2lineset(frustum)
            bbxes.append(camera)
            sphere = open3d.geometry.TriangleMesh.create_sphere(radius=1, resolution=10)
            sphere = open3d.geometry.LineSet.create_from_triangle_mesh(sphere)
            sphere.paint_uniform_color((1, 0, 0))
            bbxes.append(sphere)

            bbxes.append(open3d.geometry.TriangleMesh.create_coordinate_frame(size=5, origin=[0., road_ltitude, 0]))
            #bbxes.append(open3d.geometry.TriangleMesh.create_coordinate_frame(size=5, origin=[10., 1.65, 10]))
            # add boundary lineste
            #!The origin, up at Y, look at -Z
            #! xyz-> rgb
            bbxes.append(open3d.geometry.TriangleMesh.create_coordinate_frame(size=5, origin=[0, -20, 0]))
            bbxes.append(open3d.geometry.TriangleMesh.create_coordinate_frame(size=5, origin=[0, road_ltitude- 1 - l_bw * tan(pi * 5/180)/2, l_bw* cos(pi * 5/180)/2]))
            bbxes.append(open3d.geometry.TriangleMesh.create_coordinate_frame(size=5, origin=[-w_bw/2., road_ltitude, 0]))
            bbxes.append(open3d.geometry.TriangleMesh.create_coordinate_frame(size=5, origin=[w_bw/2.,  road_ltitude, 0]))
            bbxes.append(open3d.geometry.TriangleMesh.create_coordinate_frame(size=5, origin=[-w_bw/2., road_ltitude - l_bw * tan(pi * 5/180) , l_bw * cos(pi * 5/180)]))
            bbxes.append(open3d.geometry.TriangleMesh.create_coordinate_frame(size=5, origin=[w_bw/2., road_ltitude - l_bw * tan(pi * 5/180), l_bw* cos(pi * 5/180)]))

        vis = open3d.visualization.Visualizer()
        vis.create_window(width=1000, height=1000,visible= vis_3d, window_name = '%d'%frame_id)
        for b in bbxes:
            vis.add_geometry(b)
        ctr = vis.get_view_control()
        bv_cam = ctr.convert_to_pinhole_camera_parameters()


        bv_cam_extrinsic = np.zeros((4,4))
        bv_cam_extrinsic[3,3] = 1
        # extrinsic[:3,:3] = create_R((0., 0. , 0. ),(1,1,1))
        # extrinsic[0,3] = -10
        x_bv = 0
        y_bv = road_ltitude - l_bw * sin(theta) * 0.5 - sqrt(3)* 0.5 * l_bw * cos(theta)
        z_bv = l_bw * cos(theta) * 0.5 - sqrt(3) * l_bw * sin(theta) * 0.5
        bv_cam_R = create_R((85/180 * 3.14, 0 , 0 ),(1,1,1)) 
        bv_cam_T = np.array((0, l_bw/2 * cos(pi * 5/180) - focal * sin(pi * 5/180),focal * cos(pi * 5/180) + l_bw/2 * sin(pi * 5/180)))
        bv_cam_extrinsic[:3,:3] = create_R((-85 * pi / 180 , 0 , 0 ),(1,1,1)) 
        bv_cam_extrinsic[:3,3] = np.array((x_bv, y_bv, z_bv))
        bv_cam.extrinsic = np.linalg.inv(bv_cam_extrinsic)
        fov = 60
        focal = 1. / np.tan(0.5 * fov * np.pi/180.)
        bv_cam_intrinsic = np.array(
		    [[sqrt(3)* 500 ,0.0,499.5],
			[0.0,sqrt(3)* 500,499.5],
			[0.0,0.0,1.0]])
        bv_cam.intrinsic.intrinsic_matrix = bv_cam_intrinsic
        #a.extrinsic =  extrinsic @ np.linalg.inv(world_mat)
        ctr.convert_from_pinhole_camera_parameters(bv_cam)
        #b = a.intrinsic.get_focal_length()

        #ctr.change_field_of_view(90)

        rd = vis.get_render_option()
        rd.light_on = False
        rd.background_color = np.array([0,0,0])

        vis.update_geometry(b)
        vis.poll_events()
        vis.update_renderer()
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
        vis.capture_screen_image(os.path.join(fig_path, '%010d'%frame_id + '.jpg'))
        a = vis.capture_screen_float_buffer()
        if vis_3d:
            vis.run()
        del vis, ctr, rd
        #semantic_plane = vis.capture_screen_float_buffer()
        a = (255.0 * np.asarray(a)).astype(np.uint8)
        a = np.unique(a)
        with open(os.path.join(fig_path, '%010d'%frame_id + '.jpg'), 'rb+') as f:
            semantic_plane = imageio.imread(f)
            a = np.unique(semantic_plane)
        semantic_plane_rgb = cv2.resize(semantic_plane, (128, 128), interpolation=cv2.INTER_NEAREST)
        semantic_plane_gray = np.zeros_like(semantic_plane_rgb[...,0])


        # Rgb shot to semantic plane
        #road_semantic_list = ['ground', 'road', 'sidewalk', 'parking','rail track', 'terrain']
        clist = np.zeros((len(road_semantic_list)+1,128,128,3))
        slist = [0]
        for i, sem in enumerate(road_semantic_list):
            clist[i+1,...] = np.repeat(np.array(name2label[sem].color).reshape((1,3)), 128 * 128, axis = 0).reshape(128,128,3)
            slist.append(name2label[sem].id)
        t = np.repeat(semantic_plane_rgb.reshape((1,128,128,3)), len(road_semantic_list)+1, axis=0)
        t = np.sum(np.abs(t - clist), axis=-1).transpose((1,2,0)).argmin(axis = -1)
        semantic_plane_gray = t.astype('u1').T
        for idx, s in enumerate(slist):
            semantic_plane_gray[semantic_plane_gray == idx] = s

        # Build Rays
        semantic_meshgrid = np.zeros((128,128,2))
        semantic_meshgrid[...,0], semantic_meshgrid[...,1] =  np.meshgrid(np.linspace(-w_bw/2, w_bw/2, 128),np.linspace(l_bw, 0, 128), indexing='xy')

        #H, W = 92, 352
        #H, W = 128, 348
        rays = build_rays_camera(K, H, W)

        w_visible = w_bw
        l_visible = l_bw

        road_vertices  = [[w_visible/2,road_ltitude - l_visible * sin(3.1415 * 5/180),l_visible * cos(3.1415 * 5/180)],[w_visible/2,road_ltitude,0],
        [w_visible/2,road_ltitude - l_visible * sin(3.1415 * 5/180), l_visible * cos(3.1415 * 5/180)],[w_visible/2,road_ltitude,0],
        [-w_visible/2,road_ltitude,0],[-w_visible/2,road_ltitude - l_visible * sin(3.1415 * 5/180),l_visible * cos(3.1415 * 5/180)],
        [-w_visible/2,road_ltitude,0],[-w_visible/2,road_ltitude - l_visible * sin(3.1415 * 5/180),l_visible * cos(3.1415 * 5/180)]]
        # v = [[0.5, 0.5, 0.5],[ 0.5,  0.5, -0.5],[ 0.5, -0.5,  0.5],[ 0.5, -0.5, -0.5],
        # [-0.5,  0.5, -0.5],[-0.5,  0.5,  0.5],[-0.5, -0.5, -0.5],[-0.5, -0.5,  0.5]]   
        road_faces = [[0., 2., 1.], [2., 3., 1.], 
        [4., 6., 5.], [6., 7., 5.], 
        [4., 5., 1.], [5., 0., 1.], [7., 6., 2.], [6., 3., 2.],
        [5., 7., 0.], [7., 2., 0.],[1., 3., 4.], [3., 6., 4.]]

        a = rays[..., 3:6].reshape((94,352,3))

    
        road_mesh_tri = trimesh.Trimesh(vertices=road_vertices, faces=road_faces)
        ray_origins, ray_directions = rays[..., 0:3], rays[..., 3:6]
        interaction_loc, index_rays, index_tris = road_mesh_tri.ray.intersects_location(ray_origins=ray_origins, ray_directions=ray_directions)



        # ray idx 32925 (93,189) -> road grid (65,118)

        interaction_loc_xz = interaction_loc[...,(0,2)]


        interaction_loc_xz[...,0] = 128 * (interaction_loc_xz[...,0] + w_bw/2) / w_bw 
        interaction_loc_xz[...,1] = 128 - 128 * interaction_loc_xz[...,1] / l_bw
        interaction_loc_xz = interaction_loc_xz.astype(np.int64)

        a = (80 - interaction_loc_xz[...,1] * 80/128) - interaction_loc[...,2]

        rays_semantic = np.ones_like(ray_origins[:,0]).astype(np.int64) * -1
        rays_intersection = np.zeros_like(ray_origins)
        a = interaction_loc_xz[:,0], interaction_loc_xz[:,1]
        rays_semantic_valid = semantic_plane_gray[interaction_loc_xz[:,0], interaction_loc_xz[:,1]]
        rays_semantic[index_rays] = rays_semantic_valid

        if False:
            plt.subplot(2, 1, 1)
            plt.imshow(rays_semantic.reshape(H, W))
            plt.subplot(2, 1, 2)
            plt.imshow(semantic_plane_gray)
            # plt.subplot(3, 1, 3)
            # plt.scatter(interaction_loc[:,0], interaction_loc[:,2],)
            #plt.scatter(semantic_meshgrid[:,0], semantic_meshgrid[:,1], c = )
            # fig=plt.figure()
            # ttt = np.random.choice(range(interaction_loc.shape[0]), 1000)
            # ttt = np.argwhere(rays_semantic_valid == 7)
            # ax=p3d.Axes3D(fig)
            # ax.scatter(interaction_loc[ttt,0], interaction_loc[ttt,1], interaction_loc[ttt,2], c = rays_semantic_valid[ttt])
            plt.show()

        rays_intersection[index_rays] = interaction_loc

        road_semantic_dict['frame_id'] = frame_id
        road_semantic_dict['size'] = (l_bw, w_bw)
        road_semantic_dict['rays_direction'] = rays[:,3:6]
        road_semantic_dict['rays_semantic'] = rays_semantic
        road_semantic_dict['semantic_plane'] = semantic_plane_gray
        road_semantic_dict['rays_intersection'] = rays_intersection


        if not os.path.exists(os.path.join(road_semantic_path, sequence)):
            os.makedirs(os.path.join(road_semantic_path, sequence))
        with open(os.path.join(road_semantic_path, sequence, '%010d'%frame_id + '.pkl'), 'wb+') as f:
            pkl.dump(road_semantic_dict, f)
        #imageio.imsave(os.path.join(fig_path, '%010d'%frame_id + '.jpg'), semantic_plane_gray)
        #save_tensor_img(img_tnesor=image, save_dir=os.path.join('tmp', 'road', 'raw'), name='%010d.jpg'%(frame_id))
        save_tensor_img(img_tnesor=masked_image, save_dir=os.path.join('tmp', 'road', 'uncountable_only'), name= '%010d.jpg'%(frame_id))


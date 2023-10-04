import numpy as np
import cv2
import random
from lib.config import cfg
from torch import nn
import torch
import collections
from tools.kitti360Scripts.helpers.project import CameraPerspective

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

def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array([0.1, 0.1, 0.1, 1.])
    hwf = c2w[3:, :]
    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        # import ipdb; ipdb.set_trace()
        c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads)
        z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 0))
    return render_poses

def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

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

def convert_id_instance(intersection):
    instance2id = {}
    id2instance = {}
    instances = np.unique(intersection[..., 2])
    for index, inst in enumerate(instances):
        instance2id[index] = inst
        id2instance[inst] = index
    semantic2instance = collections.defaultdict(list)
    semantics = np.unique(intersection[..., 3])
    for index, semantic in enumerate(semantics):
        if semantic == -1:
            continue
        semantic_mask = (intersection[..., 3] == semantic)
        instance_list = np.unique(intersection[semantic_mask, 2])
        for inst in  instance_list:
            semantic2instance[semantic].append(id2instance[inst])
    instances = np.unique(intersection[..., 2])
    instance2semantic = {}
    for index, inst in enumerate(instances):
        if inst == -1:
            continue
        inst_mask = (intersection[..., 2] == inst)
        semantic = np.unique(intersection[inst_mask, 3])
        instance2semantic[id2instance[inst]] = semantic
    instance2semantic[id2instance[-1]] = 23
    return instance2id, id2instance, semantic2instance, instance2semantic

def to_cuda(batch, device=torch.device('cuda:'+str(cfg.local_rank))):
    if isinstance(batch, tuple) or isinstance(batch, list):
        batch = [to_cuda(b, device) for b in batch]
    elif isinstance(batch, dict):
        batch_ = {}
        for key in batch:
            if key == 'meta':
                batch_[key] = batch[key]
            else:
                batch_[key] = to_cuda(batch[key], device)
        batch = batch_
    else:
        batch = batch.to(device)
    return batch

def build_rays(ixt, c2w, H, W, use_cam_cordinate = True):
    X, Y = np.meshgrid(np.arange(W), np.arange(H))
    XYZ = np.concatenate((X[:, :, None], Y[:, :, None], np.ones_like(X[:, :, None])), axis=-1)
    XYZ = XYZ @ np.linalg.inv(ixt[:3, :3]).T
    if not use_cam_cordinate:
        XYZ = XYZ @ c2w[:3, :3].T
    rays_d = XYZ.reshape(-1, 3)
    rays_o = c2w[:3, 3]
    return np.concatenate((rays_o[None].repeat(len(rays_d), 0), rays_d), axis=-1) 


def draw_bbx_boundary(bbxs, image, return_mask=True):
    ''' Draw boundary of 3d bbxes on an image
    '''
    points = []
    depths = []

    bbx_ids = {}
    camera = CameraPerspective(root_dir=cfg)
    #for k,v in annotation3D.objects.items():
    for obj3d in bbxs:
        camera(obj3d, image)
        vertices = np.asarray(obj3d.vertices_proj).T
        points.append(np.asarray(obj3d.vertices_proj).T)
        depths.append(np.asarray(obj3d.vertices_depth))
        for line in obj3d.lines:
            v = [obj3d.vertices[line[0]]*x + obj3d.vertices[line[1]]*(1-x) for x in np.arange(0,1,0.01)]
            uv, d = camera.project_vertices(np.asarray(v), image)
            mask = np.logical_and(np.logical_and(d>0, uv[0]>0), uv[1]>0)
            mask = np.logical_and(np.logical_and(mask, uv[0]<image.shape[1]), uv[1]<image.shape[0])
            #plt.plot(uv[0][mask], uv[1][mask], 'r.')

    if return_mask:
        return  mask
    else:
        return image



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
    if mask_at_box.sum()>0:
        return True
    else:
        return False
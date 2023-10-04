
import numpy as np
import torch
from .transform_utils import RT2tr, create_R
from pytorch3d.structures import Meshes, join_meshes_as_scene
from pytorch3d.renderer import (

    PerspectiveCameras,
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    Textures,
    
)
import matplotlib.pyplot as plt

pi = 3.1415

def build_rays(H, W, K, c2w = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]),z_reverse = False):
    '''
    build rays in opencv camera coordiante
    '''
    assert c2w.shape == (4,4)

    X, Y = np.meshgrid(np.arange(W), np.arange(H))
    XYZ = np.concatenate((X[:, :, None], Y[:, :, None], np.ones_like(X[:, :, None]) * (1 - 2 * z_reverse)), axis=-1)
    XYZ = XYZ @ np.linalg.inv(K[:3, :3]).T
    XYZ = np.concatenate([XYZ,np.ones_like(XYZ[...,0:1])], axis = -1)
    XYZ = XYZ @ c2w.T
    rays_p = XYZ.reshape(-1, 4)[...,:3]
    rays_o = c2w[:3, 3]
    rays_d = (rays_p - rays_o)
    a = rays_d[...,2]
    if True:
        viewdir = rays_d / np.linalg.norm(rays_d, axis=-1)[:, None]

    return np.concatenate((rays_o[None].repeat(len(rays_d), 0), rays_d, viewdir), axis=-1)


def project_vertices(vertices, camera_pose, camera_intrinsic, inverse=True):

    T = camera_pose[:3,  3]
    R = camera_pose[:3, :3]

    # convert points from world coordinate to local coordinate 
    points_local = world2cam(vertices, R, T, inverse)

    # perspective projection
    u,v,depth = cam2image(points_local, camera_intrinsic)

    return (u,v), depth 

def project_3dbbox(bbox_tr, camera_pose, camera_intrinsic, inverse=True, H = 376, W = 1408, render = False, is_kitti = True):
    vertices = bbox2vertices(bbox_tr)
    (u,v), depth = project_vertices(vertices, camera_pose, camera_intrinsic, inverse)
    u.min

    if not render:

        valid_vertices_num = len([i for i in range(8) if ((u[i] >= 0) & (v[i] >= 0) & (u[i] <= W) & (v[i] <= H ))])
        return valid_vertices_num, (u.min(), u.max(), v.min(), v.max())
    else:
        mesh = None
        valid_vertices_num = len([i for i in range(8) if ((u[i] >= 0) & (v[i] >= 0) & (u[i] <= W) & (v[i] <= H ))])
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")

        bbox_tr, c2w = torch.tensor(bbox_tr, device=device, dtype= torch.float32), torch.tensor(camera_pose, device=device, dtype= torch.float32)

        s = 0.5
        bbox_verts = torch.tensor(
        [[s,s,s,1],
        [s,s,-s,1],
        [s,-s,s,1],
        [s,-s,-s,1],
        [-s,s,-s,1],
        [-s,s,s,1],
        [-s,-s,-s,1],
        [-s,-s,s,1]],
        device= device
    )
        # load_objs_as_meshes([obj_filename], device=device)
        vertexes_color =  ((bbox_verts[:,0:3].clone() + s) / (s * 2)).clip(0,1)

        bbox_faces = torch.tensor(
        [[0. ,2., 1.],
        [2. ,3. ,1.],
        [4. ,6. ,5.],
        [6. ,7. ,5.],
        [4. ,5. ,1.],
        [5. ,0. ,1.],
        [7. ,6. ,2.],
        [6. ,3. ,2.],
        [5. ,7. ,0.],
        [7. ,2. ,0.],
        [1. ,3. ,4.],
        [3. ,6. ,4.]],
        device= device)
        

    # urban giraffe  x+ -> left, y+ -> down
    # pytorch3d x+ -> right, y+ -> up
    
    coordinate_tr =  torch.tensor([
    [-1, 0,  0,  0],
    [0, is_kitti * 2 -1,  0,  0],
        [0, 0,  1,  0],
        [0., 0., 0., 1.]],device= device)
    # coordinate transform first
    c2w, bbox_tr =  c2w @ coordinate_tr,  bbox_tr @ coordinate_tr
    w2c = torch.inverse(c2w)

    z_far, z_near = 100. , 1.
    K = torch.tensor([[[camera_intrinsic[0,0],   0.      ,camera_intrinsic[0,2],0],
                    [  0.      , camera_intrinsic[1,1], camera_intrinsic[1,2 ],0],
                    [0., 0.,  0,1],
                    [0., 0., 1.,0]]], device = device, dtype= torch.float32)
    mesh = Meshes(
        verts=[(bbox_verts @ bbox_tr.T)[:,:3]],
        faces=[bbox_faces],
        textures= Textures(verts_rgb = [vertexes_color]))

    mesh = join_meshes_as_scene(mesh)


    cameras = PerspectiveCameras(
        device=device, 
        image_size = [(H,W)],
        R=w2c[:3,:3].unsqueeze(0), 
        T=w2c[:3,3].unsqueeze(0),
        K=K, in_ndc=False)

    raster_settings = RasterizationSettings(
        image_size=(H,W),
        blur_radius=0.0, 
        faces_per_pixel=1,
        z_clip_value=1., # ??????
        cull_to_frustum= False,
        cull_backfaces= False
    )

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device, 
            cameras=cameras,
            # lights=lights
        )
    )

    images = renderer(mesh)[0,:,:,:3]
    plt.imsave('tmp/cow_mesh.jpg',images[..., :3].cpu().numpy())
    return images.cpu().numpy()


def bbox2vertices(bbox_tr):
    vertices_normalize = np.array([[0.5,0.5,0.5,1],[-0.5,-0.5,-0.5,1],[0.5,-0.5,-0.5,1],[-0.5,0.5,-0.5,1],\
        [-0.5,-0.5,0.5,1],[0.5,0.5,-0.5,1],[0.5,-0.5,0.5,1],[-0.5,0.5,0.5,1]])
    vertices = vertices_normalize @ bbox_tr.T
    return vertices[:,:3]

def tr2bbox(bbox_tr):
    vertices_normalize = np.array([[0.5,0.5,0.5,1],[-0.5,-0.5,-0.5,1],[0.5,-0.5,-0.5,1],[-0.5,0.5,-0.5,1],\
        [-0.5,-0.5,0.5,1],[0.5,0.5,-0.5,1],[0.5,-0.5,0.5,1],[-0.5,0.5,0.5,1]])
    vertices = vertices_normalize @ bbox_tr.T
    return vertices[:,:3]


def cam2image(points, K):
    ndim = points.ndim
    if ndim == 2:
        points = np.expand_dims(points, 0)
    points_proj = np.matmul(K[:3,:3].reshape([1,3,3]), points)
    depth = points_proj[:,2,:]
    depth[depth==0] = -1e-6
    u = np.round(points_proj[:,0,:]/np.abs(depth)).astype(np.int)
    v = np.round(points_proj[:,1,:]/np.abs(depth)).astype(np.int)

    if ndim==2:
        u = u[0]; v=v[0]; depth=depth[0]
    return u, v, depth

def world2cam(points, R, T, inverse=False):
    assert (points.ndim==R.ndim)
    assert (T.ndim==R.ndim or T.ndim==(R.ndim-1)) 
    ndim=R.ndim
    if ndim==2:
        R = np.expand_dims(R, 0) 
        T = np.reshape(T, [1, -1, 3])
        points = np.expand_dims(points, 0)
    if not inverse:
        points = np.matmul(R, points.transpose(0,2,1)).transpose(0,2,1) + T
    else:
        points = np.matmul(R.transpose(0,2,1), (points - T).transpose(0,2,1))

    if ndim==2:
        points = points[0]
    return points


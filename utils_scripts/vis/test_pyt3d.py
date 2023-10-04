# Imports
import os
import torch
import matplotlib.pyplot as plt

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj

# Data structures and functions for rendering
import numpy as np
from pytorch3d.structures import Meshes, join_meshes_as_scene
from pytorch3d.renderer import (

    PerspectiveCameras,
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    Textures,
    
)
from  pytorch3d.renderer.cameras import get_screen_to_ndc_transform, try_get_projection_transform
import torchvision

def image_grid(
    images,
    rows=None,
    cols=None,
    fill: bool = True,
    show_axes: bool = False,
    rgb: bool = True,
):
    """
    A util function for plotting a grid of images.

    Args:
        images: (N, H, W, 4) array of RGBA images
        rows: number of rows in the grid
        cols: number of columns in the grid
        fill: boolean indicating if the space between images should be filled
        show_axes: boolean indicating if the axes of the plots should be visible
        rgb: boolean, If True, only RGB channels are plotted.
            If False, only the alpha channel is plotted.

    Returns:
        None
    """
    if (rows is None) != (cols is None):
        raise ValueError("Specify either both rows and cols or neither.")

    if rows is None:
        rows = len(images)
        cols = 1

    gridspec_kw = {"wspace": 0.0, "hspace": 0.0} if fill else {}
    fig, axarr = plt.subplots(rows, cols, gridspec_kw=gridspec_kw, figsize=(15, 9))
    bleed = 0
    fig.subplots_adjust(left=bleed, bottom=bleed, right=(1 - bleed), top=(1 - bleed))

    for ax, im in zip(axarr.ravel(), images):
        if rgb:
            # only render RGB channels
            ax.imshow(im[..., :3])
        else:
            # only render Alpha channel
            ax.imshow(im[..., 3])
        if not show_axes:
            ax.set_axis_off()
    fig.savefig('tmp/cow_mesh.jpg')
    # plt.imsave('tmp/cow_mesh.jpg',fig)
    
# # Setup
# if torch.cuda.is_available():
#     device = torch.device("cuda:0")
#     torch.cuda.set_device(device)
# else:
#     device = torch.device("cpu")
device = torch.device("cpu")
# Set paths
DATA_DIR = "/data/ybyang"
obj_filename = os.path.join(DATA_DIR, "pytorch3d_test/cow_mesh/cow.obj")

# Load obj file
bbox_verts = torch.tensor(
    [[0.5,0.5,0.5,1],
    [0.5,0.5,-0.5,1],
    [0.5,-0.5,0.5,1],
    [0.5,-0.5,-0.5,1],
    [-0.5,0.5,-0.5,1],
    [-0.5,0.5,0.5,1],
    [-0.5,-0.5,-0.5,1],
    [-0.5,-0.5,0.5,1]],
    device= device
)

# load_objs_as_meshes([obj_filename], device=device)
vertexes_color =  (bbox_verts[:,0:3].clone() + 0.5).clip(0,1)

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

bbox_tr =  torch.tensor(
    [[[ 0.33172371, -1.95761569,  0.02566545,  5.32612097],
      [-0.11306915, -0.03746872, -1.47003579, -1.19873105],
      [ 4.82823708,  0.13362189, -0.03619033, 22.25077854],
      [0., 0., 0., 1.]],
      [[ 0.30330239, -2.00150152,  0.02722662,  3.93029587],
      [-0.10631946, -0.03883294, -1.53635274, -0.8853596 ],
      [ 4.47560538,  0.13471606, -0.03834294,  7.84526632],
      [0., 0., 0., 1.]]],device= device)

R_c = torch.tensor([[1., 0., 0.],
[ 0.        ,  0.99619492,  0.08715318],
[ 0.        , -0.08715318,  0.99619492]], device = device)
T_c = torch.tensor([0,-1.55,0], device = device)


coordinate_tr =  torch.tensor([
 [-1, 0,  0,  0],
 [0, -1,  0,  0],
    [0, 0,  1,  0],
    [0., 0., 0., 1.]],device= device)

c2w = torch.eye(4, device= device)
c2w[:3,:3] = R_c
c2w[:3,3] = T_c

c2w, bbox_tr =  c2w @ coordinate_tr,  bbox_tr @ coordinate_tr
# R = (R_c.T)
# T = -(R @ T_c)
w2c = torch.inverse(c2w)


K = torch.tensor([[[552.554261,   0.      ,682.049453,0],
                 [  0.      , 552.554261, 238.769549,0],
                 [0., 0., 0.,1],
                 [0., 0., 1.,0]]], device = device)



mesh = Meshes(
    verts=[(bbox_verts @ bbox_tr[0].T)[:,:3], (bbox_verts @ bbox_tr[1].T)[:,:3]],
    faces=[bbox_faces, bbox_faces],
    textures= Textures(verts_rgb = [vertexes_color,vertexes_color]))

mesh = join_meshes_as_scene(mesh)

cameras = PerspectiveCameras(
    device=device, 
    image_size = [(376,1408)],
    R=w2c[:3,:3].unsqueeze(0), 
    T=w2c[:3,3].unsqueeze(0),
    K=K, in_ndc=False)

raster_settings = RasterizationSettings(
    image_size=(376,1408), 
    blur_radius=0.0, 
    faces_per_pixel=1, 
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

images = renderer(mesh)
plt.imsave('tmp/cow_mesh.jpg',images[0, ..., :3].cpu().numpy())



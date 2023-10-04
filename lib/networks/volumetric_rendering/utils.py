# MIT License

# Copyright (c) 2022 Petr Kellnhofer

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn.functional as F

def transform_vectors(matrix: torch.Tensor, vectors4: torch.Tensor) -> torch.Tensor:
    """
    Left-multiplies MxM @ NxM. Returns NxM.
    """
    res = torch.matmul(vectors4, matrix.T)
    return res


def normalize_vecs(vectors: torch.Tensor) -> torch.Tensor:
    """
    Normalize vector lengths.
    """
    return vectors / (torch.norm(vectors, dim=-1, keepdim=True))

def torch_dot(x: torch.Tensor, y: torch.Tensor):
    """
    Dot product of two tensors.
    """
    return (x * y).sum(-1)


def get_ray_limits_box(rays_o: torch.Tensor, rays_d: torch.Tensor, box_side_length):
    """
    Author: Petr Kellnhofer
    Intersects rays with the [-1, 1] NDC volume.
    Returns min and max distance of entry.
    Returns -1 for no intersection.
    https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection
    """
    o_shape = rays_o.shape
    rays_o = rays_o.detach().reshape(-1, 3)
    rays_d = rays_d.detach().reshape(-1, 3)


    bb_min = [-1*(box_side_length/2), -1*(box_side_length/2), -1*(box_side_length/2)]
    bb_max = [1*(box_side_length/2), 1*(box_side_length/2), 1*(box_side_length/2)]
    bounds = torch.tensor([bb_min, bb_max], dtype=rays_o.dtype, device=rays_o.device)
    is_valid = torch.ones(rays_o.shape[:-1], dtype=bool, device=rays_o.device)

    # Precompute inverse for stability.
    invdir = 1 / rays_d
    sign = (invdir < 0).long()

    # Intersect with YZ plane.
    tmin = (bounds.index_select(0, sign[..., 0])[..., 0] - rays_o[..., 0]) * invdir[..., 0]
    tmax = (bounds.index_select(0, 1 - sign[..., 0])[..., 0] - rays_o[..., 0]) * invdir[..., 0]

    # Intersect with XZ plane.
    tymin = (bounds.index_select(0, sign[..., 1])[..., 1] - rays_o[..., 1]) * invdir[..., 1]
    tymax = (bounds.index_select(0, 1 - sign[..., 1])[..., 1] - rays_o[..., 1]) * invdir[..., 1]

    # Resolve parallel rays.
    is_valid[torch.logical_or(tmin > tymax, tymin > tmax)] = False

    # Use the shortest intersection.
    tmin = torch.max(tmin, tymin)
    tmax = torch.min(tmax, tymax)

    # Intersect with XY plane.
    tzmin = (bounds.index_select(0, sign[..., 2])[..., 2] - rays_o[..., 2]) * invdir[..., 2]
    tzmax = (bounds.index_select(0, 1 - sign[..., 2])[..., 2] - rays_o[..., 2]) * invdir[..., 2]

    # Resolve parallel rays.
    is_valid[torch.logical_or(tmin > tzmax, tzmin > tmax)] = False

    # Use the shortest intersection.
    tmin = torch.max(tmin, tzmin)
    tmax = torch.min(tmax, tzmax)

    # Mark invalid.
    tmin[torch.logical_not(is_valid)] = -1
    tmax[torch.logical_not(is_valid)] = -2

    return tmin.reshape(*o_shape[:-1], 1), tmax.reshape(*o_shape[:-1], 1)


def linspace(start: torch.Tensor, stop: torch.Tensor, num: int):
    """
    Creates a tensor of shape [num, *start.shape] whose values are evenly spaced from start to end, inclusive.
    Replicates but the multi-dimensional bahaviour of numpy.linspace in PyTorch.
    """
    # create a tensor of 'num' steps from 0 to 1
    steps = torch.arange(num, dtype=torch.float32, device=start.device) / (num - 1)

    # reshape the 'steps' tensor to [-1, *([1]*start.ndim)] to allow for broadcastings
    # - using 'steps.reshape([-1, *([1]*start.ndim)])' would be nice here but torchscript
    #   "cannot statically infer the expected size of a list in this contex", hence the code below
    for i in range(start.ndim):
        steps = steps.unsqueeze(-1)

    # the output starts at 'start' and increments until 'stop' in each dimension
    out = start[None] + steps * (stop - start)[None]

    return out

def trans_vec_homo(m, v, is_vec=False, normalize = True):
    r"""3-dimensional Homogeneous matrix and regular vector multiplication
    Convert v to homogeneous vector, perform M-V multiplication, and convert back
    Note that this function does not support autograd.

    Args:
        m (4 x 4 tensor): a homogeneous matrix
        v (3 tensor): a 3-d vector
        vec (bool): if true, v is direction. Otherwise v is point
    """
    if is_vec:
        v = torch.tensor([v[0], v[1], v[2], 0], dtype=v.dtype).to(v.device)
    else:
        v = torch.tensor([v[0], v[1], v[2], 1], dtype=v.dtype).to(v.device)
    v = torch.mv(m, v)
    # v = torch.mv(m.to(torch.float32), v.to(torch.float32))
    if not is_vec:
        v = v / v[3]
    elif normalize:
        v = v[:3] / torch.sqrt(v[0] * v[0]+ v[1]* v[1]+ v[2] * v[2])

    v = v[:3]
    return v

def trans_vec_homo_batch(m, v, is_vec=False, normalize = True):
    r"""3-dimensional Homogeneous matrix and regular vector multiplication
    Convert v to homogeneous vector, perform M-V multiplication, and convert back
    Note that this function does not support autograd.

    Args:
        m (4 x 4 tensor): a homogeneous matrix
        v (3 tensor): a 3-d vector
        vec (bool): if true, v is direction. Otherwise v is point
    """
    m = m.to(v.device).to(torch.float32)
    v = v.to(torch.float32)
    batch_size = v.shape[0]
    if is_vec:
        v = torch.cat((v, torch.zeros_like(v[:,0:1])), dim=1)
    else:
        v =torch.cat((v, torch.ones_like(v[:,0:1])), dim=1)
    v = (m @ v.T).T 
    if not is_vec:
        v = v[:] / v[:,3,None]
    elif normalize:
        v = v / torch.norm(v[...,None,:] , dim = -1)

    v = v[:,:3]
    return v


# def sample_from_2dmap(map,coordinates, mode='bilinear', padding_mode='zeros', grid_form_return = False):
#     """
#     Expects coordinates in shape (batch_size, H, W, 2)
#     Expects grid in shape (batch_size, H_, W_, D_, channels)
#     (Also works if grid has batch size)
#     Returns sampled features of shape (batch_size, H_, W_, D_, feature_channels)
#     """
#     pass


    assert len(grid.shape) == 5
    if len(coordinates.shape) == 5:
        pass
    elif len(coordinates.shape) == 3:
        coordinates = coordinates.reshape(coordinates.shape[0],coordinates.shape[1],1,1,3)
    else:
        raise AttributeError
    batch_size, H_, W_, D_, _ = coordinates.shape
    batch_size, H, W, D, C = grid.shape
    coordinates_ = coordinates.to(torch.float32)
    grid_ = grid.permute((0,4,3,1,2)).to(torch.float32)

    sampled_features = F.grid_sample(input = grid_,
                                    grid = coordinates_,
                                    mode=mode, 
                                    padding_mode=padding_mode,
                                    align_corners=False) 

    sampled_features = sampled_features.permute(0, 2, 3, 4, 1)
    if not grid_form_return:
        sampled_features = sampled_features.reshape((batch_size, -1, C))
    return sampled_features

def sample_from_planes(plane_features, coordinates,plane_axes = None, mode='bilinear', padding_mode='zeros', box_warp=2):
    assert padding_mode == 'zeros'
    if plane_axes == None:
        plane_axes = generate_planes().to(coordinates.device)
    N, n_planes, C, H, W = plane_features.shape
    _, M, _ = coordinates.shape
    plane_features = plane_features.view(N*n_planes, C, H, W)

    coordinates = (2/box_warp) * coordinates # TODO: add specific box bounds

    projected_coordinates = project_onto_planes(plane_axes, coordinates).unsqueeze(1)
    output_features = torch.nn.functional.grid_sample(plane_features, projected_coordinates.float(), mode=mode, padding_mode=padding_mode, align_corners=False).permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
    return output_features

def sample_from_3dgrid(grid, coordinates, mode='bilinear', padding_mode='zeros', grid_form_return = False):
    """
    Expects coordinates in shape (batch_size, H, W, D, 3)
    Expects grid in shape (batch_size, H_, W_, D_, channels)
    (Also works if grid has batch size)
    Returns sampled features of shape (batch_size, H_, W_, D_, feature_channels)
    """
    assert len(grid.shape) == 5
    if len(coordinates.shape) == 5:
        pass
    elif len(coordinates.shape) == 3:
        coordinates = coordinates.reshape(coordinates.shape[0],coordinates.shape[1],1,1,3)
    else:
        raise AttributeError
    batch_size, H_, W_, D_, _ = coordinates.shape
    batch_size, H, W, D, C = grid.shape
    coordinates_ = coordinates.to(torch.float32)
    grid_ = grid.permute((0,4,3,1,2)).to(torch.float32)

    sampled_features = F.grid_sample(input = grid_,
                                    grid = coordinates_,
                                    mode=mode, 
                                    padding_mode=padding_mode,
                                    align_corners=False) 

    sampled_features = sampled_features.permute(0, 2, 3, 4, 1)
    if not grid_form_return:
        sampled_features = sampled_features.reshape((batch_size, -1, C))
    return sampled_features


def generate_planes():
    """
    Defines planes by the three vectors that form the "axes" of the
    plane. Should work with arbitrary number of planes and planes of
    arbitrary orientation.
    """
    return torch.tensor([[[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]],
                            [[1, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0]],
                            [[0, 0, 1],
                            [1, 0, 0],
                            [0, 1, 0]]], dtype=torch.float32)

def project_onto_planes(planes, coordinates):
    """
    Does a projection of a 3D point onto a batch of 2D planes,
    returning 2D plane coordinates.

    Takes plane axes of shape n_planes, 3, 3
    # Takes coordinates of shape N, M, 3
    # returns projections of shape N*n_planes, M, 2
    """
    N, M, C = coordinates.shape
    n_planes, _, _ = planes.shape
    coordinates = coordinates.unsqueeze(1).expand(-1, n_planes, -1, -1).reshape(N*n_planes, M, 3)
    inv_planes = torch.linalg.inv(planes).unsqueeze(0).expand(N, -1, -1, -1).reshape(N*n_planes, 3, 3)
    projections = torch.bmm(coordinates, inv_planes)
    return projections[..., :2]


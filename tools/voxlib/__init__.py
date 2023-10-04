# Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
# import torch
# from torch.nn import functional as F
# from torch.autograd import Function
# from torch.utils.cpp_extension import load

# upfirdn2d_op = load('voxlib', sources=[
#     'tools/voxlib/ray_voxel_intersection.cu'], verbose=True)

from .positional_encoding import positional_encoding
from .sp_trilinear import sparse_trilinear_interp_worldcoord
from voxlib import ray_voxel_intersection_perspective


from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# cxx_args = ['-fopenmp']
# nvcc_args = []
if os.name == "posix":
    cxx_args = ['-O3', '-std=c++14']
elif os.name == "nt":
    cxx_args = ['/O2', '/std:c++17']

nvcc_args = [
    '-O3', '-std=c++14',
    '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_HALF2_OPERATORS__',
]

'''
Usage:

python setup.py build_ext --inplace # build extensions locally, do not install (only can be used from the parent directory)

python setup.py install # build extensions and install (copy) to PATH.
pip install . # ditto but better (e.g., dependency & metadata handling)

python setup.py develop # build extensions and install (symbolic) to PATH.
pip install -e . # ditto but better (e.g., dependency & metadata handling)
'''

setup(
    name='voxrender',
    ext_modules=[
        CUDAExtension('voxlib', [
            'voxlib.cpp',
            'ray_voxel_intersection.cu',
            'sp_trilinear_worldcoord_kernel.cu',
            'positional_encoding_kernel.cu'
        ],
            extra_compile_args={'cxx': cxx_args, 
                                'nvcc': nvcc_args}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

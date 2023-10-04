# from .native_ops import FusedLeakyReLU, fused_leaky_relu, upfirdn2d
try:
    from .upfirdn2d import upfirdn2d
    from .fused_act import FusedLeakyReLU, fused_leaky_relu
    print('Using custom CUDA kernels')
except Exception as e:
    print(str(e))
    print('There was something wrong with the CUDA kernels')
    print('Reverting to native PyTorch implementation')
    from .native_ops import FusedLeakyReLU, fused_leaky_relu, upfirdn2d

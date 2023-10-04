import os
import imp
from lib.networks import urbangiraffe  #, bboxNeRF, GAN2d

network_dict = {
    # 'bboxNeRF': bboxNeRF.Network,
    'urbangiraffe': urbangiraffe.Network,
    # '2DGAN': GAN2d.Network
    }
#     'GIRAFFE':0,
#     'GANCraft':0,
# }

def make_network(cfg):
    # module = cfg.network_module
    # path = cfg.network_path
    # network = imp.load_source(module, path).Network()
    network = network_dict[cfg.network_type]()
    return network


# from lib.train.trainers.bboxNeRF import Trainer as bboxNeRFtrainer
from lib.train.trainers.urbangiraffe import Trainer as urbangiraffetrainer
# from lib.train.losses.bboxNeRF import NetworkWrapper  as bboxNeRFloss
from lib.train.losses.urbangiraffe import  NetworkWrapper as urbangiraffeloss
import imp


trainer_dict = {
    # 'bboxNeRF': bboxNeRFtrainer,
    'urbangiraffe': urbangiraffetrainer
}

loss_dict = {
    # 'bboxNeRF': bboxNeRFloss,
    'urbangiraffe': urbangiraffeloss
}

def _wrapper_factory(cfg, network):
    network_wrapper = loss_dict[cfg.loss_type](network)
    return network_wrapper

def make_trainer(cfg, network):
    network = _wrapper_factory(cfg, network)
    return trainer_dict[cfg.trainer_type](network)
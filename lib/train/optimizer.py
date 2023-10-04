from typing import Mapping
import torch
from lib.utils.optimizer.radam import RAdam
import re 

_optimizer_factory = {
    'adam': torch.optim.Adam,
    'radam': RAdam,
    'sgd': torch.optim.SGD,
    'rmsprop':  torch.optim.RMSprop
}

def get_optimizer_GRAF(cfg, net):
    optimizer = {}
    lr = cfg.train.lr
    lr_d = cfg.train.lr_d
    #weight_decay = cfg.train.weight_decay
    op = _optimizer_factory[cfg.train.optim]
    weight_decay = cfg.train.weight_decay
    # parameters_g = net.generator.parameters()
    # parameters_d = net.discriminator.parameters()
    params_g, params_d = [], []
    for key, value in net.generator.named_parameters():
        if not value.requires_grad:
            continue

        params_g += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    for key, value in net.discriminator.named_parameters():
        if not value.requires_grad:
            continue
        # elif re.search('map', key) != None: 
        #     param_lr = lr_d * cfg.train.mapping_lr_mul 
        # else:
        #     param_lr = lr_d
        params_d += [{"params": [value], "lr": lr_d, "weight_decay": weight_decay}]
    if cfg.use_patch_discriminator:
        for key, value in net.discriminator_obj.named_parameters():
            if not value.requires_grad:
                continue
            elif re.search('map', key) != None: 
                param_lr = lr_d * cfg.train.mapping_lr_mul 
            else:
                param_lr = lr_d
            params_d += [{"params": [value], "lr": param_lr, "weight_decay": weight_decay}]

    if 'adam' in cfg.train.optim:
        optimizer['op']= op(params_g, betas=[cfg.train.beta1, cfg.train.beta2],weight_decay=weight_decay)
        optimizer['op_d']= op(params_d, betas=[cfg.train.beta1, cfg.train.beta2], weight_decay=weight_decay)
    else:  
        optimizer['op']= op(params_g, lr=lr)
        optimizer['op_d']= op(params_d, lr=lr_d)

    return optimizer


def get_optimizer_inverse(cfg, net):
    optimizer = {}
    params = []
    lr = cfg.train.lr
    weight_decay = cfg.train.weight_decay
    op = _optimizer_factory[cfg.train.optim]
    
    for key, value in net.named_parameters():
        if not 'z_global' in key:
            value.requires_grad = False
        else:
            value.requires_graf = True
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    
    if 'adam' in cfg.train.optim:
        optimizer['op'] = op(params, lr, weight_decay=weight_decay)
    else:
        optimizer['op'] = op(params, lr, momentum=0.9)
        
    return optimizer
     

def get_optimizer_NeRF(cfg, net):
    optimizer = {}
    params = []
    lr = cfg.train.lr
    weight_decay = cfg.train.weight_decay

    for key, value in net.named_parameters():
        if not value.requires_grad:
            continue
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if 'adam' in cfg.train.optim:
        optimizer['op'] = _optimizer_factory[cfg.train.optim](params, lr, weight_decay=weight_decay)
    else:
        optimizer['op'] = _optimizer_factory[cfg.train.optim](params, lr, momentum=0.9)

    return optimizer


def make_optimizer(cfg, net):
    if cfg.optimizer_type =='urbangiraffe':
        optimizer = get_optimizer_GRAF(cfg, net)
    elif cfg.optimizer_type == 'inversion':
        optimizer = get_optimizer_inverse(cfg, net)
    elif cfg.optimizer_type =='bboxNeRF':
        optimizer = get_optimizer_NeRF(cfg, net)
    else:
        raise TypeError

    return optimizer

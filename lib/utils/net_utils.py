import imp
from multiprocessing.spawn import import_main_path
import torch
import os
from torch import nn
import numpy as np
import torch.nn.functional
from collections import OrderedDict
from termcolor import colored
import sys
import yaml
from lib.config import cfg
import os
import re


def load_model(net,
               optim,
               scheduler,
               recorder,
               model_dir,
               resume=True,
               epoch=-1):
    if not resume:
        os.system('rm -rf {}'.format(model_dir))

    if not os.path.exists(model_dir):
        return 0

    pths = [
        int(pth.split('.')[0]) for pth in os.listdir(model_dir)
        if pth != 'latest.pth'
    ]
    if len(pths) == 0 and 'latest.pth' not in os.listdir(model_dir):
        return 0
    if epoch == -1:
        if 'latest.pth' in os.listdir(model_dir):
            pth = 'latest'
        else:
            pth = max(pths)
    else:
        pth = epoch
    print('load model: {}'.format(os.path.join(model_dir, '{}.pth'.format(pth))))
  
    # if cfg.use_cuda:
    #     # torch.cuda.empty_cache()
    #     pretrained_model = torch.load(os.path.join(model_dir, '{}.pth'.format(pth)), map_location="cuda:"+str(cfg.local_rank))
    # else:
    pretrained_model = torch.load(
            os.path.join(model_dir, '{}.pth'.format(pth)), 'cpu')
    new_state_dict = OrderedDict()
    cur_dict = net.state_dict()
    #  cur_dict = dict(net.named_parameters())
    for k,v in pretrained_model['net'].items():
        if k in cur_dict.keys() and cur_dict[k].shape == pretrained_model['net'][k].shape:
            new_state_dict[k]=v
        else:
            print(k+ " not in currunt model, skip it")
    net.load_state_dict(new_state_dict, strict=False)

    # net.load_state_dict(pretrained_model['net'], strict=False)
    if 'op' in pretrained_model:
        optim['op'].load_state_dict(pretrained_model['op'])
        scheduler['op'].load_state_dict(pretrained_model['scheduler']['op'])
        if 'op_d' in optim:
             optim['op_d'].load_state_dict(pretrained_model['op_d'])
             scheduler['op_d'].load_state_dict(pretrained_model['scheduler']['op_d'], )
        
        recorder.load_state_dict(pretrained_model['recorder'])
        return pretrained_model['epoch'] + 1
    else:
        return 0


def save_model(net, optim, scheduler, recorder, model_dir, epoch, last=False):
    os.system('mkdir -p {}'.format(model_dir))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model = {
        'net': net.state_dict(),
        'scheduler': {k : scheduler[k].state_dict() for k in scheduler},
        'recorder': recorder.state_dict(),
        'epoch': epoch
    }
    for k in optim.keys():
        model[k] = optim[k].state_dict()
    if last:
        torch.save(model, os.path.join(model_dir, 'latest.pth'))
    else:
        torch.save(model, os.path.join(model_dir, '{}.pth'.format(epoch)))

    # remove previous pretrained model if the number of models is too big
    pths = [
        int(pth.split('.')[0]) for pth in os.listdir(model_dir)
        if pth != 'latest.pth'
    ]
    if len(pths) <= 5:
        return
    os.system('rm {}'.format(
        os.path.join(model_dir, '{}.pth'.format(min(pths)))))


def load_network(net, model_dir, resume=True, epoch=-1, strict=False, load_D=False, load_p=False):
    if not resume:
        return 0
    if not os.path.exists(model_dir):
        print(colored('pretrained model does not exist', 'red'))
        return 0

    if os.path.isdir(model_dir):
        pths = [
            int(pth.split('.')[0]) for pth in os.listdir(model_dir)
            if pth != 'latest.pth'
        ]
        if len(pths) == 0 and 'latest.pth' not in os.listdir(model_dir):
            return 0
        if epoch == -1:
            if 'latest.pth' in os.listdir(model_dir):
                pth = 'latest'
            else:
                pth = max(pths)
        else:
            pth = epoch
        model_path = os.path.join(model_dir, '{}.pth'.format(pth))
    else:
        model_path = model_dir

    print('load init model: {}'.format(model_path))
    if cfg.use_cuda:
        # torch.cuda.empty_cache()
        pretrained_model = torch.load(model_path, map_location="cuda:"+str(cfg.local_rank))
    else:
        pretrained_model = torch.load(model_path, map_location="cpu")
    new_state_dict = OrderedDict()
    # cur_dict = dict(net.named_parameters())
    cur_dict = net.state_dict()
    for k,v in pretrained_model['net'].items():
        if k in cur_dict.keys() and cur_dict[k].shape == pretrained_model['net'][k].shape and (load_D or re.search('generator', k) != None):
            new_state_dict[k]=v
        else:
            # print (cur_dict[k].shape + pretrained_model['net'][k].shape )
            print(k+ " not in currunt model, skip it")
    if not load_p and 'aug.p' in new_state_dict:
        del new_state_dict['aug.p']
    if not load_p and 'aug_obj.p' in new_state_dict:
        del new_state_dict['aug_obj.p']

    net.load_state_dict(new_state_dict, strict=False)
    if 'epoch' in pretrained_model:
        return pretrained_model['epoch'] + 1
    else:
        return 0


def remove_net_prefix(net, prefix):
    net_ = OrderedDict()
    for k in net.keys():
        if k.startswith(prefix):
            net_[k[len(prefix):]] = net[k]
        else:
            net_[k] = net[k]
    return net_


def add_net_prefix(net, prefix):
    net_ = OrderedDict()
    for k in net.keys():
        net_[prefix + k] = net[k]
    return net_


def replace_net_prefix(net, orig_prefix, prefix):
    net_ = OrderedDict()
    for k in net.keys():
        if k.startswith(orig_prefix):
            net_[prefix + k[len(orig_prefix):]] = net[k]
        else:
            net_[k] = net[k]
    return net_


def remove_net_layer(net, layers):
    keys = list(net.keys())
    for k in keys:
        for layer in layers:
            if k.startswith(layer):
                del net[k]
    return net

def save_trained_config(cfg):
    if not cfg.resume:
        os.system('rm -rf ' + cfg.trained_config_dir+'/*')
    os.system('mkdir -p ' + cfg.trained_config_dir)
    train_cmd = ' '.join(sys.argv)
    train_cmd_path = os.path.join(cfg.trained_config_dir, 'train_cmd.txt')
    train_config_path = os.path.join(cfg.trained_config_dir, 'train_config.yaml')
    open(train_cmd_path, 'w').write(train_cmd)
    yaml.dump(cfg, open(train_config_path, 'w'))

def load_pretrain(net, model_dir):

    model_dir = os.path.join('data/trained_model', cfg.task, model_dir)
    if not os.path.exists(model_dir):
        return 1

    pths = [int(pth.split('.')[0]) for pth in os.listdir(model_dir) if pth != 'latest.pth']
    if len(pths) == 0 and 'latest.pth' not in os.listdir(model_dir):
        return 1

    if 'latest.pth' in os.listdir(model_dir):
        pth = 'latest'
    else:
        pth = max(pths)

    print('Load pretrain model: {}'.format(os.path.join(model_dir, '{}.pth'.format(pth))))
    pretrained_model = torch.load(os.path.join(model_dir, '{}.pth'.format(pth)), 'cpu')
    net.load_state_dict(pretrained_model['net'])
    return 0

def save_pretrain(net, task, model_dir):
    model_dir = os.path.join('data/trained_model', task, model_dir)
    os.system('mkdir -p ' +  model_dir)
    model = {'net': net.state_dict()}
    torch.save(model, os.path.join(model_dir, 'latest.pth'))
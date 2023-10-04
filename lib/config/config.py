from tracemalloc import start
from cv2 import exp
from .yacs import CfgNode as CN
import argparse
import os
import numpy as np

cfg = CN()

# task settings
cfg.eval_mode = 0
cfg.input_sparse = False
cfg.combine_3d_2d = 1
cfg.save_img = False
cfg.N_rays = 2048
cfg.ft_scene = ''
cfg.eval_setting = 'enerf' # ['mvsnerf', 'enerf']
cfg.depth_inv = True
cfg.render_scale = 1.0
cfg.train_frames = 100
cfg.test_frames = 5
cfg.recenter_start_frame = 1070
cfg.recenter_frames = 64
cfg.intersection_start_frame = 1070
cfg.intersection_frames = 64
cfg.render_cam = -1
cfg.sample_more_onmask = False
cfg.use_pspnet = True
cfg.enhance_3d = False
cfg.val_list = []
cfg.depth_object = []
cfg.start = 6060
cfg.decay_rate = 1.
cfg.lambda_depth = 0.1
cfg.bbox_sp = 10
cfg.mode = 0
cfg.mask_parking = True
cfg.semantic_weight = 0.0005
cfg.weight_th = 0.
cfg.dist_lambda = 0.005
cfg.xyz_res = 6
cfg.view_res = 4
cfg.lambda_fix = 1.
cfg.lambda_semantic_2d = 1.
cfg.lambda_3d = 1.
cfg.crf_seq = -1
cfg.consistency_thres = -1.
cfg.pseudo_filter = False
cfg.use_decay = True
cfg.train_baseline1 = False
cfg.only_baseline2 = False
cfg.lidar_frames = 1
cfg.samples_all = 192
cfg.vis_index = 0
cfg.vis_x = 298
cfg.vis_y = 97
cfg.vis_depth = 1
cfg.center_pose = []
cfg.cam_interval = 1
cfg.log_sample = False
cfg.use_pspnet = False
cfg.postprocessing = False
cfg.max_depth = -1.
cfg.lidar_samples = 64
cfg.detach = True
cfg.use_depth = True
cfg.use_stereo = True
cfg.dist = 300
cfg.sampling_change = False
cfg.test_start = 7300
cfg.iterative_train = False
cfg.render_instance = False
cfg.init_network = False
cfg.init_name = 'None'
cfg.trained_model_dir_init = 'data/trained_model'
cfg.lidar_depth_root = ''
cfg.semantic_gt_root = ''
cfg.panoptic_gt_root = ''

# module
cfg.train_dataset_module = 'lib.datasets.dtu.neus'
cfg.test_dataset_module = 'lib.datasets.dtu.neus'
cfg.val_dataset_module = 'lib.datasets.dtu.neus'
cfg.network_module = 'lib.neworks.neus.neus'
cfg.loss_module = 'lib.train.losses.neus'
cfg.evaluator_module = 'lib.evaluators.neus'
cfg.test_start = -1
# experiment name
cfg.exp_name = 'gittag_hello'
cfg.pretrain = ''

# network
cfg.distributed = False

# task
cfg.task = 'hello'

# gpus
cfg.gpus = list(range(4))
# if load the pretrained network
cfg.resume = False

# epoch
cfg.ep_iter = -1
cfg.save_ep = 1
cfg.save_latest_ep = 1
cfg.eval_ep = 1
log_interval : 20

# -----------------------------------------------------------------------------
# train
# -----------------------------------------------------------------------------

cfg.train = CN()
cfg.train.dataset = 'CocoTrain'
cfg.train.epoch = 10000
cfg.train.num_workers = 0
cfg.train.collator = 'default'
cfg.train.batch_sampler = 'default'
cfg.train.sampler_meta = CN({'min_hw': [256, 256], 'max_hw': [480, 640], 'strategy': 'range'})
cfg.train.shuffle = True

cfg.train.weight_color = 1.

# use adam as default
cfg.train.optim = 'adam'
cfg.train.lr = 1e-4
cfg.train.weight_decay = 0.
cfg.train.scheduler = CN({'type': 'multi_step', 'milestones': [80, 120, 200, 240], 'gamma': 0.5})
cfg.train.batch_size = 4
cfg.train.acti_func = 'relu'
cfg.frozen = False

# test
cfg.test = CN()
cfg.test.dataset = 'CocoVal'
cfg.test.val_dataset = ''
cfg.test.batch_size = 1
cfg.test.collator = 'default'
cfg.test.epoch = -1
cfg.test.batch_sampler = 'default'
cfg.test.sampler_meta = CN({'min_hw': [480, 640], 'max_hw': [480, 640], 'strategy': 'origin'})

# evaluation
cfg.skip_eval = False
cfg.fix_random = False


def get_exp_info(cfg, args):

    device = args.use_cuda
    exp_info_list = []

    if cfg.exp_comment != '':
        exp_info_list.append(cfg.exp_comment)
    else:
        exp_info_list.append('default')
    if cfg.is_debug:
        exp_info_list.append('debug')

    if cfg.train.use_trajectory:
        exp_info_list.append("trajectory")

    if cfg.network_kwargs.aug_p > 0:
        exp_info_list.append("aug_p:%.2f"%cfg.network_kwargs.aug_p)

    for k in cfg.train.loss_weight:
        exp_info_list.append("%s:%.1f"%(k, cfg.train.loss_weight[k]))

    exp_info_list.append('%.2f,%d'%(cfg.ratio,cfg.super_resolution))
    if cfg.task == 'bboxNeRF':
        if cfg.network_kwargs.generator_kwargs.use_neural_renderer:
            exp_info_list.append('NR')
        if 'start' in cfg.keys():
            # exp_info_list.append('s' + str(cfg.start))
            exp_info_list.append('fnum:' + str(cfg.train.frame_num))  
        if cfg.render_obj:
            if cfg.network_kwargs.generator_kwargs.obj_decoder_type == 'eg3d':
                exp_info_list.append('eg3d')
            exp_info_list.append('semantcis:' + ','.join(cfg.valid_object))
            exp_info_list.append('N:%d'%cfg.max_obj_num)  
            exp_info_list.append('pixel:%d'%cfg.min_visible_pixel)       
        # if cfg.render_road:
        #     exp_info_list.append('road')
        #     exp_info_list.append(str(cfg.network_kwargs.generator_kwargs.z_map_generator_type)) 
        #     exp_info_list.append('z_global:' + str(cfg.network_kwargs.generator_kwargs.z_dim_global))
            # exp_info_list.append('z_trainable:' + str(cfg.network_kwargs.generator_kwargs.z_trainable))      
            exp_info_list.append('image_nc:' + str(cfg.network_kwargs.discriminator_kwargs.in_channels)) 
        if cfg.render_stuff:  
            exp_info_list.append('stuff')
            # exp_info_list.append('max_nc:' + str(cfg.network_kwargs.generator_kwargs.z_grid_generator_kwargs.max_nc))
            # if not cfg.network_kwargs.generator_kwargs.ray_voxel_sampling:
            #     exp_info_list.append('uniform%d;'% cfg.network_kwargs.generator_kwargs.n_samples_stuff)

    elif cfg.task == 'kitti-360':
        
        # if not cfg.network_kwargs.generator_kwargs.use_occupancy_mask:
        #     exp_info_list.append('no_mask')
        # if cfg.network_kwargs.use_depth:
        #     exp_info_list.append('depth')
        # if cfg.network_kwargs.generator_kwargs.use_neural_renderer:
        #     exp_info_list.append('NR%d'%cfg.network_kwargs.generator_kwargs.feature_dim)
        # if cfg.network_kwargs.use_color_aug:
        #     exp_info_list.append("C")
        # if cfg.render_stuff:  
        #     exp_info_list.append('stuff')
        #     if cfg.network_kwargs.generator_kwargs.z_grid_generator_kwargs.h != 64:
        #         exp_info_list.append(str(cfg.network_kwargs.generator_kwargs.z_grid_generator_kwargs.h))
        #     if cfg.network_kwargs.generator_kwargs.z_grid_generator_kwargs.use_oasis:
        #         exp_info_list.append('OASIS')
        #     if cfg.network_kwargs.generator_kwargs.z_grid_generator_kwargs.use_uncondition_layer:
        #         exp_info_list.append('uncondition')
        #     exp_info_list.append('nc:%d'%cfg.network_kwargs.generator_kwargs.z_grid_generator_kwargs.max_nc) 
        #     exp_info_list.append('f_dim:%d'%cfg.network_kwargs.generator_kwargs.z_grid_generator_kwargs.ngf) 
        #     exp_info_list.append('nc:%d'%cfg.network_kwargs.generator_kwargs.z_grid_generator_kwargs.max_nc) 
        #     exp_info_list.append('nl:%d'%cfg.network_kwargs.generator_kwargs.stuff_decoder_kwargs.n_block_num) 
        #     if not cfg.network_kwargs.generator_kwargs.stuff_decoder_kwargs.use_positonal_encoding:
        #         exp_info_list.append('no_pe') 
        #     if not cfg.network_kwargs.generator_kwargs.stuff_decoder_kwargs.use_seg:
        #          exp_info_list.append('no_seg') 
        #     if cfg.network_kwargs.generator_kwargs.stuff_decoder_kwargs.use_pts:
        #         exp_info_list.append('pts')
        #     if not cfg.network_kwargs.generator_kwargs.ray_voxel_sampling:
        #         exp_info_list.append('uniform') 
        if cfg.render_obj:
            if cfg.network_kwargs.generator_kwargs.obj_decoder_type == 'eg3d':
                exp_info_list.append('eg3d')
            exp_info_list.append('semantcis:' + ','.join(cfg.valid_object))
            exp_info_list.append('N:%d'%cfg.max_obj_num)  
            exp_info_list.append('pixel:%d'%cfg.min_visible_pixel['car'])  
        # if cfg.render_road:
        #     exp_info_list.append('road')
        #     exp_info_list.append(str(cfg.network_kwargs.generator_kwargs.z_map_generator_type)) 
        #     exp_info_list.append('z_global:%d'%cfg.network_kwargs.generator_kwargs.z_dim_global)
        # if cfg.render_sky:  
        #     exp_info_list.append('sky')
            # exp_info_list.append('z_trainable:' + str(cfg.network_kwargs.generator_kwargs.z_trainable))      
            # exp_info_list.append('image_nc:%d'%cfg.network_kwargs.discriminator_kwargs.in_channels)      
    # elif cfg.task == 'urbanGIRAFFE_2d':
    #     exp_info_list.append('seg_scale:%f'%cfg.semantic2d_scale) 
    #     exp_info_list.append(cfg.network_kwargs.generator_kwargs.feature_type) 
    #     exp_info_list.append(cfg.network_kwargs.generator_kwargs.pts_type) 
    # else:
    #     exp_info_list.append('default')
    exp_info = '_'.join(exp_info_list)
    return exp_info

def parse_cfg(cfg, args):
    if len(cfg.task) == 0:
        raise ValueError('task must be specified')
    # assign the gpus
    if args.gpus != []:
         cfg.gpus = args.gpus

    os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join([str(gpu) for gpu in cfg.gpus])
    print(os.environ['CUDA_VISIBLE_DEVICES'])
    # os.environ['OMP_NUM_THREADS'] = '4'
    cfg.exp_name = cfg.exp_name.replace('gittag', os.popen('git describe --tags --always').readline().strip())
    cfg.exp_info = get_exp_info(cfg, args)
    cfg.trained_model_dir_init = os.path.join(cfg.trained_model_dir, cfg.task, cfg.exp_name,cfg.init_name)
    cfg.trained_model_dir = os.path.join(cfg.trained_model_dir, cfg.task, cfg.exp_name, cfg.exp_info)    
    cfg.trained_config_dir = os.path.join(cfg.trained_config_dir, cfg.task, cfg.exp_name, cfg.exp_info)
    cfg.record_dir = os.path.join(cfg.record_dir, cfg.task, cfg.exp_name)
    cfg.result_dir = os.path.join(cfg.result_dir, cfg.task, cfg.exp_name)
    cfg.local_rank = args.local_rank
    cfg.use_cuda = args.use_cuda
    cfg.is_debug = args.is_debug

    cfg.DP = args.DP # Data parallel
    cfg.distributed = args.distributed
    cfg.load_D = args.load_D

    modules = [key for key in cfg if '_module' in key]
    for module in modules:
        cfg[module.replace('_module', '_path')] = cfg[module].replace('.', '/') + '.py'

def make_cfg(args):
    cfg.merge_from_file(args.cfg_file) # Get task name

    cfg.merge_from_file(os.path.join('configs','default.yaml'))
    root1 = os.path.split(args.cfg_file)[0] 
    cfg.merge_from_file(os.path.join(root1, 'default.yaml'))

    # if 'ybyang' not in os.getcwd():
    #     cfg.merge_from_file(os.path.join('configs', cfg.task, 'server.yaml'))
        # os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join([str(gpu) for gpu in cfg.gpus])
    
    # cfg.merge_from_file(os.path.join('configs','urbangiraffe', 'default.yaml'))
    # cfg.merge_from_file(os.path.join('configs','urbangiraffe', 'default.yaml'))
    # cfg.merge_from_file(os.path.join('configs','bbox2d', 'default.yaml'))
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    parse_cfg(cfg, args)
    return cfg

parser = argparse.ArgumentParser(conflict_handler="resolve")
parser.add_argument("--cfg_file", default="configs/kitti-360/gan2d.yaml",type=str)
parser.add_argument('--test', action='store_true', dest='test', default=False)
parser.add_argument('--render', action='store_true', dest='render', default=False)
parser.add_argument("--use_cuda", action='store_true', default=False)
parser.add_argument("--gpus", default=[],type=list)
parser.add_argument("--distributed", action='store_true', default=False)
parser.add_argument("--is_debug", action='store_true', default=False)
parser.add_argument("--DP", action='store_true', default=False)
parser.add_argument("--type", type=str, default="")
parser.add_argument('--det', type=str, default='')
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--launcher', type=str, default='none', choices=['none', 'pytorch'])
parser.add_argument('--load_D', action='store_true', default=False)
parser.add_argument('--is_debug', action='store_true', default=False)

parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
args = parser.parse_args()
if len(args.type) > 0:
    cfg.task = "run"
cfg = make_cfg(args)


import re
import time
import datetime
import torch
import tqdm
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from lib.config import cfg
from lib.utils.data_utils import to_cuda
import os
from lib.utils.img_utils import make_gif, save_tensor_img
import torchvision
from .utils import *
import torch.distributed as dist
import shutil

from lib.networks.GAN.utils.stylegan2_official_ops import conv2d_gradfix

from .losses.perceptual import PerceptualLoss



def random_crop_img(img, new_size = -1):
    h_img, w_img=  img.shape[-2], img.shape[-1]
    start_num = abs(w_img - h_img)
    n = torch.randint(0, start_num,())
    if h_img <= w_img:
        img = img[:,:,:,n:n+h_img]
    else: 
        img = img[:,:,n:n+w_img,:]

    if new_size == -1:
        size = min(h_img, w_img)
        for i in range(12):
            if size < pow(2,i):
                break
        new_size = pow(2,i)
    img = F.interpolate(img, size=(new_size, new_size))
    return img
def padding_img(img, new_size = -1):
    h_img, w_img=  img.shape[-2], img.shape[-1]
    pad_num = int(abs(w_img - h_img)/2)
    if h_img <= w_img:
        pad = nn.ZeroPad2d((0,0,pad_num,pad_num))
    else: 
        pad = nn.ZeroPad2d((pad_num,pad_num,0,0))

    if new_size == -1:
        size = max(h_img, w_img)
        for i in range(12):
            if size < pow(2,i):
                break
        new_size = pow(2,i-1)

    img = pad(img)
    img = F.interpolate(img, size=(new_size, new_size))
    return img

def reduce_mean(tensor, nprocs):  
    # 用于平均所有gpu上的运行结果，比如loss
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


# class depth_rank_loss(nn.Module):
#     def __init__(self):  
#         super().__init__()
#     def forward(self, depth_gt, depth_pred):
#         # high res, only calc on fg
#         batch_size = depth_gt.shape[0]
#         depth_gt = depth_gt.reshape(batch_size, -1,1)
#         rank_loss_target = (depth_gt - depth_gt.permute(0,2,1)).sign().reshape(batch_size,-1)
#         output = depth_pred.reshape(batch_size, -1,1)
#         # output = output[self.fg_idx]
#         num = output.shape[1] # [n, 1]
#         # print(num)
#         output = output.reshape(1, -1)
#         o1 = output.expand(num, -1).reshape(batch_size,-1)
#         o2 = output.T.expand(-1, num).reshape(batch_size,-1)
#         return F.margin_ranking_loss(o1, o2, rank_loss_target)


class Trainer():
    def __init__(self, network):
        if cfg.distributed:
            self.device = torch.device('cuda:{}'.format(cfg.local_rank))
        elif cfg.use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.network = network.net.to(self.device)

        if cfg.distributed:
            network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(network)
            network = DistributedDataParallel(
                network,
                device_ids=[cfg.local_rank],
                output_device=cfg.local_rank,
                find_unused_parameters=True
           )

        self.use_aug = cfg.use_data_augment
        self.render_obj = cfg.render_obj
        self.render_sky = cfg.render_sky
        self.render_stuff = cfg.render_stuff
        self.use_patch_discriminator = cfg.use_patch_discriminator
        self.local_rank = cfg.local_rank
        #if cfg.distributed:
        # else:

        self.z_global = self.network.z_global
        self.generator = self.network.generator
        self.discriminator = self.network.discriminator
        self.generator_test =self.network.generator_test
        if hasattr(self.network, 'discriminator_obj'):
            self.discriminator_obj = self.network.discriminator_obj
        if hasattr(self.network, 'aug'):
            self.aug = self.network.aug
        if hasattr(self.network, 'aug_obj'):
            self.aug_obj = self.network.aug_obj


        self.loss_type = 'standard' #{'wgan', 'standard'}
        #self.rag_type = 'nothing'
        self.lambda_reg = 10.
        

        self.use_occupancy_mask = cfg.use_occupancy_mask
        self.use_depth = cfg.use_depth
        # self.return_alpha_map = cfg.network_kwargs.generator_kwargs.return_alpha_map
        
        self.use_semantic_aware = cfg.network_kwargs.use_semantic_aware
        self.use_multi_domain_D = cfg.network_kwargs.multi_domain_D
        
        self.train_img_dir = os.path.join(cfg.out_img_dir, 'train', cfg.exp_name, cfg.exp_info)
        self.test_img_dir = os.path.join(cfg.out_img_dir, 'test', cfg.exp_name, cfg.exp_info)
        self.render_img_dir = os.path.join(cfg.out_img_dir, 'render', cfg.exp_name, cfg.exp_info)
        self.exp_name = cfg.exp_name
        self.task = cfg.task
        if not os.path.exists(self.train_img_dir):
            os.makedirs(self.train_img_dir)
        if not os.path.exists(self.test_img_dir):
            os.makedirs(self.test_img_dir)

        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1) 

        self.criteria = torch.nn.ModuleDict()

        self.weights = dict()
        self.gen_losses = dict()
        self.dis_losses = dict()
        self._init_loss(cfg)
        for loss_name, loss_weight in self.weights.items():
            print("Loss {:<20} Weight {}".format(loss_name, loss_weight))
            if loss_name in self.criteria.keys() and \
                    self.criteria[loss_name] is not None:
                self.criteria[loss_name].to('cuda')


        self.models = {'discriminator' : self.discriminator, 'generator' : self.generator, 'inversion': self.z_global}
        self.val_use_occupancy = cfg.test.use_occupancy
        self.enable_amp = False
        self.amp_scaler = torch.cuda.amp.GradScaler(enabled=self.enable_amp)

    def reduce_loss_stats(self, loss_stats):
        reduced_losses = {k: torch.mean(v) for k, v in loss_stats.items()}
        return reduced_losses

    def clip_model_gradient(self, name, nan=0.0, min_val=-1e5, max_val=1e5):
        """Clips the gradient of a particular model.

        Args:
            name: The name of the model, i.e., in `self.models`.
            nan: The value to which `nan` is clipped. This should always be set
                as `0`. (default: 0)
            min_val: The minimum value to cutoff. (default: -1e5)
            max_val: The maximum value to cutoff. (default: 1e5)
        """
        assert nan == 0
        for param_name, param in self.models[name].named_parameters():
            if param.grad is None:
                # if self.model_has_unused_param[name]:
                    continue
                # raise ValueError(f'Parameter `{param_name}` from '
                #                  f'model `{name}` does not have gradient!')
            if min_val is None:
                min_val = torch.finfo(param.grad.dtype).min
            if max_val is None:
                max_val = torch.finfo(param.grad.dtype).max
            torch.clamp(param.grad.unsqueeze(0).nansum(0),
                        min=min_val, max=max_val, out=param.grad)

    def zero_grad_optimizer(self, name, set_to_none=None):
        """Wraps `optimizer.zero_grad()` with `set_to_none` option.

        When clear gradients, setting `set_to_none` as `True` is slightly
        efficient, however, it may cause the problem of `adding tensor with
        None` when some gradients are missing. By default, we use
        `has_unused_parameter` to determine whether the gradient should be set
        to zeros or None.
        """
        # if set_to_none is None:
        #     set_to_none = not self.model_has_unused_param[name]
        self.optimizers[name].zero_grad(set_to_none=set_to_none)

    def step_optimizer(self, name, clip_grad=True, **clip_kwargs):
        """Wraps stepping optimizer with gradient clip and AMP scalar."""
        # NOTE: AMP will use inf/NaN to adjust its behavior, hence the gradient
        # should not be clipped.
        if not self.enable_amp and clip_grad and name != 'inversion':
            self.clip_model_gradient(name, **clip_kwargs)
        self.amp_scaler.step(self.optimizers[name])

    def _init_loss(self, cfg):
        r"""Initialize loss terms.

        Args:
            cfg (obj): Global configuration.
        """
        if hasattr(cfg.train.loss_weight, 'gan'):
            # self.criteria['GAN'] = GANLoss()
            self.weights['gan_g'] = cfg.train.loss_weight.gan
            self.weights['gan_d_real'] = cfg.train.loss_weight.gan 
            self.weights['gan_d_fake'] = cfg.train.loss_weight.gan 
            # self.weights['gan_d_real'] = cfg.train.loss_weight.gan / 2
            # self.weights['gan_d_fake'] = cfg.train.loss_weight.gan / 2

        if hasattr(cfg.train.loss_weight, 'l2'):
            self.criteria['L2'] = nn.MSELoss()
            self.weights['L2'] = cfg.train.loss_weight.l2
        if hasattr(cfg.train.loss_weight, 'l1'):
            self.criteria['L1'] = nn.L1Loss()
            self.weights['L1'] = cfg.train.loss_weight.l1
        if hasattr(cfg.train.loss_weight, 'l1_depth'):
            self.criteria['L1_depth'] = nn.L1Loss()
            # self.criteria['L1_depth'] = depth_rank_loss()
            self.weights['L1_depth'] = cfg.train.loss_weight.l1_depth
        if hasattr(cfg.train.loss_weight, 'perceptual'):
            self.criteria['Perceptual'] = \
                PerceptualLoss(
                    network=cfg.train.perceptual_loss.mode,
                    layers=cfg.train.perceptual_loss.layers,
                    weights=cfg.train.perceptual_loss.weights)
            self.weights['Perceptual'] = cfg.train.loss_weight.perceptual

        self.weights['gan_d_reg'] = cfg.train.loss_weight.gan_reg

        if hasattr(cfg.train.loss_weight, 'gan_obj'):
            # self.criteria['GAN'] = GANLoss()
            self.weights['gan_obj_d_reg'] = cfg.train.loss_weight.gan_reg
            self.weights['gan_obj_g'] = cfg.train.loss_weight.gan_obj
            self.weights['gan_obj_d_real'] = cfg.train.loss_weight.gan_obj 
            self.weights['gan_obj_d_fake'] = cfg.train.loss_weight.gan_obj 

        if hasattr(cfg.train.loss_weight, 'diversity'):
            self.weights['diversity'] = cfg.train.loss_weight.diversity


    def to_cuda(self, batch):
        for k in batch:
            if isinstance(batch[k], tuple) or isinstance(batch[k], list):
                #batch[k] = [b.cuda() for b in batch[k]]
                batch[k] = [b.to(self.device) for b in batch[k]]
            elif isinstance(batch[k], dict):
                batch[k] = {key: self.to_cuda(batch[k][key]) for key in batch[k]}
            else:
                # batch[k] = batch[k].cuda()
                batch[k] = batch[k].to(self.device)
        return batch

    def add_iter_step(self, batch, iter_step):
        if isinstance(batch, tuple) or isinstance(batch, list):
            for batch_ in batch:
                self.add_iter_step(batch_, iter_step)

        if isinstance(batch, dict):
            batch['iter_step'] = iter_step

            
    def calc_G_adv_loss(self, data):
        x_fake = self.get_D_input(data, is_real = False)

        # if cfg.render_stuff:
        d_fake = self.network(x_fake,type = 'D')['score']
        # self.gen_losses['gan_g'] = compute_bce(d_fake, 1)
        self.gen_losses['gan_g'] = F.softplus(-d_fake).mean()

        if self.use_patch_discriminator and 'bbox_semantic' in data and data['bbox_semantic'].max() > -1:
            bbox_idx = data['bbox_semantic'] != -1
            patches = data['bbox_patches'][bbox_idx]
            bbox_poses = data['bbox_poses'][bbox_idx]
            bbox_semantic = data['bbox_semantic'][bbox_idx]
            bbox_semantic[bbox_semantic == 11] = 1
            bbox_semantic[bbox_semantic == 26] = 0
            bbox_domain_idx = torch.stack((torch.arange(0, bbox_semantic.shape[0]).to(bbox_semantic.device),bbox_semantic), dim=1).to(torch.int64)

            if hasattr(self, 'aug_obj'):
                patches = self.aug_obj(patches)
            d_fake_obj = self.network(patches,type = 'D_obj', c = bbox_poses)['score']

            self.gen_losses['gan_obj_g'] = compute_bce(d_fake_obj, 1)

            
    def get_D_input(self, data,  is_real = True):
        x = []
        if not is_real:
            rgb = data['rgb']
            if self.use_occupancy_mask:
                occupancy_mask = torch.where(data['occupancy_mask'] != 0, torch.ones_like(data['occupancy_mask']), torch.zeros_like(data['occupancy_mask']))
                rgb = data['rgb'] * occupancy_mask
            x_whole = [rgb]           
            x= torch.cat(x_whole, dim = 1)

        else: 
            rgb_real = data['rgb_gt']
            if self.use_occupancy_mask:
                occupancy_mask_gt = torch.where(data['occupancy_mask_gt'] != 0, torch.ones_like(data['occupancy_mask_gt']), torch.zeros_like(data['occupancy_mask_gt']))
                rgb_real = rgb_real * occupancy_mask_gt
            x_whole = [rgb_real]    
            x = torch.cat(x_whole, dim = 1)

        if hasattr(self, 'aug'):
            x = self.aug(x)
        if cfg.is_debug:
            save_tensor_img(torchvision.utils.make_grid(x[:,:3].detach().cpu()), 'tmp', 'rgb_Din.jpg')

        return x
    
    def calc_D_adv_loss(self, batch):
        
        x_real  = self.get_D_input(batch, is_real = True)
        loss_d_full = 0.

        # if cfg.render_stuff:
        x_real.requires_grad_()
        d_real = self.network(x_real, type = 'D')['score']
        #discriminator(x_real)
        # self.dis_losses['gan_d_real'] =  compute_bce(d_real, 1)
        self.dis_losses['gan_d_real'] = F.softplus(-d_real).mean()
        # For ada aug
        self.dis_losses['real_score'] = d_real.mean()
        self.dis_losses['real_sign'] = d_real.sign().mean()
        if True:
            self.dis_losses['gan_d_reg'] = self.r1_penalty(d_real, x_real).mean()
        else:
            self.dis_losses['gan_d_reg'] = compute_grad2(d_real, x_real).mean()

        # obj loss real
        if self.use_patch_discriminator and torch.max(batch['bbox_semantic_gt']) > -1:
            valid_idx = torch.argwhere(batch['bbox_semantic_gt'] != -1)
            patches_real = batch['bbox_patches_gt'][valid_idx[:,0],valid_idx[:,1]]
            # pose_patches_real = batch['bbox_pose_patches_gt'][valid_idx[:,0],valid_idx[:,1]]
            # patches_real = torch.cat((patches_real,pose_patches_real), dim=1)
            bbox_poses = batch['bbox_pose_gt'][valid_idx[:,0],valid_idx[:,1]]
            patches_real.requires_grad_()

            bbox_semantic = batch['bbox_semantic_gt'][valid_idx[:,0],valid_idx[:,1]]
            bbox_semantic[bbox_semantic == 11] = 1
            bbox_semantic[bbox_semantic == 26] = 0
            bbox_domain_idx = torch.stack((torch.arange(0, bbox_semantic.shape[0]).to(bbox_semantic.device),bbox_semantic), dim=1).to(torch.int64)

            if hasattr(self, 'aug_obj'):
                patches_real = self.aug_obj(patches_real)
            d_real_obj = self.network(patches_real,type = 'D_obj' , c = bbox_poses)['score'][bbox_domain_idx[:,0],bbox_domain_idx[:,1]]
            self.dis_losses['gan_obj_d_real'] = compute_bce(d_real_obj, 1)
            self.dis_losses['gan_obj_d_reg'] = compute_grad2(d_real_obj, patches_real).mean()

            self.dis_losses['real_obj_score'] = d_real_obj.mean()
            self.dis_losses['real_obj_sign'] = d_real_obj.sign().mean()


        with torch.no_grad():
            fake_output = self.network(batch, type = 'G' ,data_type  = 'dis')
        x_fake = self.get_D_input(fake_output, is_real = False)
        x_fake.requires_grad_()
        d_fake = self.network(x_fake, type = 'D')['score']
        self.dis_losses['gan_d_fake'] = F.softplus(d_fake).mean()
        # self.dis_losses['gan_d_fake'] = compute_bce(d_fake, 0)

        if self.use_patch_discriminator and 'bbox_semantic' in fake_output and (fake_output['bbox_semantic']).max() > -1:
            valid_idx = torch.argwhere(fake_output['bbox_semantic'] != -1)
            patches_fake = fake_output['bbox_patches'][valid_idx[:,0],valid_idx[:,1]]
            bbox_poses_fake = batch['bbox_pose_fake'][valid_idx[:,0],valid_idx[:,1]]
            patches_fake.requires_grad_()

            bbox_semantic = fake_output['bbox_semantic'][valid_idx[:,0],valid_idx[:,1]]
            bbox_semantic[bbox_semantic == 11] = 1
            bbox_semantic[bbox_semantic == 26] = 0
            bbox_domain_idx = torch.stack((torch.arange(0, bbox_semantic.shape[0]).to(bbox_semantic.device),bbox_semantic), dim=1).to(torch.int64)
            
            if hasattr(self, 'aug_obj'):
                patches_fake = self.aug_obj(patches_fake)
            d_fake_obj = self.network(patches_fake,type = 'D_obj', c = bbox_poses_fake)['score'][bbox_domain_idx[:,0],bbox_domain_idx[:,1]]
            self.dis_losses['gan_obj_d_fake'] = compute_bce(d_fake_obj, 0)
     
    def calc_reconstruction_loss(self, batch, fake_output):
        real_raw, pred_raw = batch['rgb'], fake_output['rgb']
        batch_size = real_raw.shape[0]

        if self.use_occupancy_mask:
            assert 'occupancy_mask' in batch
            occupancy_mask = batch['occupancy_mask'].to(torch.float32)
            
            occupancy_mask_rgb = occupancy_mask.detach().clone()
            occupancy_mask_rgb[occupancy_mask_rgb != 0] = 1
           
            occupancy_mask_depth = occupancy_mask.detach().clone()
            occupancy_mask_depth[occupancy_mask_depth == 1]= 0
            occupancy_mask_depth[occupancy_mask_depth != 0] = 1
        else:
            occupancy_mask_rgb = torch.ones_like(real_raw[:,0:1])
            occupancy_mask_depth = torch.ones_like(real_raw[:,0:1])


        if self.render_obj and 'bbox_mask' in fake_output:
            occupancy_mask -= fake_output['bbox_mask']
            occupancy_mask = torch.where(occupancy_mask > 0, torch.ones_like(occupancy_mask), torch.zeros_like(occupancy_mask))

        real = real_raw * occupancy_mask_rgb
        pred = pred_raw * occupancy_mask_rgb

        #     valid_pixel_mun = torch.sum(occupancy_mask.reshape(batch_size,-1), dim = -1)
        # else: 
        #     valid_pixel_mun = 94 * 352

        if 'L2' in self.criteria:
            self.gen_losses['L2'] = self.criteria['L2'](pred, real)
        if 'L1' in self.criteria:
            self.gen_losses['L1'] = self.criteria['L1'](pred, real)
        if 'Perceptual' in self.criteria:
            self.gen_losses['Perceptual'] = self.criteria['Perceptual'](
                pred, real)
                        

        if 'L1_depth' in self.criteria:
            assert self.use_depth
            real_depth, pred_depth = batch['depth'], fake_output['depth']
            real_depth = real_depth * occupancy_mask_depth
            pred_depth = pred_depth * occupancy_mask_depth

            pred_depth_patch = pred_depth[occupancy_mask_depth == 1]
            real_depth_patch = real_depth[occupancy_mask_depth == 1]
            self.gen_losses['L1_depth'] = self.criteria['L1_depth'](pred_depth_patch, real_depth_patch)
            
        if cfg.is_debug:
            # save_tensor_img(real_raw, save_dir= 'tmp', name='rec_real_raw.jpg')
            save_tensor_img(pred, save_dir= 'tmp', name='rec_pred.jpg')
            save_tensor_img(real, save_dir= 'tmp', name='rec_real.jpg')
            if 'L1_depth' in self.criteria:
                save_tensor_img(pred_depth, save_dir= 'tmp', name='rec_pred_depth.jpg', type = 'depth')
                save_tensor_img(real_depth, save_dir= 'tmp', name='rec_real_depth.jpg', type = 'depth')
                save_tensor_img(abs(real_depth - pred_depth), save_dir= 'tmp', name='rec_error_depth.jpg', type = 'depth')
               

    def r1_penalty(self, d_real, x_real):
        with conv2d_gradfix.no_weight_gradients():
            r1_grads = torch.autograd.grad(outputs=d_real.sum(), inputs=x_real, create_graph=True, only_inputs=True)
        r1_grads_image = r1_grads[0]
        r1_penalty = r1_grads_image.square().sum([1,2,3]) / 2
        return r1_penalty



    def train_step_generator(self, batch, optimizers):
        if not cfg.inversion:
            self.models['discriminator'].requires_grad_(False)
            self.models['generator'].requires_grad_(True)
        # toggle_grad(generator, True)
        # toggle_grad(discriminator, False)       
        if self.use_patch_discriminator:
            discriminator_obj = self.discriminator_obj
            toggle_grad(discriminator_obj, False)

        # optimizer_g.zero_grad()
        fake_output = self.network(batch, type='G', data_type='gen', mode='train')

        if 'gan_g' in self.weights:
            self.calc_G_adv_loss(fake_output)
        if cfg.render_stuff:
            self.calc_reconstruction_loss(batch, fake_output)
        if 'diversity' in self.weights:
            self.calc_diversity_loss(batch, fake_output)
            
        g_loss = 0
        for key in self.gen_losses:
            if key in self.weights:
                g_loss = g_loss + self.gen_losses[key] * self.weights[key]

        self.gen_losses['g_total'] = g_loss
        g_loss.backward()

        if 'generator' in optimizers.keys():
            optimizer_g = optimizers['generator']
            if cfg.distributed:
                torch.distributed.barrier()
                for k in self.gen_losses:
                    self.gen_losses[k] = reduce_mean(self.gen_losses[k], dist.get_world_size())
                for i, p in enumerate(optimizer_g.param_groups):
                    if p['params'][0].grad != None:
                        optimizer_g.param_groups[i]['params'][0].grad = reduce_mean(p['params'][0].grad, dist.get_world_size())
            
            self.zero_grad_optimizer('generator')
            self.step_optimizer('generator')
        
        elif 'inversion' in optimizers.keys():
            optimizer = optimizers['inversion']
            if cfg.distributed:
                torch.distributed.barrier()
                for k in self.dis_losses:
                    self.dis_losses[k] = reduce_mean(self.dis_losses[k], dist.get_world_size())
                for i, p in enumerate(optimizer.param_groups):
                    # if p['params'][0].grad != None:
                    optimizer.param_groups[i]['params'][0].grad = reduce_mean(p['params'][0].grad, dist.get_world_size())
            
            self.zero_grad_optimizer('inversion')
            self.step_optimizer('inversion')

        if self.generator_test is not None:
            update_average(self.generator_test, self.generator, beta=0.999)

        fake_image_dict = {'rgb':fake_output['rgb']}
        if self.use_depth:
            fake_image_dict['depth'] = fake_output['depth']
            fake_image_dict['depth_gt'] = batch['depth']
        if self.use_patch_discriminator and 'bbox_semantic' in fake_output:
            fake_image_dict['obj_patch'] = fake_output['bbox_patches'][fake_output['bbox_semantic'] == 26]
        return fake_image_dict 
    

    def train_setp_discriminator(self, batch, optimizers):
        if not cfg.inversion:
            self.models['discriminator'].requires_grad_(True)
            self.models['generator'].requires_grad_(False)

        if self.use_patch_discriminator:
            self.discriminator_obj.train()
            discriminator_obj = self.discriminator_obj
            toggle_grad(discriminator_obj, True)

        self.calc_D_adv_loss(batch)
        d_loss = 0
        for k in self.dis_losses:
            if not (k in ['d_total', 'real_score', 'real_sign','real_obj_score', 'real_obj_sign']):
                d_loss += self.dis_losses[k] * self.weights[k]

        self.dis_losses['d_total'] = d_loss
        d_loss.backward()
        
        if 'discriminator' in optimizers.keys():
            optimizer_d = optimizers['discriminator']
            if cfg.distributed:
                torch.distributed.barrier()
                for k in self.dis_losses:
                    self.dis_losses[k] = reduce_mean(self.dis_losses[k], dist.get_world_size())
                for i, p in enumerate(optimizer_d.param_groups):
                    # if p['params'][0].grad != None:
                    optimizer_d.param_groups[i]['params'][0].grad = reduce_mean(p['params'][0].grad, dist.get_world_size())
            self.zero_grad_optimizer('discriminator')
            self.step_optimizer('discriminator')
        
        elif 'inversion' in optimizers.keys():
            optimizer = optimizers['inversion']
            if cfg.distributed:
                torch.distributed.barrier()
                for k in self.dis_losses:
                    self.dis_losses[k] = reduce_mean(self.dis_losses[k], dist.get_world_size())
                for i, p in enumerate(optimizer.param_groups):
                    # if p['params'][0].grad != None:
                    optimizer.param_groups[i]['params'][0].grad = reduce_mean(p['params'][0].grad, dist.get_world_size())
            self.zero_grad_optimizer('inversion')
            self.step_optimizer('inversion')


    def train(self, epoch, data_loader, optimizer, recorder):
        if not cfg.inversion:
            optimizer_g, optimizer_d = optimizer['op'], optimizer['op_d']
            self.optimizers = {'discriminator' : optimizer_d, 'generator' : optimizer_g}
        else: 
            self.optimizers = {'inversion': optimizer['op']}
            
        max_iter = len(data_loader)
        self.network.train()
        end = time.time()
        for iteration, batch in enumerate(data_loader):
            torch.cuda.reset_peak_memory_stats()
            data_time = time.time() - end
            iteration = iteration + 1

            batch = to_cuda(batch)#to_cuda(batch, self.device)
            self.add_iter_step(batch, epoch * max_iter + iteration)
            loss_stats, image_stats = {}, {}
            batch['z_global'] = self.z_global
            fake_image_dict = self.train_step_generator(batch, optimizers=self.optimizers)
            if 'gan_d_real' in self.weights:
                self.train_setp_discriminator(batch, optimizers=self.optimizers)

            loss_stats.update(self.gen_losses)
            loss_stats.update(self.dis_losses)

            image_stats = {k :torchvision.utils.make_grid(fake_image_dict[k]) for k in  fake_image_dict}

            # data recording stage: loss_stats, time, image_stats
            recorder.step += 1

            loss_stats = self.reduce_loss_stats(loss_stats)
            recorder.update_loss_stats(loss_stats)
            batch_time = time.time() - end
            end = time.time()
            recorder.batch_time.update(batch_time)
            recorder.data_time.update(data_time)
            

            if cfg.local_rank == 0:
                # Execute ADA heuristic.
                if hasattr(self, 'aug') and iteration % cfg.ada_interval == 0:
                    adjust = torch.sign(self.dis_losses['real_sign'].detach().cpu() - cfg.ada_target) * (cfg.train.batch_size * cfg.ada_interval * cfg.world_size) / (cfg.ada_kimg * 1000)
                    # IF donot detach memory cost will always increase 
                    self.aug.p = torch.clamp(self.aug.p + adjust, 0, 1)
                    recorder.update_loss_stats(
                        {'aug_p' : self.aug.p})
                if hasattr(self, 'aug_obj') and hasattr(self.dis_losses, 'real_obj_sign')  and iteration % cfg.ada_interval == 0:
                    adjust_obj = torch.sign(self.dis_losses['real_obj_sign'].detach().cpu() - cfg.ada_target) * (cfg.train.batch_size * cfg.ada_interval * cfg.world_size) / (cfg.ada_kimg * 1000)
                    # IF donot detach memory cost will always increase 
                    self.aug_obj.p = torch.clamp(self.aug_obj.p + adjust_obj, 0, 1)
                    recorder.update_loss_stats(
                        {'aug_obj_p' : self.aug_obj.p})

                if iteration % cfg.img_log_interval == 0:
                    with torch.no_grad():
                        fake_output_test = self.network(batch, type = 'G_test' ,data_type  = 'gen', mode = 'testing')
                        rgb_test_tmp = torchvision.utils.make_grid(fake_output_test['rgb'])

                    rgb_gt = torchvision.utils.make_grid(batch['rgb'])
                    save_tensor_img(image_stats['rgb'], save_dir= self.train_img_dir, name='%d_rgb.jpg'%recorder.step)
                    if cfg.use_depth:
                        save_tensor_img(image_stats['depth'], save_dir= self.train_img_dir, name='%d_depth.jpg'%recorder.step,type =  'depth')
                        save_tensor_img(image_stats['depth_gt'], save_dir= self.train_img_dir, name='%d_depth_gt.jpg'%recorder.step,type =  'depth')
                    
                    if cfg.render_stuff:
                        save_tensor_img(rgb_gt , save_dir= self.train_img_dir, name='%d_rgb_gt.jpg'%recorder.step)
                        save_tensor_img(rgb_test_tmp, save_dir= self.train_img_dir, name='%d_rgb_test.jpg'%recorder.step)
                    if cfg.render_obj and self.use_patch_discriminator:
                        save_tensor_img(image_stats['obj_patch'], save_dir= self.train_img_dir, name='%d_patch.jpg'%recorder.step)
                        patch_test = fake_output_test['bbox_patches'][fake_output_test['bbox_semantic'] == 26]
                        save_tensor_img(patch_test, save_dir= self.train_img_dir, name='%d_patch_test.jpg'%recorder.step)

                if (iteration % cfg.log_interval == 0 or iteration == (max_iter - 1)):
                # print training state
                    eta_seconds = recorder.batch_time.global_avg * (max_iter - iteration)
                    eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                    lr = optimizer['op'].param_groups[0]['lr']
                    if not cfg.inversion:
                        lr_d = optimizer['op_d'].param_groups[0]['lr']
                    memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
                    
                    if not cfg.inversion:
                        training_state = '  '.join(['eta: {}', '{}', 'lr_g: {:.6f}', 'lr_d: {:.6f}', 'max_mem: {:.0f}'])
                        training_state = training_state.format(eta_string, str(recorder), lr, lr_d, memory)
                    else:
                        training_state = '  '.join(['eta: {}', '{}', 'lr: {:.6f}', 'max_mem: {:.0f}'])
                        training_state = training_state.format(eta_string, str(recorder), lr, memory)
                    print(training_state)
                    # record loss_stats and image_dict
                    recorder.record('train', save_img = True if iteration % cfg.img_log_interval == 0 else False)
            
            if hasattr(self, 'gen_losses'):
                self.gen_losses = {}
            if hasattr(self, 'dis_losses'):
                self.dis_losses = {}
            torch.cuda.empty_cache()


    def val(self, epoch, data_loader, evaluator=None, recorder=None):
        val_tmp_dir = os.path.join(cfg.out_img_dir, 'val_tmp')
                
        if not os.path.exists(val_tmp_dir):
            os.makedirs(val_tmp_dir)
        gt_dir = os.path.join(val_tmp_dir, 'gt')
        pred_dir = os.path.join(val_tmp_dir, 'pred')
        pred_gt_dir = os.path.join(val_tmp_dir, 'pred_gt')
        if self.render_obj:
            gt_obj_dir = os.path.join(val_tmp_dir, 'gt_obj')
            pred_obj_dir = os.path.join(val_tmp_dir, 'pred_obj')


        shutil.rmtree(val_tmp_dir)
        # for f in os.listdir(val_tmp_dir):
        #     os.remove(os.path.join(val_tmp_dir, f))
        self.network.eval()
        torch.cuda.empty_cache()
        val_loss_stats = {}
        data_size = len(data_loader)
        image_stats = {}
        # print('%d'%data_size)
        for it, batch in enumerate(tqdm.tqdm(data_loader)):
            batch = to_cuda(batch, self.device)
            batch['z_global'] = self.z_global
            with torch.no_grad():
                output = self.network(batch, type='G', data_type='gen', mode='test')
                #if evaluator is not None:
                if self.val_use_occupancy:
                    occupancy_mask = torch.where(batch['occupancy_mask'] != 0, torch.ones_like(batch['occupancy_mask']), torch.zeros_like(batch['occupancy_mask']))
                    occupancy_mask_real = torch.where(batch['occupancy_mask_gt'] != 0, torch.ones_like(batch['occupancy_mask_gt']), torch.zeros_like(batch['occupancy_mask_gt']))
                    real_image = batch['rgb_gt'] * occupancy_mask_real
                    fake_image = output['rgb'] * occupancy_mask
                    fake_image_gt = batch['rgb'] * occupancy_mask
                else:
                    real_image = batch['rgb_gt']
                    fake_image = output['rgb']
                    fake_image_gt = batch['rgb']
                real_id, fake_id = batch['frame_id_gt'], batch['frame_id']

                for i in range(real_id.shape[0]):
                    save_tensor_img(real_image[i], save_dir = gt_dir, name='%d.jpg'%real_id[i])
                    save_tensor_img(fake_image[i], save_dir = pred_dir, name='%d.jpg'%fake_id[i])
                    save_tensor_img(fake_image_gt[i], save_dir = pred_gt_dir, name='%d.jpg'%fake_id[i])

                if self.render_obj and self.use_patch_discriminator:
                    if 'bbox_patches' in output:
                        fake_patches = output['bbox_patches'][output['bbox_semantic'] ==26]
                        for i in range(fake_patches.shape[0]):
                            save_tensor_img(fake_patches[i], save_dir = pred_obj_dir, name='%d_%d.jpg'%(it, i))

                    if torch.sum(batch['bbox_semantic_gt'] == 26) >0:
                        real_patches = batch['bbox_patches_gt'][batch['bbox_semantic_gt'] == 26]
                        for i in range(real_patches.shape[0]):
                            save_tensor_img(real_patches[i], save_dir = gt_obj_dir, name='%d_%d.jpg'%(it, i))
  
        if evaluator is not None:
            # if self.render_stuff:
            eval_result = evaluator.evaluate_fid(gt_dir = gt_dir, pred_dir = pred_dir, tag = 'scene')
            val_loss_stats.update(eval_result)
            print(eval_result)

            if self.render_obj and self.use_patch_discriminator:
                eval_result_obj = evaluator.evaluate_fid(gt_dir = gt_obj_dir, pred_dir = pred_obj_dir, tag = 'obj')
                val_loss_stats.update(eval_result_obj)
                print(eval_result_obj)

        if recorder:
            recorder.record('val', epoch, val_loss_stats, image_stats)


    def render(self, epoch, data_loader, renderer ,evaluator=None, recorder=None):
        render_out_dir = os.path.join(cfg.out_img_dir, 'render',self.task, self.exp_name)
        if not os.path.exists(render_out_dir):
            os.makedirs(render_out_dir)
        if False:
            shutil.rmtree(render_out_dir)
        self.network.eval()
        # step_num = 64

        for batch in tqdm.tqdm(data_loader):
            batch = to_cuda(batch, self.device)
            renderer.render_tasks(self.network, batch, out_dir=render_out_dir)

import nntplib
import numpy as np
from lib.config import cfg, args
from lib.networks.make_network import make_network
from lib.train import make_trainer, make_optimizer, make_lr_scheduler, make_recorder, set_lr_scheduler
from lib.datasets import make_data_loader
from lib.utils.net_utils import load_model, save_model, load_network, save_trained_config, load_pretrain
from lib.evaluators import make_evaluator
from lib.renderers import make_renderer
import torch.multiprocessing
import random
import torch
import torch.distributed as dist
import os
torch.autograd.set_detect_anomaly(True)


torch.manual_seed(cfg.random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(cfg.random_seed)
np.random.seed(cfg.random_seed)
torch.manual_seed(cfg.random_seed)
torch.cuda.manual_seed_all(cfg.random_seed)


def train(cfg, network):
    trainer = make_trainer(cfg, network)
    optimizer = make_optimizer(cfg, network)
    scheduler = make_lr_scheduler(cfg, optimizer)
    recorder = make_recorder(cfg)
    evaluator = make_evaluator(cfg)
    load_network(network, cfg.trained_model_dir_init, load_D = True, load_p = False)
    begin_epoch = load_model(network,
                             optimizer,
                             scheduler,
                             recorder,
                             cfg.trained_model_dir,
                             resume=cfg.resume)      
    
    if begin_epoch == 0 and cfg.pretrain != '':
        load_pretrain(network, cfg.pretrain)

    for k in scheduler:
        set_lr_scheduler(cfg, scheduler[k])

    train_loader = make_data_loader(cfg,
                                    is_train=True,
                                    is_val=False,
                                    is_distributed=cfg.distributed,
                                    max_iter=cfg.ep_iter)
    val_loader = make_data_loader(cfg,
                                    is_train=False,
                                    is_val=True,
                                    # is_distributed=cfg.distributed,
                                    max_iter= cfg.test.frame_num // cfg.test.batch_size)
    
    # network = torch.nn.DataParallel(network, device_ids= [1,2,3,4])
    for epoch in range(begin_epoch, cfg.train.epoch):
        recorder.epoch = epoch
        # Train
        if cfg.distributed:
            train_loader.batch_sampler.sampler.set_epoch(epoch)
        trainer.train(epoch, train_loader, optimizer, recorder)
        for k in scheduler:
            scheduler[k].step()

        if (epoch + 1) % cfg.eval_ep == 0 and cfg.local_rank == 0:
            print(cfg.local_rank)
            trainer.val(epoch, val_loader, evaluator, recorder)

        if (epoch + 1) % cfg.save_latest_ep == 0:
            # save_trained_config(cfg)
            save_model(network, optimizer, scheduler, recorder, cfg.trained_model_dir, epoch, last=True)
        if (epoch + 1) % cfg.save_ep == 0:
            save_trained_config(cfg)
            save_model(network, optimizer, scheduler, recorder, cfg.trained_model_dir, epoch)
    
    return network


def test(cfg, network):
    trainer = make_trainer(cfg, network)
    val_loader = make_data_loader(cfg, is_train=False, is_val=True,max_iter= cfg.test.frame_num // cfg.test.batch_size)
    evaluator = make_evaluator(cfg)
    if cfg.test.ckpt_dir != '':
        cfg.trained_model_dir = cfg.test.ckpt_dir
    load_network(network, cfg.trained_model_dir_init, load_D = False)
    epoch = load_network(network,
                         cfg.trained_model_dir,
                         resume=cfg.resume,
                         epoch=cfg.test.epoch)
    trainer.val(epoch, val_loader, evaluator)


def render(cfg, network):
    trainer = make_trainer(cfg, network)
    render_loader = make_data_loader(cfg, 
    is_train=False, 
    is_val = False)
    evaluator = make_evaluator(cfg)
    renderer = make_renderer(cfg)
    if cfg.render.ckpt_dir != '':
        cfg.trained_model_dir = cfg.render.ckpt_dir
    # load_network(network, cfg.trained_model_dir_init, load_D = True)
    epoch = load_network(network,
                         cfg.trained_model_dir,
                         resume=cfg.resume,
                         epoch=cfg.test.epoch,
                         strict=False)


    trainer.render(epoch, render_loader, renderer)


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()

def main():
    # torch.set_deterministic(True)
    if cfg.distributed:
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = str(len(cfg.gpus))
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12345'
        cfg.local_rank = int(os.environ['RANK']) % torch.cuda.device_count()
        torch.cuda.set_device(cfg.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        print(f"[init] == local rank: {cfg.local_rank}, global rank: {os.environ['RANK']} ==")
        synchronize()
        cfg.world_size = dist.get_world_size()
    else:
        cfg.world_size = 1

    network = make_network(cfg)
    if args.test:
        test(cfg, network)
    elif args.render:
        render(cfg, network)
    else:
        train(cfg, network)

if __name__ == "__main__":
    main()

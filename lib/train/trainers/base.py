
from torch.nn.parallel import DistributedDataParallel
from lib.config import cfg
import torch

class BaseTrainer(object):
    def __init__(self):
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

# class ReconTrainer(BaseTrainer):
#     ''' Base trainer class.
#     '''

#     def evaluate(self, *args, **kwargs):
#         ''' Performs an evaluation.
#         '''
#         eval_list = defaultdict(list)

#         # for data in tqdm(val_loader):
#         eval_step_dict = self.eval_step()

#         for k, v in eval_step_dict.items():
#             eval_list[k].append(v)
#         # eval_dict = {k: v for k, v in eval_list.items()}
#         eval_dict = {k: np.mean(v) for k, v in eval_list.items()}
#         return eval_dict

#     def train_step(self, *args, **kwargs):
#         ''' Performs a training step.
#         '''
#         raise NotImplementedError

#     def eval_step(self, *args, **kwargs):
#         ''' Performs an evaluation step.
#         '''
#         raise NotImplementedError

#     def visualize(self, *args, **kwargs):
#         ''' Performs  visualization.
#         '''
#         raise NotImplementedError

class GANTrainer(BaseTrainer):
    pass

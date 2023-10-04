from lib.config import cfg
import torch
import torch.nn as nn
from collections import OrderedDict

class GANGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        if cfg.use_cuda: 
            if cfg.distributed:
                self.device = torch.device('cuda:{}'.format(cfg.local_rank))
            else:
                self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.render_option = OrderedDict()

    def sample_z(self, size, to_device=True, tmp=1.):
        z = torch.randn(*size) * tmp
        if to_device:
            z = z.to(self.device)
        return z
    
from collections import deque, defaultdict
from numpy import False_
import torch
from tensorboardX import SummaryWriter
import os
from lib.config.config import cfg
from termcolor import colored
from datetime import datetime

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def update(self, value):
        self.deque.append(value)
        self.count += 1
        self.total += value

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque))
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count


class Recorder(object):
    def __init__(self, cfg):

        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        log_dir = os.path.join(cfg.record_dir, current_time + '-' +cfg.exp_info)

        if not cfg.resume:
            print(colored('remove contents of directory %s' % log_dir, 'red'))
            os.system('rm -r %s/*' % log_dir)
        if cfg.local_rank == 0:
            self.writer = SummaryWriter(log_dir=log_dir, comment = cfg.exp_info)

        # scalars
        self.epoch = 0
        self.step = 0
        self.loss_stats = defaultdict(SmoothedValue)
        self.other_stats = defaultdict(SmoothedValue)
        self.batch_time = SmoothedValue()
        self.data_time = SmoothedValue()

        # images
        self.image_stats = defaultdict(object)
        if 'process_' + cfg.task in globals():
            self.processor = globals()['process_' + cfg.task]
        else:
            self.processor = None

    def update_loss_stats(self, loss_dict):
        for k, v in loss_dict.items():
            self.loss_stats[k].update(v.detach().cpu())

    def update_image_stats(self, image_stats):
        for k, v in image_stats.items():
            self.image_stats[k] = v.detach().cpu()

    def update_other_stats(self, other_dict):
        for k, v in other_dict.items():
            self.other_stats[k].update(v.detach().cpu())

    def record(self, prefix, step=-1, loss_stats=None, other_stats=None, image_stats=None, save_img = False):
        pattern = prefix + '/{}'
        step = step if step >= 0 else self.step
        loss_stats = loss_stats if loss_stats else self.loss_stats
        other_stats = other_stats if other_stats else self.other_stats

        for k, v in loss_stats.items():
            if isinstance(v, SmoothedValue):
                self.writer.add_scalar(pattern.format(k), v.median, step)
            else:
                self.writer.add_scalar(pattern.format(k), v, step)

        for k, v in other_stats.items():
            if isinstance(v, SmoothedValue):
                self.writer.add_scalar(pattern.format(k), v.median, step)
            else:
                self.writer.add_scalar(pattern.format(k), v, step)
        # if self.processor is None:
        #     return
        if save_img:
            image_stats = self.processor(image_stats) if image_stats else self.image_stats
            for k, v in image_stats.items():
                self.writer.add_image(pattern.format(k), v, step)

    def state_dict(self):
        scalar_dict = {}
        scalar_dict['step'] = self.step
        return scalar_dict

    def load_state_dict(self, scalar_dict):
        self.step = scalar_dict['step']

    def __str__(self):
        loss_state = []
        for k, v in self.loss_stats.items():
            loss_state.append('{}: {:.4f}'.format(k, v.deque[-1]))
        loss_state = '  '.join(loss_state)

        recording_state = '  '.join(['epoch: {}', 'step: {}', '{}', 'data: {:.4f}', 'batch: {:.4f}'])
        return recording_state.format(self.epoch, self.step, loss_state, self.data_time.avg, self.batch_time.avg)


def make_recorder(cfg):
    return Recorder(cfg)
'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import errno
import os
import sys
import time
import math

import importlib
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable

__all__ = ['get_mean_and_std', 'init_params', 'mkdir_p', 'AverageMeter']


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = trainloader = torch.utils.total_data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)

def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self) -> object:
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AverageMetricMeter(AverageMeter):
    def __init__(self, metric_name):
        self.metric_name = metric_name
        index = self.metric_name.rfind('.')
        package, name = self.metric_name[:index],self.metric_name[index+1:]
        self.metric = getattr(importlib.import_module(package), name)()


    def update(self, val, n=1):
        if isinstance(val, dict):
            super(AverageMetricMeter, self).update(self.metric(**val), n)
        else:
            super(AverageMetricMeter, self).update(val, n)

    def get_metric_name(self):
        return self.metric_name


def load_pretrained_weights(ckpt_path, model, model_key = None, step_key = None):
    import torch, os
    from utils.logger import Logger
    Logger.info('==> Resuming from checkpoint %s' % ckpt_path)
    assert os.path.isfile(ckpt_path), 'Error: no checkpoint directory found!'

    ckpt = torch.load(ckpt_path)
    try:
        # 在这里读取的时候由于可能你修改了通道，所以会导致无法导入权重的情况，出现错误，调用上面方法
        model.load_state_dict(ckpt[model_key if model_key is not None else 'model'])
    except:
        model_dict = model.state_dict()
        # 从与训练的网络中读取相应的权重，如果在现在的网络中拥有，就保留，否则忽略
        prediction_dict = {}
        for k, v in ckpt['model' if model_key is None else model_key].items():
            if k in model_dict and v.shape == model_dict[k].shape:
                prediction_dict[k] = v
            else:
                Logger.info('Layer %s not load, Pretrain %s: Define %s'%(k, v.shape, model_dict[k].shape if k in model_dict.keys() else '%s is not exist!'%k))
        model_dict.update(prediction_dict)
        model.load_state_dict(model_dict)

    step_key = step_key if step_key is not None else 'step'
    step = ckpt[step_key] if step_key in ckpt.keys() else ' No step'
    Logger.info("Load pretrain file from %s, %s:%s"%(ckpt_path, step_key, step))
    return model

if __name__ == '__main__':
    metric_name = 'adsd.yyy.aaaadfdf'
    index = metric_name.rfind('.')
    package, name = metric_name[:index],metric_name[index+1:]
    print(package, name)

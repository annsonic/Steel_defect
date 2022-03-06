import math
import numpy as np
from torch import nn
from torch.optim.adam import Adam
from torch.optim.sgd import SGD


def split_params(model: nn.Module):
    param_other, param_weight_decay, param_bias = list(), list(), list()  # optimizer parameter groups
    for k, v in model.named_parameters():
        if v.requires_grad:
            if '.bias' in k:
                param_bias.append(v)  # biases
            elif '.weight' in k and '.bn' not in k:
                param_weight_decay.append(v)  # apply weight decay
            else:
                param_other.append(v)  # all else
    return param_weight_decay, param_bias, param_other


def split_optimizer(model: nn.Module, cfg: dict):
    param_weight_decay, param_bias, param_other = split_params(model)
    if cfg['optimizer'] == 'Adam':
        optimizer = Adam(param_other, lr=cfg['lr'])
    elif cfg['optimizer'] == 'SGD':
        optimizer = SGD(param_other, lr=cfg['lr'], momentum=cfg['momentum'])
    else:
        raise NotImplementedError("optimizer {:s} is not support!".format(cfg['optimizer']))
    optimizer.add_param_group(
        {'params': param_weight_decay, 'weight_decay': cfg['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': param_bias})
    return optimizer


class IterWarmUpCosineDecayMultiStepLRAdjust(object):
    def __init__(self, init_lr=0.01,
                 epochs=300,
                 milestones=None,
                 warm_up_epoch=1,
                 iter_per_epoch=1000,
                 gamma=1.0,
                 alpha=0.1,
                 bias_idx=None):
        self.init_lr = init_lr
        self.epochs = epochs
        if milestones is None:
            milestones = []
        milestones.sort()
        assert warm_up_epoch >= 0
        if len(milestones) > 0:
            assert warm_up_epoch < milestones[0] and milestones[-1] <= epochs
        self.milestones = milestones
        self.warm_up_epoch = warm_up_epoch
        self.iter_per_epoch = iter_per_epoch
        self.gamma = gamma
        self.alpha = alpha
        last_epoch = epochs + 1 if len(milestones) > 0 and milestones[-1] == epochs else epochs
        self.flag = np.array([warm_up_epoch] + self.milestones + [last_epoch]).astype(np.int)
        self.flag = np.unique(self.flag)
        self.warm_up_iter = self.warm_up_epoch * iter_per_epoch
        self.bias_idx = bias_idx

    def cosine(self, current, total):
        return ((1 + math.cos(current * math.pi / total)) / 2) ** self.gamma * (1 - self.alpha) + self.alpha

    def get_lr(self, ite, epoch):
        current_iter = self.iter_per_epoch * epoch + ite
        if epoch < self.warm_up_epoch:
            up_lr = np.interp(current_iter, [0, self.warm_up_iter], [0, self.init_lr])
            down_lr = np.interp(current_iter, [0, self.warm_up_iter], [0.1, self.init_lr])
            return up_lr, down_lr
        num_pow = (self.flag <= epoch).sum() - 1
        multi_step_weights = self.alpha ** num_pow
        if num_pow == len(self.flag) - 2:
            lr = multi_step_weights * self.init_lr
            return lr, lr
        cosine_ite = (epoch - self.flag[num_pow]) * self.iter_per_epoch + ite
        cosine_all_ite = (self.flag[num_pow + 1] - self.flag[num_pow]) * self.iter_per_epoch
        cosine_weights = self.cosine(cosine_ite, cosine_all_ite)
        lr = multi_step_weights * cosine_weights * self.init_lr
        return lr, lr

    def __call__(self, optimizer, ite, epoch):
        ulr, dlr = self.get_lr(ite, epoch)
        for i, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = dlr if self.bias_idx is not None and i == self.bias_idx else ulr
        return ulr, dlr


class EpochWarmUpCosineDecayLRAdjust(object):
    def __init__(self, init_lr=0.01,
                 epochs=300,
                 warm_up_epoch=1,
                 iter_per_epoch=1000,
                 gamma=1.0,
                 alpha=0.1,
                 bias_idx=None):
        assert warm_up_epoch < epochs and epochs - warm_up_epoch >= 1
        self.init_lr = init_lr
        self.warm_up_epoch = warm_up_epoch
        self.iter_per_epoch = iter_per_epoch
        self.gamma = gamma
        self.alpha = alpha
        self.bias_idx = bias_idx
        self.flag = np.array([warm_up_epoch, epochs]).astype(np.int)
        self.flag = np.unique(self.flag)
        self.warm_up_iter = self.warm_up_epoch * iter_per_epoch

    def cosine(self, current, total):
        return ((1 + math.cos(current * math.pi / total)) / 2) ** self.gamma * (1 - self.alpha) + self.alpha

    def get_lr(self, ite, epoch):
        current_iter = self.iter_per_epoch * epoch + ite
        if epoch < self.warm_up_epoch:
            up_lr = np.interp(current_iter, [0, self.warm_up_iter], [0, self.init_lr])
            down_lr = np.interp(current_iter, [0, self.warm_up_iter], [0.1, self.init_lr])
            return up_lr, down_lr
        num_pow = (self.flag <= epoch).sum() - 1
        cosine_ite = (epoch - self.flag[num_pow] + 1)
        cosine_all_ite = (self.flag[num_pow + 1] - self.flag[num_pow])
        cosine_weights = self.cosine(cosine_ite, cosine_all_ite)
        lr = cosine_weights * self.init_lr
        return lr, lr

    def __call__(self, optimizer, ite, epoch):
        ulr, dlr = self.get_lr(ite, epoch)
        for i, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = dlr if self.bias_idx is not None and i == self.bias_idx else ulr
        return ulr, dlr

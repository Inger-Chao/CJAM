# -*- coding: utf-8 -*-
# @Time    : 2020-10-19 18:21
# @Author  : Inger

from torch.utils.tensorboard import SummaryWriter

# default `log_dir`is "runs" - we'll be more specific here
writer = SummaryWriter('runs/fashion_mnist_experiment_1')


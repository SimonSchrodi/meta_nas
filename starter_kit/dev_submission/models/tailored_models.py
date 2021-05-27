import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional
from copy import deepcopy

from .resnet import *
from .densenet import *

__all__ = ['resnet18dstacktailored', 'resnet18dstackdown', 'resnet18dstackdowntailored', 'densenet121stackdown', 'resnet18dstacktailoredD']


class StackTailored(nn.Module):
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model_tailored = base_model
        self.fc_combine = nn.Linear(10, 10)

    def forward(self, x):
        x_copy = deepcopy(x)
        xout = None
        for i in range(3):
            x = x_copy[:, i, ...][:, None, ...]
            x = self.base_model_tailored(x)
            if xout is None:
                xout = x
            else:
                xout = torch.cat((xout, x), 1)

        x = self.fc_combine(xout)
        return x


class FCOut(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_combine = nn.Linear(10, 48)
        self.fc_1 = nn.Linear(48, 32)
        self.fc_2 = nn.Linear(32, 16)
        self.fc_out = nn.Linear(16, 10)
    
    def forward(self, x):
        x = self.fc_combine(x)
        x = self.fc_1(x)
        x = self.fc_2(x)
        x = self.fc_out(x)
        return x


class StackTailoredD(nn.Module):
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model_tailored = base_model
        self.fc_out = FCOut()

    def forward(self, x):
        x_copy = deepcopy(x)
        xout = None
        for i in range(3):
            x = x_copy[:, i, ...][:, None, ...]
            x = self.base_model_tailored(x)
            if xout is None:
                xout = x
            else:
                xout = torch.cat((xout, x), 1)

        x = self.fc_out(xout)
        return x


class StackDown(nn.Module):
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(x)
        x = self.base_model(x)
        return x

def resnet18dstacktailored():
    """Constructs a ResNet-18-D model.
    """
    base_model = resnet18d()
    return StackTailored(base_model=base_model)

def resnet18dstacktailoredD():
    """Constructs a ResNet-18-D model.
    """
    base_model = resnet18d()
    return StackTailoredD(base_model=base_model)

def resnet18dstackdown():
    """Constructs a ResNet-18-D model.
    """
    base_model = resnet18d()
    return StackDown(base_model=base_model)

def densenet121stackdown():
    """Constructs a DenseNet121 model.
    """
    base_model = densenet121()
    return StackDown(base_model=base_model)

def resnet18dstackdowntailored():
    """Constructs a ResNet-18-D model.
    """
    base_model = resnet18d()
    return StackTailored(StackDown(base_model=base_model))
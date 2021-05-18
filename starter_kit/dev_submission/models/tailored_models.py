import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional
from copy import deepcopy

from .resnet import *

__all__ = ['resnet18dstacktailored']


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


def resnet18dstacktailored():
    """Constructs a ResNet-18-D model.
    """
    base_model = resnet18d()
    return StackTailored(base_model=base_model)

from .baselines import *
from .resnet import *

global model_portfolio
model_portfolio = {
            "ResNet": ResNet,
            "ResNet18": resnet18,
            "ResNet50": resnet50,
            "ResNet101": resnet101,
            "WideResNet18": wide_resnet18_2,
            "WideResNet50": wide_resnet50_2,
            "WideResNet101": wide_resnet101_2,
            "ResNext18_32x8d": resnext18_32x8d,
            "ResNext50_32x8d": resnext50_32x4d,
            "ResNext101_32x8d": resnext101_32x8d
        }
global light_portfolio
light_portfolio = {
            "ResNet18": resnet18,
            "ResNet50": resnet50,
            "ResNet101": resnet101,
            "WideResNet18": wide_resnet18_2,
            "WideResNet50": wide_resnet50_2,
            "ResNext18_32x8d": resnext18_32x8d,
        }
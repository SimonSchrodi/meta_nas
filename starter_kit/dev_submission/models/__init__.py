from .baselines import *
from .byobnet import *
from .cspnet import *
from .densenet import *
from .efficientnet import *
from .gluon_xception import *
from .regnet import *
from .resnet import *
from .resnest import *
from .rexnet import *
from .sknet import *
from .vovnet import *
from .xception import *
from .xception_aligned import *
from .layers import *

# here just function reference to save time
global model_portfolio
model_portfolio = {
            "ResNet": ResNet,
            "ResNet18": resnet18,
            "ResNet50": resnet50,
            "ResNet101": resnet101,
            "WideResNet18": wide_resnet18_2,
            "WideResNet50": wide_resnet50_2,
            "WideResNet101": wide_resnet101_2,
            "ResNext18_32x4d": resnext18_32x4d,
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
            "ResNext18_32x4d": resnext18_32x4d,
        }

global timm_portfolio
# sub1
#timm_portfolio = {
#            "DenseNet161": densenet161,
#            "ResNest14d": resnest14d,
#            "MixNet_XXL": mixnet_xxl,
#            "DenseNet121": densenet121,
#            "DenseNetBlur121d": densenetblur121d,
#            "Gluon_xception65": gluon_xception65
#        }

# sub2
#timm_portfolio = {
#            "ResNest14d": resnest14d,
#            "DenseNet161": densenet161,
#            "VovNet39a": vovnet39a,
#            "SeResNext26t_32x4d": seresnext26t_32x4d,
#            "Gluon_xception65": gluon_xception65,
#            "SeResNext26tn_32x4d": seresnext26tn_32x4d,
#            "DenseNetBlur121d": densenetblur121d,
#        }

# sub3
#timm_portfolio = {
#            "DenseNet161": densenet161,
#            "DenseNet121": densenet121,
#            "TV_DenseNet121": tv_densenet121, 
#            "DenseNet169": densenet169,
#            "ResNest14d": resnest14d,    
#            "DenseNetBlur121d": densenetblur121d,
#            "SeResNext26t_32x4d": seresnext26t_32x4d,
#            "SeResNext26tn_32x4d": seresnext26tn_32x4d, 
#            "SeResNet50t": seresnet50t, 
#            "DenseNet121d": densenet121d, 
#            "SeResNext26d_32x4d": seresnext26d_32x4d,
#        }

# sub4
timm_portfolio = {
            "DenseNet161": densenet161,
            "ResNest14d": resnest14d,
            "Ese_VovNet39b": ese_vovnet39b, 
            "SeResNet50t": seresnet50t,
            "SeResNext26d_32x4d": seresnext26d_32x4d,
            "xception71": xception71,
            "ResNet18d": resnet18d,
            "Regnety_320": regnety_320,
            "Gernet_m": gernet_m
        }
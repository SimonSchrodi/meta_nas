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
from .tailored_models import *
from .vovnet import *
from .xception import *
from .xception_aligned import *
from .layers import *

from functools import partial

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

global timm_portfolio, timm_portfolio_a, timm_portfolio_b, timm_portfolio_c

# dummy portfolio
timm_portfolio = {"ResNet18": resnet18}

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
#timm_portfolio = {
#            "DenseNet161": partial(densenet161, drop_rate=0.2),
#            "DenseNet121": densenet121,
#            "DenseNet169": densenet169,
#            "ResNest14d": resnest14d,
#            "VovNet39a": vovnet39a,
#            "Ese_VovNet39b": ese_vovnet39b, 
#            "SeResNet50t": seresnet50t,
#            "SeResNext26d_32x4d": seresnext26d_32x4d,
#            "xception71": xception71,
#            "ResNet18d": resnet18d,
#            "Regnety_320": regnety_320,
#            "Gernet_m": gernet_m,
#            "Gluon_xception65": gluon_xception65,
#            "DenseNetBlur121d": densenetblur121d,
#            "SeResNext26tn_32x4d": seresnext26tn_32x4d
#        }

# sub5
#timm_portfolio_a = {
#            "ResNest14d_drop01": partial(resnest14d, drop_rate=0.1),
#            "ResNest14d_drop02": partial(resnest14d, drop_rate=0.2),
#            "ResNest14d_drop03": partial(resnest14d, drop_rate=0.3),
#            "ResNest14d_drop04": partial(resnest14d, drop_rate=0.4),
#            "ResNest14d_drop05": partial(resnest14d, drop_rate=0.5),
#            "ResNest14d_drop06": partial(resnest14d, drop_rate=0.6),
#            "ResNest14d_drop07": partial(resnest14d, drop_rate=0.7),
#            "ResNest14d_drop08": partial(resnest14d, drop_rate=0.8),
#        }

#timm_portfolio_b = {
#            "DenseNet121_drop01": partial(densenet121, drop_rate=0.2),
#            "StackTailored": resnet18dstacktailored,
#            "DenseNet121_drop02": partial(densenet121, drop_rate=0.1),
#            "DenseNet121_drop03": partial(densenet121, drop_rate=0.3),
#            "DenseNet121_drop04": partial(densenet121, drop_rate=0.4),
#            "DenseNet121_drop05": partial(densenet121, drop_rate=0.5),
#            "DenseNet121_drop06": partial(densenet121, drop_rate=0.6),
#        }

#timm_portfolio_c = {
#            "StackTailored": resnet18dstacktailored,
#            "DenseNet161_drop02": partial(densenet161, drop_rate=0.2),
#            "DenseNet161": densenet161,
#            "DenseNet161_drop01": partial(densenet161, drop_rate=0.1),
#            "DenseNet161_drop03": partial(densenet161, drop_rate=0.3),
#            "DenseNet161_drop04": partial(densenet161, drop_rate=0.4),
#            "DenseNet161_drop05": partial(densenet161, drop_rate=0.5),
#            "DenseNet161_drop06": partial(densenet161, drop_rate=0.6),
#        }

# sub6
#timm_portfolio_a = {
#            "ResNest14d": resnest14d,
#            "VovNet39a_drop02": partial(vovnet39a, drop_rate=0.2),
#            "xception71": xception71,
#            "VovNet39a_drop01": partial(vovnet39a, drop_rate=0.1),
#            "VovNet39a": vovnet39a,
#            "VovNet39a_drop06": partial(vovnet39a, drop_rate=0.6),
#            "VovNet39a_drop07": partial(vovnet39a, drop_rate=0.7),
#            "VovNet39a_drop08": partial(vovnet39a, drop_rate=0.8),
#        }

#timm_portfolio_b = {
#            "DenseNet121": densenet121,
#            "DenseNet101": densenet101,
#            "DenseNet101_drop02": partial(densenet101, drop_rate=0.1),
#            "DenseNet101_drop03": partial(densenet101, drop_rate=0.3),
#            "DenseNet101_drop04": partial(densenet101, drop_rate=0.4),
#            "DenseNet101_drop05": partial(densenet101, drop_rate=0.5),
#            "DenseNet101_drop06": partial(densenet101, drop_rate=0.6),
#        }

#timm_portfolio_c = {
#            "StackTailored": resnet18dstacktailored,
#            "DenseNet161": densenet161,
#            "DenseNet169": densenet169,
#            "DenseNet169_drop02": partial(densenet169, drop_rate=0.2),
#            "DenseNet169_drop01": partial(densenet169, drop_rate=0.1),
#            "DenseNet169_drop03": partial(densenet169, drop_rate=0.3),
#            "DenseNet169_drop04": partial(densenet169, drop_rate=0.4),
#            "DenseNet169_drop05": partial(densenet169, drop_rate=0.5),
#            "DenseNet169_drop06": partial(densenet169, drop_rate=0.6),
#            "DenseNet121": densenet121,
#        }

# sub 7
# timm_portfolio_a = {
#             "ResNest14d": resnest14d,
#             "DenseNet101": densenet101,
#             "VovNet39a_drop09": partial(vovnet39a, drop_rate=0.9),
#             "DenseNet121": densenet121,
#             "xception71": xception71,
#             "DenseNet101_drop09": partial(densenet101, drop_rate=0.9),
#             "VovNet39a_drop08": partial(vovnet39a, drop_rate=0.8),
#             "VovNet39a": vovnet39a,
#             "VovNet39a_drop07": partial(vovnet39a, drop_rate=0.7),
#         }

# timm_portfolio_b = {
#             "DenseNet101": densenet101,
#             "DenseNet121": densenet121,
#             "DenseNet101_drop09": partial(densenet101, drop_rate=0.9),
#             "DenseNet101_drop08": partial(densenet101, drop_rate=0.8),
#             "DenseNet101_drop07": partial(densenet101, drop_rate=0.7),
#             "DenseNet101_drop05": partial(densenet101, drop_rate=0.5),
#             "DenseNet101_drop06": partial(densenet101, drop_rate=0.6),
#         }

# timm_portfolio_c = {
#             "StackTailored": resnet18dstacktailored,
#             "DenseNet161": densenet161,
#             "DenseNet169": densenet169,
#             "DenseNet169_drop02": partial(densenet169, drop_rate=0.6),
#             "DenseNet169_drop01": partial(densenet169, drop_rate=0.1),
#             "DenseNet169_drop03": partial(densenet169, drop_rate=0.3),
#             "DenseNet169_drop04": partial(densenet169, drop_rate=0.4),
#             "DenseNet169_drop05": partial(densenet169, drop_rate=0.5),
#             "DenseNet169_drop06": partial(densenet169, drop_rate=0.6),
#             "DenseNet121": densenet121,
#         }

# sub 8
timm_portfolio_a = {
            "DenseNet101_drop09": partial(densenet101, drop_rate=0.9),
            "ResNest14d": resnest14d,
            "DenseNet101": densenet101,
            "DenseNet169_drop095": partial(densenet169, drop_rate=0.95),
            "DenseNet169_drop09": partial(densenet169, drop_rate=0.9),
            "VovNet39a_drop09": partial(vovnet39a, drop_rate=0.9),
            "DenseNet121": densenet121,
            "xception71": xception71,
            "DenseNet101_drop09": partial(densenet101, drop_rate=0.9),
            "VovNet39a_drop08": partial(vovnet39a, drop_rate=0.8),
            "VovNet39a": vovnet39a,
            "VovNet39a_drop07": partial(vovnet39a, drop_rate=0.7),
        }

timm_portfolio_b = {
            "DenseNet101": densenet101,
            "VovNet39a_drop08": partial(vovnet39a, drop_rate=0.8),
            "VovNet39a": vovnet39a,
            "VovNet39a_drop07": partial(vovnet39a, drop_rate=0.7),
            "DenseNet121": densenet121,
            "DenseNet101_drop09": partial(densenet101, drop_rate=0.9),
            "DenseNet101_drop08": partial(densenet101, drop_rate=0.8),
            "DenseNet101_drop07": partial(densenet101, drop_rate=0.7),
            "DenseNet101_drop05": partial(densenet101, drop_rate=0.5),
            "DenseNet101_drop06": partial(densenet101, drop_rate=0.6),
        }

timm_portfolio_c = {
            "DenseNet101": densenet101,
            "DenseNet161": densenet161,
            "DenseNet169_drop095": partial(densenet169, drop_rate=0.95),
            "DenseNet169_drop09": partial(densenet169, drop_rate=0.9),
            "DenseNet121": densenet121,
            "VovNet39a_drop08": partial(vovnet39a, drop_rate=0.8),
            "VovNet39a": vovnet39a,
            "VovNet39a_drop07": partial(vovnet39a, drop_rate=0.7),
            "DenseNet101_drop09": partial(densenet101, drop_rate=0.9),
            "DenseNet101_drop08": partial(densenet101, drop_rate=0.8),
            "DenseNet101_drop07": partial(densenet101, drop_rate=0.7),
            "DenseNet101_drop05": partial(densenet101, drop_rate=0.5),
            "DenseNet101_drop06": partial(densenet101, drop_rate=0.6),
        }
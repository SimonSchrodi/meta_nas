import os
import numpy as np
from functools import partial
import inspect
import time

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from dev_submission.models import *
from dev_submission.models.helpers import build_model_with_cfg

from ingestion_program.nascomp.helpers import load_datasets
from ingestion_program.nascomp.torch_evaluator import torch_evaluator

"""
This is based on the nas_benchmarks implementation
https://github.com/automl/nas_benchmarks/blob/master/tabular_benchmarks/nas_cifar10.py

Currently, multi_fidelity does not work. Arches will always be trained to self.max_epochs
"""

FULL_FAMILIES = ['resnet', 'densenet', 'efficientnet', 'gluon_xception', 'resnest', 'sknet', 'vovnet', 'cspnet']
FAMILIES = ['densenet']

POOL_SIZES = [(2, 2), (3, 3), (5, 5), (7, 7), (9, 9)]
GROWTH_RATE = {'lower': 16, 'upper':48, 'default_value':32}
DENSENET_LAYERS_PER_BLOCK = {'lower':8, 'upper':32, 'default_value':16}
BN_SIZE = {'lower':2, 'upper':8, 'default_value':4}

class NASModel(object):

    def __init__(self, dataset_path, max_epochs, multi_fidelity=False):

        self.dataset_path = dataset_path
        self.max_epochs = max_epochs
        self.multi_fidelity = multi_fidelity
        self.X = []
        self.y_valid = []
        self.costs = []

    def record_arch(self, config, valid_error, costs):
        self.X.append(config)
        self.y_valid.append(valid_error)
        self.costs.append(costs)
    
    def get_results(self, ignore_invalid_configs=False):

        validation = []
        runtime = []
        rt = 0
        inc_valid = np.inf

        for i in range(len(self.X)):

            if ignore_invalid_configs and self.costs[i] == 0:
                continue

            if inc_valid > self.y_valid[i]:
                inc_valid = self.y_valid[i]

            validation.append(float(inc_valid))
            rt += self.costs[i]
            runtime.append(float(rt))

        res = dict()
        res['validation'] = validation
        res['runtime'] = runtime

        return res

    def objective_function(self, config):
        """
        Import: config, budget
        call self.record_arch()
        Output: loss, train_time
        """

        # load the dataset
        (train_x, train_y), (valid_x, valid_y), test_x, metadata = load_datasets(self.dataset_path)
        data = (train_x, train_y), (valid_x, valid_y), test_x

        input_shape = train_x.shape[1:]
        model = self.convert_config_to_model(config, input_shape)
        model = self.reshape_model(model, channels=train_x.shape[1], n_classes=metadata['n_classes'])
        print('config is \n', config)
        print('epochs', self.max_epochs)

        start_time = time.time()
        try:
            # train model for $n_epochs, recover test predictions from best validation epoch
            results = torch_evaluator(model, data, metadata, n_epochs=self.max_epochs, full_train=True)
            predictions = results['test_predictions']
            train_details = {k: v for k, v in results.items() if k!='test_predictions'}
            print('Training succeeded!')
            print(str(train_details)+'\n\n')

            # todo: get the validation accuracy
            valid_error = 1 - train_details['best_val_score'] / 100
            runtime = time.time() - start_time
            self.record_arch(config, valid_error, runtime)
            return valid_error, runtime

        except:
            print('Training failed!\n\n')        
            runtime = time.time() - start_time
            self.record_arch(config, 1, runtime)
            return 1, runtime

    def get_configuration_space(self):

        cs = CS.ConfigurationSpace()

        ### COMMON ###
        drop_rate = CSH.UniformFloatHyperparameter('drop_rate', lower=0, upper=1, default_value = 0)
        pool_size = CSH.CategoricalHyperparameter('pool_size', POOL_SIZES, default_value = (7, 7))
        cs.add_hyperparameters([drop_rate, pool_size])

        network_family = CSH.CategoricalHyperparameter('network_family', FAMILIES)
        cs.add_hyperparameter(network_family)

        ### RESNET ###
        if 'resnet' in FAMILIES:
            block = CSH.CategoricalHyperparameter('block', ['basic', 'bottleneck'])
            cardinality = CSH.UniformIntegerHyperparameter("cardinality", lower=1, upper=32, default_value = 1)
            base_width = CSH.UniformIntegerHyperparameter("base_width", lower=8, upper=256, log=True, default_value = 64)
            stem_width = CSH.UniformIntegerHyperparameter("stem_width", lower=8, upper=64, log=True, default_value = 64)
            resnet_stem_type = CSH.CategoricalHyperparameter("resnet_stem_type", ['', 'deep', 'deep_tiered'], default_value = '')
            avg_down = CSH.CategoricalHyperparameter("avg_down", [True, False], default_value = False)
            attn_layer = CSH.CategoricalHyperparameter("attn_layer", ['', 'se', 'eca', partial(layers.get_attn('se'), 
                                                                                            reduction_ratio=0.25)], default_value = '')
            cs.add_hyperparameters([block, cardinality, base_width, stem_width, resnet_stem_type, avg_down, attn_layer])

            block_cond = CS.EqualsCondition(block, network_family, 'resnet')
            stem_width_cond = CS.EqualsCondition(stem_width, network_family, 'resnet')
            stem_type_cond = CS.EqualsCondition(resnet_stem_type, network_family, 'resnet')
            avg_down_cond = CS.EqualsCondition(avg_down, network_family, 'resnet')
            attn_layer_cond = CS.EqualsCondition(attn_layer, network_family, 'resnet')

            cardinality_cond = CS.EqualsCondition(cardinality, block, 'bottleneck')
            base_width_cond = CS.EqualsCondition(base_width, block, 'bottleneck')
            cs.add_conditions([block_cond, stem_width_cond, stem_type_cond, avg_down_cond, 
                               cardinality_cond, base_width_cond, attn_layer_cond])

            layers_per_block = CSH.UniformIntegerHyperparameter("layers_per_block", lower=1, upper=32, log=True)
            layers_per_block_cond = CS.EqualsCondition(layers_per_block, network_family, 'resnet')
            cs.add_hyperparameter(layers_per_block)
            cs.add_condition(layers_per_block_cond)

        ### DENSENET ###
        if 'densenet' in FAMILIES:
            growth_rate = CSH.UniformIntegerHyperparameter("growth_rate", lower=GROWTH_RATE['lower'], 
                                                        upper=GROWTH_RATE['upper'], default_value=GROWTH_RATE['default_value'])
            densenet_layers_per_block = CSH.UniformIntegerHyperparameter("densenet_layers_per_block", 
                                                                        lower=DENSENET_LAYERS_PER_BLOCK['lower'], 
                                                                        upper=DENSENET_LAYERS_PER_BLOCK['upper'], 
                                                                        log=True, 
                                                                        default_value=DENSENET_LAYERS_PER_BLOCK['default_value'])
            bn_size = CSH.UniformIntegerHyperparameter("bn_size", lower=BN_SIZE['lower'], 
                                                    upper=BN_SIZE['upper'], 
                                                    default_value=BN_SIZE['default_value'])
            densenet_stem_type = CSH.CategoricalHyperparameter("densenet_stem_type", ['', 'deep'], default_value = '')
            cs.add_hyperparameters([growth_rate, bn_size, densenet_stem_type, densenet_layers_per_block])

            growth_rate_cond = CS.EqualsCondition(growth_rate, network_family, 'densenet')
            densenet_layers_per_block_cond = CS.EqualsCondition(densenet_layers_per_block, network_family, 'densenet')
            bn_size_cond = CS.EqualsCondition(bn_size, network_family, 'densenet')
            stem_type_cond = CS.EqualsCondition(densenet_stem_type, network_family, 'densenet')
            cs.add_conditions([growth_rate_cond, densenet_layers_per_block_cond, bn_size_cond, stem_type_cond])

        ### EFFICIENTNET ###
        if 'efficientnet' in FAMILIES:
            channel_multiplier = CSH.UniformFloatHyperparameter('channel_multiplier', lower=0.1, upper=2.0, default_value = 1.0)
            depth_multiplier = CSH.UniformFloatHyperparameter('depth_multiplier', lower=0.1, upper=2.0, default_value = 1.0)
            cs.add_hyperparameters([channel_multiplier, depth_multiplier])

            channel_multiplier_cond = CS.EqualsCondition(channel_multiplier, network_family, 'efficientnet')
            depth_multiplier_cond = CS.EqualsCondition(depth_multiplier, network_family, 'efficientnet')
            cs.add_conditions([channel_multiplier_cond, depth_multiplier_cond])

        # VOVNET
        if 'vovnet' in FAMILIES:
            vovnet_model_cfg = CSH.CategoricalHyperparameter("vovnet_model_cfg", ['vovnet39a', 'ese_vovnet39b', 'eca_vovnet39b', 'ese_vovnet19b_slim'], default_value = 'vovnet39a')
    
            vovnet_model_cfg_cond = CS.EqualsCondition(vovnet_model_cfg, network_family, 'vovnet')
            cs.add_hyperparameter(vovnet_model_cfg)
            cs.add_condition(vovnet_model_cfg_cond)

        # CSPNET
        if 'cspnet' in FAMILIES:
            cspnet_model_cfg = CSH.CategoricalHyperparameter("cspnet_model_cfg", ['cspresnet50', 'cspresnet50d', 'cspresnet50w'], default_value = 'cspresnet50')
        
            cspnet_model_cfg_cond = CS.EqualsCondition(cspnet_model_cfg, network_family, 'cspnet')
            cs.add_hyperparameter(cspnet_model_cfg)
            cs.add_condition(cspnet_model_cfg_cond)

        # RESNEST
        if 'resnest' in FAMILIES:
            resnest_layers_per_block = CSH.UniformIntegerHyperparameter("resnest_layers_per_block", lower=1, upper=6)

            resnest_layers_per_block_cond = CS.EqualsCondition(resnest_layers_per_block, network_family, 'resnest')
            cs.add_hyperparameter(resnest_layers_per_block)
            cs.add_condition(resnest_layers_per_block_cond)

        return cs

    def convert_config_to_model(self, config, input_shape):

        drop_rate = config['drop_rate']
        pool_size = config['pool_size']

        network_name = config['network_family']

        if network_name == 'resnet':
            default_cfg = resnet._cfg()
            default_cfg['input_size'] = input_shape
            default_cfg['pool_size'] = pool_size

            layers = [config['layers_per_block']]*4

            block = config['block']
            if block == 'bottleneck':
                block = resnet.Bottleneck
                cardinality = config['cardinality']
                base_width = config['base_width']
            elif block == 'basic':
                block = resnet.BasicBlock
                cardinality = 1
                base_width = 64

            stem_width = config['stem_width']
            stem_type = config['resnet_stem_type']
            avg_down = config['avg_down']
            attn_layer = None if config['attn_layer']=='' else config['attn_layer']

            model = build_model_with_cfg(resnet.ResNet, 'custom_nas', False, default_cfg=default_cfg,
                                block = block, 
                                layers = layers,
                                cardinality = cardinality, 
                                base_width = base_width, 
                                stem_width = stem_width, 
                                stem_type = stem_type, 
                                avg_down=avg_down, 
                                block_args=dict(attn_layer=attn_layer),
                                drop_rate=drop_rate)


        elif network_name == 'densenet':
            default_cfg = densenet._cfg()
            default_cfg['input_size'] = input_shape
            default_cfg['pool_size'] = pool_size

            layers = [config['densenet_layers_per_block']]*4

            growth_rate = config['growth_rate']
            bn_size = config['bn_size']
            stem_type = config['densenet_stem_type']

            model = build_model_with_cfg(densenet.DenseNet, 'custom_nas', False, default_cfg=default_cfg,
                                growth_rate = growth_rate, 
                                block_config = layers, 
                                bn_size = bn_size, 
                                stem_type = stem_type, 
                                drop_rate = drop_rate)

        elif network_name == 'efficientnet':
            default_cfg = efficientnet._cfg()
            default_cfg['input_size'] = input_shape
            default_cfg['pool_size'] = pool_size

            channel_multiplier = config['channel_multiplier']
            depth_multiplier = config['depth_multiplier']

            arch_def = [
            ['ds_r1_k3_s1_e1_c16_se0.25'],
            ['ir_r2_k3_s2_e6_c24_se0.25'],
            ['ir_r2_k5_s2_e6_c40_se0.25'],
            ['ir_r3_k3_s2_e6_c80_se0.25'],
            ['ir_r3_k5_s1_e6_c112_se0.25'],
            ['ir_r4_k5_s2_e6_c192_se0.25'],
            ['ir_r1_k3_s1_e6_c320_se0.25'],
            ]

            block_args = efficientnet_builder.decode_arch_def(arch_def, depth_multiplier)
            model = build_model_with_cfg(efficientnet.EfficientNet, 'custom_nas', False, default_cfg=default_cfg, 
                                block_args = block_args,
                                channel_multiplier = channel_multiplier,
                                drop_rate = drop_rate) 

        elif network_name == 'cspnet':
            default_cfg = cspnet._cfg()
            default_cfg['input_size'] = input_shape 
            default_cfg['pool_size'] = pool_size

            variant = config['cspnet_model_cfg']

            model = build_model_with_cfg(cspnet.CspNet, 'custom_nas', False, default_cfg=default_cfg,
                                feature_cfg=dict(flatten_sequential=True),
                                model_cfg=cspnet.model_cfgs[variant],
                                drop_rate = drop_rate) # add other kwargs

        elif network_name == 'gluon_xception':
            default_cfg = gluon_xception.default_cfgs
            default_cfg['input_size'] = input_shape 
            default_cfg['pool_size'] = pool_size

            model = build_model_with_cfg(gluon_xception.Xception65, 'custom_nas', False, default_cfg=default_cfg,
                                feature_cfg=dict(feature_cls='hook'),
                                drop_rate = drop_rate) # add other kwargs

        elif network_name == 'resnest':
            default_cfg = resnest._cfg()
            default_cfg['input_size'] = input_shape 
            default_cfg['pool_size'] = pool_size

            layers = [config['resnest_layers_per_block']]*4

            model = build_model_with_cfg(resnet.ResNet, 'custom_nas', False, default_cfg=default_cfg,
                                block = resnest.ResNestBottleneck,
                                layers = layers,
                                stem_type='deep', 
                                stem_width=32, 
                                avg_down=True, 
                                base_width=64, 
                                cardinality=1,
                                block_args=dict(radix=2, avd=True, avd_first=False),
                                drop_rate = drop_rate)

        elif network_name == 'sknet':
            default_cfg = sknet._cfg()
            default_cfg['input_size'] = input_shape 
            default_cfg['pool_size'] = pool_size

            model = build_model_with_cfg(resnet.ResNet, 'custom_nas', False, default_cfg=default_cfg, 
                                block=sknet.SelectiveKernelBasic,
                                layers=[2, 2, 2, 2], 
                                block_args=dict(sk_kwargs=dict(min_attn_channels=16, attn_reduction=8, split_input=True)),
                                zero_init_last_bn=False,
                                drop_rate = drop_rate)

        elif network_name == 'vovnet':
            default_cfg = vovnet._cfg()
            default_cfg['input_size'] = input_shape 
            default_cfg['pool_size'] = pool_size

            variant = config['vovnet_model_cfg']

            model = build_model_with_cfg(vovnet.VovNet, 'custom_nas', False, default_cfg=default_cfg,
                                feature_cfg=dict(flatten_sequential=True),
                                model_cfg=vovnet.model_cfgs[variant],
                                drop_rate = drop_rate)

        return model


    def reshape_model(self, model: nn.Module, channels: int, n_classes: int, copy_type: str = 'Starter') -> nn.Module:
        """Reshapes input and output for the model. Note that currently this does not detect models
        which are not suited for the given input data size!

        Args:
            model (nn.Module): PyTorch model
            channels (int): input channels
            n_classes (int): number of classes

        Raises:
            NotImplementedError: reshape for model is not implemented

        Returns:
            nn.Module: model with reshaped input and output layer
        """
        def _get_member_str(model: nn.Module, first: bool):
            model_state_dict = list(model.state_dict())
            idx = 0 if first else -1
            member_str = model_state_dict[idx][:model_state_dict[idx].rfind('.')]
            return member_str

        def _reshape(model: nn.Module, member_str: str, channels:int = None, n_classes:int = None, copy_type='Copy'):
            def __build_instructions(obj, q):
                """
                Breaks down a query string into a series of actionable instructions.

                Each instruction is a (_type, arg) tuple.
                arg -- The key used for the __getitem__ or __setitem__ call on
                    the current object.
                _type -- Used to determine the data type for the value of
                        obj.__getitem__(arg)

                If a key/index is missing, _type is used to initialize an empty value.
                In this way _type provides the ability to
                """
                arg = []
                _type = None
                instructions = []
                for i, ch in enumerate(q):
                    if ch == "[":
                        # Begin list query
                        if _type is not None:
                            arg = "".join(arg)
                            if _type == list and arg.isalpha():
                                _type = dict
                            instructions.append((_type, arg))
                            _type, arg = None, []
                        _type = list
                    elif ch == ".":
                        # Begin dict query
                        if _type is not None:
                            arg = "".join(arg)
                            if _type == list and arg.isalpha():
                                _type = dict
                            instructions.append((_type, arg))
                            _type, arg = None, []

                        _type = dict
                    elif ch.isalnum() or ch == "_" or ch == '-':
                        if i == 0:
                            # Query begins with alphanum, assume dict access
                            _type = type(obj)

                        # Fill out args
                        arg.append(ch)
                    else:
                        TypeError("Unrecognized character: {}".format(ch))

                if _type is not None:
                    # Finish up last query
                    instructions.append((_type, "".join(arg)))

                return instructions

            def _setattr(obj, query, val):
                """
                This is a special setattr function that will take in a string query,
                interpret it, add the appropriate data structure to obj, and set val.
                """
                instructions = __build_instructions(obj, query)
                for i, (_, arg) in enumerate(instructions[:-1]):
                    _type = instructions[i + 1][0]
                    obj = _set(obj, _type, arg)

                _type, arg = instructions[-1]
                _set(obj, _type, arg, val)

            def _set(obj, _type, arg, val=None):
                """
                Helper function for calling obj.__setitem__(arg, val or _type()).
                """
                if val is not None:
                    # Time to set our value
                    _type = type(val)

                if isinstance(obj, dict):
                    if arg not in obj:
                        # If key isn't in obj, initialize it with _type()
                        # or set it with val
                        obj[arg] = (_type() if val is None else val)
                    obj = obj[arg]
                elif isinstance(obj, list):
                    n = len(obj)
                    arg = int(arg)
                    if n > arg:
                        obj[arg] = (_type() if val is None else val)
                    else:
                        # Need to amplify our list, initialize empty values with _type()
                        obj.extend([_type() for x in range(arg - n + 1)])
                    obj = obj[arg]
                elif isinstance(obj, nn.Module):
                    if val is not None:
                        setattr(obj, arg, val)
                    else:
                        obj = getattr(obj, arg)
                return obj

            def _getattr(obj, query):
                """
                Very similar to _setattr. Instead of setting attributes they will be
                returned. As expected, an error will be raised if a __getitem__ call
                fails.
                """
                instructions = __build_instructions(obj, query)
                for i, (_, arg) in enumerate(instructions[:-1]):
                    _type = instructions[i + 1][0]
                    obj = _get(obj, _type, arg)

                _type, arg = instructions[-1]
                return _get(obj, _type, arg)

            def _get(obj, _type, arg):
                """
                Helper function for calling obj.__getitem__(arg).
                """
                if isinstance(obj, dict):
                    obj = obj[arg]
                elif isinstance(obj, list):
                    arg = int(arg)
                    obj = obj[arg]
                elif isinstance(obj, nn.Module):
                    try:
                        arg = int(arg)
                        obj = obj[arg]
                    except ValueError:
                        obj = getattr(obj, arg)
                return obj

            def __create_new_layer(old_member, copy_type: str, val:int, is_input: bool):
                def _get_default_args(func):
                    signature = inspect.signature(func)
                    return {
                        k: v.default
                        for k, v in signature.parameters.items()
                        if v.default is not inspect.Parameter.empty
                    }

                func = getattr(nn, copy_member.__class__.__name__)

                # get and set default parameters
                kwargs = _get_default_args(func)
                if 'Conv2d' in copy_member.__class__.__name__:
                    kwargs['in_channels'] = channels if is_input else copy_member.in_channels
                    kwargs['out_channels'] = copy_member.out_channels if is_input else n_classes
                    kwargs['kernel_size'] = copy_member.kernel_size
                elif copy_member.__class__.__name__ == 'Linear':
                    kwargs['in_features'] = channels if is_input else copy_member.in_features
                    kwargs['out_features'] = copy_member.out_features if is_input else n_classes
                else:
                    raise NotImplementedError

                if copy_type == 'Default':
                    pass
                elif copy_type == 'Copy':
                    if 'Conv2d' in copy_member.__class__.__name__:
                        for key in ['stride', 'padding', 'padding_mode', 'dilation', 'groups']:
                            if copy_member.__class__.__name__ == 'Conv2dSame':    
                                continue
                            kwargs[key] = getattr(copy_member, key)
                    kwargs['bias'] = True if copy_member.bias is not None else False # same for linear
                elif copy_type == 'Starter':
                    if 'Conv2d' in copy_member.__class__.__name__:
                        kwargs['stride'] = 1
                        kwargs['padding'] = 3
                    elif copy_member.__class__.__name__ == 'Linear':
                        kwargs['bias']= True
                else:
                    raise NotImplementedError

                return func(**kwargs)


            if not ((channels and not n_classes) or (not channels and n_classes)):
                raise Exception('Not allowed')
            is_input = channels is not None
            copy_member = _getattr(model, member_str)
            new_layer = __create_new_layer(copy_member, copy_type, channels if is_input else n_classes, is_input)
            _setattr(model, member_str, new_layer)


        # first layer
        member_str = _get_member_str(model, True) 
        _reshape(model, member_str, channels=channels, copy_type=copy_type)

        # last layer
        member_str = _get_member_str(model, False) 
        _reshape(model, member_str, n_classes=n_classes, copy_type=copy_type)

        return model

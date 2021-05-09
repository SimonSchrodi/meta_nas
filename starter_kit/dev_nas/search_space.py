from functools import partial
from dev_submission.models import *
from dev_submission.models.helpers import build_model_with_cfg
from dev_submission.nas_helpers import reshape_model

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

def get_configspace():
    

    cs = CS.ConfigurationSpace()

    ### COMMON ###
    drop_rate = CSH.UniformFloatHyperparameter('drop_rate', lower=0, upper=1, default_value = 0)
    pool_size = CSH.CategoricalHyperparameter('pool_size', [(2, 2), (3, 3), (5, 5), (7, 7), (9, 9)], default_value = (7, 7))
    cs.add_hyperparameters([drop_rate, pool_size])

    network_family = CSH.CategoricalHyperparameter('network_family', ['resnet', 'densenet', 'efficientnet', 'gluon_xception', 'resnest', 'sknet', 'vovnet', 'cspnet'])
    cs.add_hyperparameter(network_family)

    ### RESNET ###

    block = CSH.CategoricalHyperparameter('block', ['basic', 'bottleneck'])
    cardinality = CSH.UniformIntegerHyperparameter("cardinality", lower=1, upper=32, default_value = 1)
    base_width = CSH.UniformIntegerHyperparameter("base_width", lower=8, upper=256, log=True, default_value = 64)
    stem_width = CSH.UniformIntegerHyperparameter("stem_width", lower=8, upper=64, log=True, default_value = 64)
    resnet_stem_type = CSH.CategoricalHyperparameter("resnet_stem_type", ['', 'deep', 'deep_tiered'], default_value = '')
    avg_down = CSH.CategoricalHyperparameter("avg_down", [True, False], default_value = False)
    attn_layer = CSH.CategoricalHyperparameter("attn_layer", ['', 'se', 'eca', partial(layers.get_attn('se'), reduction_ratio=0.25)], default_value = '')
    cs.add_hyperparameters([block, cardinality, base_width, stem_width, resnet_stem_type, avg_down, attn_layer])

    block_cond = CS.EqualsCondition(block, network_family, 'resnet')
    stem_width_cond = CS.EqualsCondition(stem_width, network_family, 'resnet')
    stem_type_cond = CS.EqualsCondition(resnet_stem_type, network_family, 'resnet')
    avg_down_cond = CS.EqualsCondition(avg_down, network_family, 'resnet')
    attn_layer_cond = CS.EqualsCondition(attn_layer, network_family, 'resnet')

    cardinality_cond = CS.EqualsCondition(cardinality, block, 'bottleneck')
    base_width_cond = CS.EqualsCondition(base_width, block, 'bottleneck')
    cs.add_conditions([block_cond, stem_width_cond, stem_type_cond, avg_down_cond, cardinality_cond, base_width_cond, attn_layer_cond])

    layers_per_block = CSH.UniformIntegerHyperparameter("layers_per_block", lower=1, upper=32, log=True)
    layers_per_block_cond = CS.EqualsCondition(layers_per_block, network_family, 'resnet')
    cs.add_hyperparameter(layers_per_block)
    cs.add_condition(layers_per_block_cond)

    ### DENSENET ###

    growth_rate = CSH.UniformIntegerHyperparameter("growth_rate", lower=8, upper=48, default_value = 32)
    densenet_layers_per_block = CSH.UniformIntegerHyperparameter("densenet_layers_per_block", lower=1, upper=64, log=True, default_value = 16)
    bn_size = CSH.UniformIntegerHyperparameter("bn_size", lower=1, upper=16, default_value = 4)
    densenet_stem_type = CSH.CategoricalHyperparameter("densenet_stem_type", ['', 'deep'], default_value = '')
    cs.add_hyperparameters([growth_rate, bn_size, densenet_stem_type, densenet_layers_per_block])

    growth_rate_cond = CS.EqualsCondition(growth_rate, network_family, 'densenet')
    densenet_layers_per_block_cond = CS.EqualsCondition(densenet_layers_per_block, network_family, 'densenet')
    bn_size_cond = CS.EqualsCondition(bn_size, network_family, 'densenet')
    stem_type_cond = CS.EqualsCondition(densenet_stem_type, network_family, 'densenet')
    cs.add_conditions([growth_rate_cond, densenet_layers_per_block_cond, bn_size_cond, stem_type_cond])

    ### EFFICIENTNET ###

    channel_multiplier = CSH.UniformFloatHyperparameter('channel_multiplier', lower=0.1, upper=2.0, default_value = 1.0)
    depth_multiplier = CSH.UniformFloatHyperparameter('depth_multiplier', lower=0.1, upper=2.0, default_value = 1.0)
    cs.add_hyperparameters([channel_multiplier, depth_multiplier])

    channel_multiplier_cond = CS.EqualsCondition(channel_multiplier, network_family, 'efficientnet')
    depth_multiplier_cond = CS.EqualsCondition(depth_multiplier, network_family, 'efficientnet')
    cs.add_conditions([channel_multiplier_cond, depth_multiplier_cond])

    # VOVNET

    vovnet_model_cfg = CSH.CategoricalHyperparameter("vovnet_model_cfg", ['vovnet39a', 'ese_vovnet39b', 'eca_vovnet39b', 'ese_vovnet19b_slim'], default_value = 'vovnet39a')
    vovnet_model_cfg_cond = CS.EqualsCondition(vovnet_model_cfg, network_family, 'vovnet')
    cs.add_hyperparameter(vovnet_model_cfg)
    cs.add_condition(vovnet_model_cfg_cond)

    # CSPNET

    cspnet_model_cfg = CSH.CategoricalHyperparameter("cspnet_model_cfg", ['cspresnet50', 'cspresnet50d', 'cspresnet50w'], default_value = 'cspresnet50')
    cspnet_model_cfg_cond = CS.EqualsCondition(cspnet_model_cfg, network_family, 'cspnet')
    cs.add_hyperparameter(cspnet_model_cfg)
    cs.add_condition(cspnet_model_cfg_cond)

    # RESNEST

    resnest_layers_per_block = CSH.UniformIntegerHyperparameter("resnest_layers_per_block", lower=1, upper=6)
    resnest_layers_per_block_cond = CS.EqualsCondition(resnest_layers_per_block, network_family, 'resnest')
    cs.add_hyperparameter(resnest_layers_per_block)
    cs.add_condition(resnest_layers_per_block_cond)

    return cs

def sample_config_return_model(config, input_shape):
    
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

if __name__ == "__main__":

    '''
    Iterate over devel datasets, sample random configs
    '''
 
    from ingestion_program.nascomp.helpers import get_dataset_paths, load_datasets
    from ingestion_program.nascomp.torch_evaluator import torch_evaluator

    input_dir = 'public_data_12-03-2021_13-33'

    cs = get_configspace()

    f = open('search_space_validation_summary.txt', 'w+')

    for dataset_path in get_dataset_paths(input_dir):
        (train_x, train_y), (valid_x, valid_y), test_x, metadata = load_datasets(dataset_path)

        # package data for evaluator
        data = (train_x, train_y), (valid_x, valid_y), test_x

        input_shape = train_x.shape[1:]
        for _ in range(10):
            config = cs.sample_configuration()
            model= sample_config_return_model(config, input_shape)
            model = reshape_model(model, channels=train_x.shape[1], n_classes=metadata['n_classes'])
            
            print(config)
            f.write(metadata['name']+'\n')
            f.write(str(config)+'\n')
            f.write(str(model.default_cfg)+'\n')
            
            try:
                # train model for $n_epochs, recover test predictions from best validation epoch
                results = torch_evaluator(model, data, metadata, n_epochs=1, full_train=True)
                predictions = results['test_predictions']
                train_details = {k: v for k, v in results.items() if k!='test_predictions'}
                f.write(str(train_details)+'\n\n')
            except:
                f.write('Training failed!\n\n')
                continue

    f.close()


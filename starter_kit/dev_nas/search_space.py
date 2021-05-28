import os
import sys
from time import time
sys.path.append(os.getcwd())
sys.path.append('dev_submission/')
from torch import nn

from models import *
from models.layers import BlurPool2d
from models.helpers import build_model_with_cfg
from nas_helpers import reshape_model

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

FAMILIES = {'eval_0_densenet': ['densenet'],
            'eval_0_resnest': ['resnest'],
            'eval_1_densenet': ['densenet'],
            'eval_1_resnest': ['resnest'],
            'eval_2_resnest': ['resnest'],
            'eval_2_vovnet': ['vovnet']}

def get_configspace(network_name):
    
    cs = CS.ConfigurationSpace()

    ### COMMON ###
    drop_rate = CSH.UniformFloatHyperparameter('drop_rate', lower=0, upper=1, default_value = 0)
    pool_size = CSH.CategoricalHyperparameter('pool_size', [(2, 2), (3, 3), (5, 5), (7, 7), (9, 9)], default_value = (7, 7))
    cs.add_hyperparameters([drop_rate, pool_size])

    if 'densenet' == network_name:
        growth_rate = CSH.UniformIntegerHyperparameter("growth_rate", lower=16, upper=48, default_value = 32)
        layers_I = CSH.UniformIntegerHyperparameter("layers_I", lower=3, upper=12)
        layers_II = CSH.UniformIntegerHyperparameter("layers_II", lower=6, upper=18)
        layers_III = CSH.UniformIntegerHyperparameter("layers_III", lower=6, upper=48)
        layers_IV = CSH.UniformIntegerHyperparameter("layers_IV", lower=6, upper=48)
        densenet_stem_type = CSH.CategoricalHyperparameter("densenet_stem_type", ['', 'deep'], default_value = '')
        aa_layer = CSH.CategoricalHyperparameter('aa_layer', [True, False], default_value = False)
        cs.add_hyperparameters([growth_rate, layers_I, layers_II, layers_III, layers_IV, densenet_stem_type, aa_layer])

    if 'resnest' == network_name:
        layers_I = CSH.UniformIntegerHyperparameter("layers_I", lower=1, upper=6)
        layers_II = CSH.UniformIntegerHyperparameter("layers_II", lower=1, upper=6)
        layers_III = CSH.UniformIntegerHyperparameter("layers_III", lower=1, upper=6)
        layers_IV = CSH.UniformIntegerHyperparameter("layers_IV", lower=1, upper=6)
        stem_width = CSH.UniformIntegerHyperparameter("stem_width", lower=24, upper=64, default_value = 32)

        cs.add_hyperparameters([layers_I, layers_II, layers_III, layers_IV, stem_width])

    # VOVNET
    if 'vovnet' == network_name:
        layer_per_block = CSH.UniformIntegerHyperparameter("layer_per_block", lower=2, upper=7)
        residual = CSH.CategoricalHyperparameter('residual', [True, False], default_value = False)
        depthwise = CSH.CategoricalHyperparameter('depthwise', [True, False], default_value = False)
        attn = CSH.CategoricalHyperparameter('attn', ['', 'ese', 'eca'], default_value = '')
        cs.add_hyperparameters([layer_per_block, residual, depthwise, attn])
        
        res_cond = CS.EqualsCondition(residual, depthwise, False)
        cs.add_condition(res_cond)
    
    return cs

def return_model(config, network_name, input_shape):

    drop_rate = config['drop_rate']
    pool_size = config['pool_size']
    
    if network_name == 'densenet':
        default_cfg = densenet._cfg()
        default_cfg['input_size'] = input_shape
        default_cfg['pool_size'] = pool_size

        growth_rate = config['growth_rate']
        layers = [config['layers_I'], config['layers_II'], config['layers_III'], config['layers_IV']]
        stem_type = config['densenet_stem_type']
        aa_layer = BlurPool2d if config['aa_layer'] else None

        model = build_model_with_cfg(densenet.DenseNet, 'custom_nas', False, default_cfg=default_cfg,
                            growth_rate = growth_rate, 
                            block_config = layers,
                            stem_type = stem_type,
                            drop_rate = drop_rate)

    elif network_name == 'resnest':
        default_cfg = resnest._cfg()
        default_cfg['input_size'] = input_shape 
        default_cfg['pool_size'] = pool_size

        layers = [config['layers_I'], config['layers_II'], config['layers_III'], config['layers_IV']]
        stem_width = config['stem_width']

        model = build_model_with_cfg(resnet.ResNet, 'custom_nas', False, default_cfg=default_cfg,
                            block = resnest.ResNestBottleneck,
                            layers = layers,
                            stem_type='deep', 
                            stem_width=stem_width, 
                            avg_down=True, 
                            base_width=64, 
                            cardinality=1,
                            block_args=dict(radix=2, avd=True, avd_first=False),
                            drop_rate = drop_rate)

    elif network_name == 'vovnet':
        default_cfg = vovnet._cfg()
        default_cfg['input_size'] = input_shape 
        default_cfg['pool_size'] = pool_size

        model_cfg = vovnet.model_cfgs['vovnet39a']
        model_cfg['layer_per_block'] = config['layer_per_block']
        model_cfg['depthwise'] = config['depthwise']
        model_cfg['attn'] = config['attn']
        
        if not config['depthwise']:
            model_cfg['residual'] = config['residual']

        model = build_model_with_cfg(vovnet.VovNet, 'custom_nas', False, default_cfg=default_cfg,
                            feature_cfg=dict(flatten_sequential=True),
                            model_cfg=model_cfg,
                            drop_rate = drop_rate)

    return model

def validation(network_name, config, dataset_path, output_file):
    (train_x, train_y), (valid_x, valid_y), test_x, metadata = load_datasets(dataset_path)
    data = (train_x, train_y), (valid_x, valid_y), test_x # package data for evaluator
    input_shape = train_x.shape[1:]

    model = return_model(config, network_name, input_shape)
    model = reshape_model(model, channels=train_x.shape[1], n_classes=metadata['n_classes'])
    
    # train model for $n_epochs, recover test predictions from best validation epoch
    start = time()
    results = torch_evaluator(model, data, metadata, n_epochs=64, full_train=True)
    end = time()
    predictions = results['test_predictions']
    ref_y = np.load(os.path.join(dataset_path, 'test_y.npy')) # load the reference values
    score = sum(ref_y == predictions)/float(len(ref_y)) * 100 # calculate the test score
    train_details = {k: v for k, v in results.items() if k!='test_predictions'}
    train_details['test_score'] = score # record test score to the dictionary
    train_details['training_time'] = end-start
    
    f = open(output_file, 'w+')
    f.write(metadata['name']+'\n')
    f.write(str(config)+'\n')
    f.write(str(model.default_cfg)+'\n')
    f.write(str(train_details)+'\n\n')
    f.close()

if __name__ == "__main__":

    from ingestion_program.nascomp.helpers import get_dataset_paths, load_datasets
    from ingestion_program.nascomp.torch_evaluator import torch_evaluator
    
    import numpy as np

    ##############
    ### EVAL 0 ###
    ##############
    #1
    network_name = 'densenet'
    config = {"aa_layer": True,"densenet_stem_type": "deep","drop_rate": 0.11066464826204615,"growth_rate": 30,"layers_I": 8,"layers_II": 18,"layers_III": 10,"layers_IV": 24,"pool_size": (3,3)} # "origin": "Local Search"
    dataset_path = '/work/dlclarge2/ozturk-nascomp_track_3/meta_dataset_finalized/eval_0_datasets/1-emnist_balanced_dataset'
    output_file = 'eval_0/densenet_incumbent_1-emnist_balanced.txt'
    #validation(network_name, config, dataset_path, output_file)
    #RUNNING #DONE

    #2
    network_name = 'resnest'
    config = {"drop_rate": 0.8326211678917067, "layers_I": 6, "layers_II": 2, "layers_III": 1, "layers_IV": 1,"pool_size": [9,9],"stem_width": 54} # "origin": "Local Search"
    dataset_path = '/work/dlclarge2/ozturk-nascomp_track_3/meta_dataset_finalized/eval_0_datasets/1-emnist_balanced_dataset'
    output_file = 'eval_0/resnest_incumbent_1-emnist_balanced.txt'
    #validation(network_name, config, dataset_path, output_file)
    #RUNNING #DONE

    #3
    network_name = 'densenet'
    config = {"aa_layer": False,"densenet_stem_type": "deep","drop_rate": 0.09556629879001344,"growth_rate": 48,"layers_I": 4,"layers_II": 9,"layers_III": 29,"layers_IV": 36,"pool_size": (3,3)} # "origin": "Random Search (sorted)"
    dataset_path = '/work/dlclarge2/ozturk-nascomp_track_3/meta_dataset_finalized/eval_0_datasets/5-emnist_balanced_dataset'
    output_file = 'eval_0/densenet_incumbent_5-emnist_balanced.txt'
    #validation(network_name, config, dataset_path, output_file)
    #RUNNING #DONE

    #4
    network_name = 'resnest'
    config = {"drop_rate": 0.7108079675045018, "layers_I": 4, "layers_II": 1, "layers_III": 1, "layers_IV": 1,"pool_size": [7,7],"stem_width": 29} # "origin": "Local Search"
    dataset_path = '/work/dlclarge2/ozturk-nascomp_track_3/meta_dataset_finalized/eval_0_datasets/5-emnist_balanced_dataset'
    output_file = 'eval_0/resnest_incumbent_5-emnist_balanced.txt'
    #validation(network_name, config, dataset_path, output_file)
    #RUNNING #DONE

    #5
    network_name = 'densenet'
    config = {"aa_layer": False,"densenet_stem_type": "","drop_rate": 0.02941226271493451,"growth_rate": 48,"layers_I": 8,"layers_II": 11,"layers_III": 39,"layers_IV": 48,"pool_size": (3,3)} # "origin": "Local Search"
    dataset_path = '/work/dlclarge2/ozturk-nascomp_track_3/meta_dataset_finalized/eval_0_datasets/devel_dataset_0'
    output_file = 'eval_0/densenet_incumbent_devel_dataset_0.txt'
    #validation(network_name, config, dataset_path, output_file)
    #RUNNING #>1HRS #DONE

    #6
    network_name = 'resnest'
    config = {"drop_rate": 0.13800114829672555, "layers_I": 6, "layers_II": 1, "layers_III": 3, "layers_IV": 1,"pool_size": (5,5),"stem_width": 59} # "origin": "Local Search"
    dataset_path = '/work/dlclarge2/ozturk-nascomp_track_3/meta_dataset_finalized/eval_0_datasets/devel_dataset_0'
    output_file = 'eval_0/resnest_incumbent_devel_dataset_0.txt'
    #validation(network_name, config, dataset_path, output_file)
    #RUNNING #DONE

    ##############
    ### EVAL 2 ###
    ##############
    #1
    network_name = 'resnest'
    config = {"drop_rate": 0.21046366453023607, "layers_I": 1, "layers_II": 1, "layers_III": 2, "layers_IV": 1,"pool_size": (2,2),"stem_width": 38} # "origin": "Local Search"
    dataset_path = '/work/dlclarge2/ozturk-nascomp_track_3/meta_dataset_finalized/eval_2_datasets/devel_dataset_2'
    output_file = 'eval_2/resnest_incumbent_devel_dataset_2.txt'
    #validation(network_name, config, dataset_path, output_file)
    #RUNNING #DONE
    
    #2
    network_name = 'vovnet'
    config = {"attn": "eca","depthwise": False,"drop_rate": 0.35332097362419645, "layer_per_block": 5, "pool_size": [3,3], "residual": False} # "origin": "Random Search (sorted)"
    dataset_path = '/work/dlclarge2/ozturk-nascomp_track_3/meta_dataset_finalized/eval_2_datasets/devel_dataset_2'
    output_file = 'eval_2/vovnet_incumbent_devel_dataset_2.txt'
    #validation(network_name, config, dataset_path, output_file)
    #RUNNING #DONE
    
    #3
    network_name = 'resnest'
    config = {"drop_rate": 0.0025640751507195886, "layers_I": 3, "layers_II": 4, "layers_III": 1, "layers_IV": 1,"pool_size": (7,7),"stem_width": 58} # "origin": "Local Search"
    dataset_path = '/work/dlclarge2/ozturk-nascomp_track_3/meta_dataset_finalized/eval_2_datasets/devel_dataset_0'
    output_file = 'eval_2/resnest_incumbent_devel_dataset_0.txt'
    #validation(network_name, config, dataset_path, output_file)
    #RUNNING #DONE
    
    #4
    network_name = 'vovnet'
    config = {"attn": "ese","depthwise": False,"drop_rate": 0.05800216665321023, "layer_per_block": 4,"pool_size": [2,2], "residual": False} # "origin": "Random Search"
    dataset_path = '/work/dlclarge2/ozturk-nascomp_track_3/meta_dataset_finalized/eval_2_datasets/devel_dataset_0'
    output_file = 'eval_2/vovnet_incumbent_devel_dataset_0.txt'
    #validation(network_name, config, dataset_path, output_file)
    #RUNNING #DONE
    
    #5
    network_name = 'resnest'
    config = {"drop_rate": 0.26837106794067533, "layers_I": 1, "layers_II": 1, "layers_III": 2, "layers_IV": 1,"pool_size": (5, 5),"stem_width": 25} # "origin": "Random Search (sorted)"
    dataset_path = '/work/dlclarge2/ozturk-nascomp_track_3/meta_dataset_finalized/eval_2_datasets/3-svhn_cropped_dataset'
    output_file = 'eval_2/resnest_incumbent_3-svhn_cropped.txt'
    #validation(network_name, config, dataset_path, output_file)
    #RUNNING #DONE
    
    #6
    network_name = 'vovnet'
    config = {"attn": "eca","depthwise": False,"drop_rate": 0.5596265812610938, "layer_per_block": 2,"pool_size": [9,9], "residual": False} # "origin": "Random Search (sorted)"
    dataset_path = '/work/dlclarge2/ozturk-nascomp_track_3/meta_dataset_finalized/eval_2_datasets/3-svhn_cropped_dataset'
    output_file = 'eval_2/vovnet_incumbent_3-svhn_cropped.txt'
    #validation(network_name, config, dataset_path, output_file)
    #RUNNING #DONE

    ##############
    ### EVAL 1 ###
    ##############
    #1
    network_name = 'densenet'
    config = {"aa_layer": False,"densenet_stem_type": "deep","drop_rate": 0.1782496728052363,"growth_rate": 40,"layers_I": 11,"layers_II": 18,"layers_III": 8,"layers_IV": 6,"pool_size": (5,5)} # "origin": "Local Search"
    dataset_path = '/work/dlclarge2/ozturk-nascomp_track_3/meta_dataset_finalized/eval_1_datasets/10-kmnist_dataset'
    output_file = 'eval_1/densenet_incumbent_10-kmnist_dataset.txt'
    validation(network_name, config, dataset_path, output_file)
    #RUNNING #DONE

    #2
    network_name = 'resnest'
    config = {"drop_rate": 0.0686727893944451, "layers_I": 5, "layers_II": 1, "layers_III": 2, "layers_IV": 1,"pool_size": [7,7],"stem_width": 58} # "origin": "Random Search (sorted)"
    dataset_path = '/work/dlclarge2/ozturk-nascomp_track_3/meta_dataset_finalized/eval_1_datasets/10-kmnist_dataset'
    output_file = 'eval_1/resnest_incumbent_10-kmnist_dataset.txt'
    validation(network_name, config, dataset_path, output_file)
    #RUNNING #DONE

    #3
    network_name = 'densenet'
    config = {"aa_layer": True,"densenet_stem_type": "deep","drop_rate": 0.02798004501286422,"growth_rate": 19,"layers_I": 12,"layers_II": 16,"layers_III": 40,"layers_IV": 41,"pool_size": (7,7)} # "origin": "Local Search"
    dataset_path = '/work/dlclarge2/ozturk-nascomp_track_3/meta_dataset_finalized/eval_1_datasets/11-mnist_dataset'
    output_file = 'eval_1/densenet_incumbent_11-mnist_dataset.txt'
    validation(network_name, config, dataset_path, output_file)
    #RUNNING #DONE

    #4
    network_name = 'resnest'
    config = {"drop_rate": 0.10117824729075853, "layers_I": 3, "layers_II": 3, "layers_III": 2, "layers_IV": 1,"pool_size": [7,7],"stem_width": 43} # "origin": "Random Search"
    dataset_path = '/work/dlclarge2/ozturk-nascomp_track_3/meta_dataset_finalized/eval_1_datasets/11-mnist_dataset'
    output_file = 'eval_1/resnest_incumbent_11-mnist_dataset.txt'
    validation(network_name, config, dataset_path, output_file)
    #RUNNING #DONE

    #5
    network_name = 'densenet'
    config = {"aa_layer": True, "densenet_stem_type": "deep", "drop_rate": 0.05277735095985825,"growth_rate": 18,"layers_I": 12,"layers_II": 18,"layers_III": 46,"layers_IV": 45,"pool_size": (2,2)} # "origin": "Random Search (sorted)"
    dataset_path = '/work/dlclarge2/ozturk-nascomp_track_3/meta_dataset_finalized/eval_1_datasets/27-emnist_balanced_dataset'
    output_file = 'eval_1/densenet_incumbent_27-emnist_balanced_dataset.txt'
    validation(network_name, config, dataset_path, output_file)
    #RUNNING #DONE

    #6
    network_name = 'resnest'
    config = {"drop_rate": 0.6214006503868467, "layers_I": 1, "layers_II": 1, "layers_III": 1, "layers_IV": 1,"pool_size": (5,5),"stem_width": 43} # "origin": "Local Search"
    dataset_path = '/work/dlclarge2/ozturk-nascomp_track_3/meta_dataset_finalized/eval_1_datasets/27-emnist_balanced_dataset'
    output_file = 'eval_1/resnest_incumbent_27-emnist_balanced_dataset.txt'
    validation(network_name, config, dataset_path, output_file)
    #RUNNING #DONE



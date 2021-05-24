import os
import sys
import json
from time import time

import numpy as np
from scipy import stats
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())
sys.path.append('unedited_sample_submission/')
from unedited_sample_submission import nas as nas_baseline

sys.path.append('dev_submission/')
from dev_submission import nas_helpers
from dev_submission.models import *

sys.path.append('ingestion_program/')
from ingestion_program.nascomp.helpers import get_dataset_paths, load_datasets
from ingestion_program.nascomp.torch_evaluator import torch_evaluator

MODELS = {
        "DenseNet161": densenet161,
        "ResNest14d": resnest14d,
        "MixNet_XXL": mixnet_xxl,
        "DenseNet121": densenet121,
        "DenseNetBlur121d": densenetblur121d,
        "VovNet39a": vovnet39a,
        "SeResNext26t_32x4d": seresnext26t_32x4d,
        "Gluon_xception65": gluon_xception65,
        "SeResNext26tn_32x4d": seresnext26tn_32x4d
    }

def main_performance(dataset_paths, output_dir, model_key = 'baseline', n_epochs = 64):

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # iterate through input datasets
    full_results = {}
    for dataset_path in dataset_paths:
        
        (train_x, train_y), (valid_x, valid_y), test_x, metadata = load_datasets(dataset_path)
        # package data for evaluator
        data = (train_x, train_y), (valid_x, valid_y), test_x

        if model_key == 'baseline':
            # for the baseline
            nas_algorithm = nas_baseline.NAS()
            model = nas_algorithm.search(train_x, train_y, valid_x, valid_y, metadata)
        else:
            model = MODELS[model_key]()
            model = nas_helpers.reshape_model(model=model, channels=train_x.shape[1], n_classes=int(metadata['n_classes']), copy_type = 'Starter')
        
        try:
            # train model for $n_epochs, recover test predictions from best validation epoch
            start = time()
            results = torch_evaluator(model, data, metadata, n_epochs=n_epochs, full_train=True)
            end = time()
            details = {k: v for k, v in results.items() if k!='test_predictions'}
            details['training_time'] = end-start

            predictions = results['test_predictions'] # get test predictions
            ref_y = np.load(os.path.join(dataset_path, 'test_y.npy')) # load the reference values
            score = sum(ref_y == predictions)/float(len(ref_y)) * 100 # calculate the score
            details['test_score'] = score # record to the dictionary
            np.save(os.path.join(output_dir, "predictions_{}.npy".format(metadata['name'])), predictions) # save these predictions to the output dir
        except:
            details = {'training_time': 0, 'valid_accuracies': [0]*n_epochs, 'best_val_score': 0, 'test_score': 0}

        full_results[metadata['name']] = details
        json.dump(full_results, open(os.path.join(output_dir, "full_results_"+metadata['name']+".json"),"w"))

    print("=== FINISHED EVALUATION ===")
    print(full_results)



if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser("PerformanceRun")

    parser.add_argument("--model_key", type=str, required = True) 
    parser.add_argument("--dataset_subgroup", type=str, required = True)
    parser.add_argument("--output_dir", type=str, default='dataset_portfolio')
    parser.add_argument("--dataset_path", type=str, default=None)

    args, _ = parser.parse_known_args()

    if args.dataset_path is None:
        input_dir = '/work/dlclarge2/ozturk-nascomp_track_3/meta_dataset_cosine_with_curation_no_similarity_eval_'+args.dataset_subgroup
        all_dataset_paths = get_dataset_paths(input_dir)
    else:
        all_dataset_paths = [args.dataset_path]

    overall_start = time()    
    model_key = args.model_key
    dataset_paths = all_dataset_paths
    print('Evaluating model {}'.format(model_key))
    output_dir = os.path.join(args.output_dir, str(args.dataset_subgroup), model_key)
    main_performance(dataset_paths, output_dir, model_key, 64)
    overall_end = time()
    print(overall_end-overall_start)  

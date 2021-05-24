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

from available_datasets import *
import meta_features_extractor as mfe

# FROM DOCS
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

# FROM DOCS
EVAL_SCORES = \
    {
    'eval_dataset_0':
        {
        "DenseNet161": 93.13,
        "ResNest14d": 91.18,
        "MixNet_XXL": 86.273,
        "DenseNet121": 0,
        "DenseNetBlur121d": 0,
        "VovNet39a": 91.93,
        "SeResNext26t_32x4d": 88.77,
        "Gluon_xception65": 59.64,
        "SeResNext26tn_32x4d": 76.0
        }, 
    'eval_dataset_1':
        {
        "DenseNet161": 39.746,
        "ResNest14d": 65.0,
        "MixNet_XXL": 24.82,
        "DenseNet121": 49.846,
        "DenseNetBlur121d": 39.32,
        "VovNet39a": 27.193,
        "SeResNext26t_32x4d": 24.786,
        "Gluon_xception65": 24.82,
        "SeResNext26tn_32x4d": 24.94
        },
    'eval_dataset_2':
        {
        "DenseNet161": 43.26,
        "ResNest14d": 43.52,
        "MixNet_XXL": 31.0,
        "DenseNet121": 41.65,
        "DenseNetBlur121d": 38.746,
        "VovNet39a": 43.567,
        "SeResNext26t_32x4d": 41.35,
        "Gluon_xception65": 0,
        "SeResNext26tn_32x4d": 0
        }
    }

EVAL_CURVES = \
    {
    'eval_dataset_0':[ "DenseNet161", "ResNest14d", "DenseNet121", "DenseNetBlur121d", "VovNet39a", "SeResNext26t_32x4d", "SeResNext26tn_32x4d", "Gluon_xception65"],
    'eval_dataset_1':[ "DenseNet161", "ResNest14d", "DenseNet121", "DenseNetBlur121d", "VovNet39a", "SeResNext26t_32x4d", "SeResNext26tn_32x4d", "Gluon_xception65"],
    'eval_dataset_2':[ "DenseNet161", "ResNest14d", "DenseNet121", "DenseNetBlur121d", "VovNet39a", "SeResNext26t_32x4d", "SeResNext26tn_32x4d"]
    }

def main_correlation(input_dir, output_dir, embeddings_dir = None):
    results_path = os.path.join(output_dir, 'full_results.json')
    full_results = json.load(open(results_path,"r"))

    valid_accuracies = []
    test_accuracies = []
    meta_features = []
    complexities = []
    corr_matrix = np.zeros([len(full_results.keys()), len(full_results.keys())])
    for i, (name_i, res_i) in enumerate(full_results.items()):
        valid_accuracies.append(res_i['valid_accuracies'])
        test_accuracies.append(res_i['test_score'])

        train_dataset, _, _ = mfe.load_dataset(os.path.join(input_dir, name_i))
        simple_mf = mfe.extract_simple_mf_from_dataset(train_dataset)
        if embeddings_dir is not None:
            task2vec_mf = mfe.get_embeddings_of_dataset(embeddings_dir, name_i)
        else:
            task2vec_mf = mfe.extract_embeddings_from_dataset(train_dataset)

        meta_features.append(simple_mf)
        complexities.append(np.linalg.norm(task2vec_mf, 1))

    labels = list(full_results.keys())
    task_complexities = complexities

    errors = [100-s for s in test_accuracies]
    avg_samples_per_class = [mf[0]/mf[1] for mf in meta_features]
    complexities = [c/spc for c, spc in zip(complexities, avg_samples_per_class)]
    
    normalized_test_accuracies = [s/max(test_accuracies) for s in test_accuracies]
    normalized_errors = [e/max(errors) for e in errors]
    normalized_complexities = [c/max(complexities) for c in complexities]
    normalized_avg_samples_per_class = [avg/max(avg_samples_per_class) for avg in avg_samples_per_class]
    
    plt.figure(figsize = (10, 10))
    for label, err, comp in zip(labels, normalized_errors, normalized_complexities):
        plt.scatter(comp, err, marker = 'o', label = label)
    plt.xlabel('Complexity')
    plt.ylabel('Error')
    plt.plot([0, 1])
    plt.savefig(os.path.join(output_dir, 'complexity_vs_performance.svg'))
    plt.close()

    plt.figure(figsize = (len(labels), 5))
    for i, (label, acc) in enumerate(zip(labels, test_accuracies)):
        plt.scatter(i, acc, marker = '*')
    plt.ylabel('Accuracy')
    plt.xticks(range(len(labels)), ['\n'.join(label.split('_')) for label in labels])
    plt.savefig(os.path.join(output_dir, 'test_scores.svg'))
    plt.close()

    plt.figure(figsize = (len(labels)//3, len(labels)//3))
    for label, accs in zip(labels, valid_accuracies):
        plt.plot(accs, label = label)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'curves.svg'))
    plt.close()

    from matplotlib import cm
    viridis = cm.get_cmap('autumn')

    plt.figure(figsize = (10, 10))
    for label, acc, simple_mf in zip(labels, test_accuracies, meta_features):
        num_samples, num_classes, num_channels, width, height = simple_mf
        plt.scatter(num_samples, num_classes, c = [viridis(acc/100)], marker = 'o', label = label)
    plt.xlabel('# of samples')
    plt.ylabel('# of classes')

    plt.savefig(os.path.join(output_dir, 'meta_features_sc.svg'))
    plt.close()

    plt.figure(figsize = (10, 10))
    for label, acc, simple_mf in zip(labels, test_accuracies, meta_features):
        num_samples, num_classes, num_channels, width, height = simple_mf
        plt.scatter(num_samples, width, c = [viridis(acc/100)], marker = 'o', label = label)
    plt.xlabel('# of samples')
    plt.ylabel('Resolution')

    plt.savefig(os.path.join(output_dir, 'meta_features_sr.svg'))
    plt.close()

    plt.figure(figsize = (10, 10))

    for label, acc, spc in zip(labels, normalized_test_accuracies, normalized_avg_samples_per_class):
        plt.scatter(spc, acc, marker = 'o', label = label)
    plt.xlabel('Avg # of samples per class')
    plt.ylabel('Accuracy')
    plt.plot([0, 1])
    plt.savefig(os.path.join(output_dir, 'spc_vs_performance.svg'))
    plt.close()

    print(output_dir)
    print('Task complexity vs test error correlation: {}, p-value: {}'.format(*stats.pearsonr(task_complexities, errors)))
    print('# samples vs test accuracy correlation: {}, p-value: {}'.format(*stats.pearsonr([f[0] for f in meta_features], test_accuracies)))
    print('# samples per class vs test accuracy correlation: {}, p-value: {}'.format(*stats.pearsonr(avg_samples_per_class, test_accuracies)))
    print('Complexity vs test error correlation: {}, p-value: {}'.format(*stats.pearsonr(complexities, errors)))

    f = open(os.path.join(output_dir, 'correlations.txt'), 'w+')
    f.write('Task complexity vs test error correlation: {}, p-value: {}'.format(*stats.pearsonr(task_complexities, errors))+'\n')
    f.write('# samples vs test accuracy correlation: {}, p-value: {}'.format(*stats.pearsonr([f[0] for f in meta_features], test_accuracies))+'\n')
    f.write('# samples per class vs test accuracy correlation: {}, p-value: {}'.format(*stats.pearsonr(avg_samples_per_class, test_accuracies))+'\n')
    f.write('Complexity vs test error correlation: {}, p-value: {}'.format(*stats.pearsonr(complexities, errors))+'\n')
    f.close()

def get_training_times(output_dir):
    results_path = os.path.join(output_dir, 'full_results.json')
    full_results = json.load(open(results_path,"r"))
    total_time = sum([res_i['training_time'] for _, res_i in full_results.items()])
    return total_time

def get_performances(output_dir):
    results_path = os.path.join(output_dir, 'full_results.json')
    full_results = json.load(open(results_path,"r"))

    valid_curves = {name_i: res_i['valid_accuracies'] for name_i, res_i in full_results.items()} # num_epochs x num_datasets
    valid_accuracies = {name_i: res_i['best_val_score'] if 'best_val_score' in res_i else 0 for name_i, res_i in full_results.items()} # 1 x num_datasets
    test_accuracies = {name_i: res_i['test_score'] for name_i, res_i in full_results.items()} # 1 x num_datasets
    
    return valid_curves, valid_accuracies, test_accuracies

def get_test_performances(output_dir, subgroup):
    results_path = os.path.join(output_dir, 'full_results_'+str(subgroup)+'.json')
    full_results = json.load(open(results_path,"r"))

    valid_curves = {name_i: res_i['valid_accuracies'] for name_i, res_i in full_results.items()} # num_epochs x num_datasets
    valid_accuracies = {name_i: res_i['best_val_score'] if 'best_val_score' in res_i else 0 for name_i, res_i in full_results.items()} # 1 x num_datasets
    test_accuracies = {name_i: res_i['test_score'] for name_i, res_i in full_results.items()} # 1 x num_datasets
    
    return valid_curves, valid_accuracies, test_accuracies


def get_defaults_from_baseline(output_main_dir):
    results_path = os.path.join(output_main_dir, 'baseline/full_results.json')
    full_results = json.load(open(results_path,"r"))
    datasets = list(full_results.keys())

    return datasets

def score_corr(performance_df, datasets, output_main_dir):
    corr_matrix = np.zeros([len(datasets), len(datasets)])
    for i, dataset_i in enumerate(datasets):
        test_scores_i = performance_df[dataset_i].values
        for j, dataset_j in enumerate(datasets):
            test_scores_j = performance_df[dataset_j].values
            corr_matrix[i][j] = stats.pearsonr(test_scores_i, test_scores_j)[0]

    figsize = max(0.31*len(datasets), 10)

    corr_df = pd.DataFrame(corr_matrix, index=datasets, columns=datasets)
    sns.clustermap(corr_df, cmap='viridis_r', figsize = (figsize, figsize))
    plt.savefig(os.path.join(output_main_dir, 'pearson_corr.svg'))

    return corr_df

def ranking_corr(performance_df, datasets, output_main_dir, method = 'spearman'):

    if method == 'spearman':
        # spearman
        corr_matrix = np.zeros([len(datasets), len(datasets)])
        for i, dataset_i in enumerate(datasets):
            test_scores_i = performance_df[dataset_i].values
            for j, dataset_j in enumerate(datasets):
                test_scores_j = performance_df[dataset_j].values
                corr_matrix[i][j] = stats.spearmanr(test_scores_i, test_scores_j)[0]

        figsize = max(0.31*len(datasets), 10)

        corr_df = pd.DataFrame(corr_matrix, index=datasets, columns=datasets)
        sns.clustermap(corr_df, cmap='viridis_r', figsize = (figsize, figsize))
        plt.savefig(os.path.join(output_main_dir, 'spearman_corr.svg'))

        return corr_df

    elif method == 'kendalltau':
        # kendall tau
        dataset_portfolio = {dataset: performance_df[dataset].sort_values(ascending=False).index.values for dataset in datasets}

        corr_matrix = np.zeros([len(datasets), len(datasets)])
        for i, dataset_i in enumerate(datasets):
            rankings_i = dataset_portfolio[dataset_i]
            for j, dataset_j in enumerate(datasets):
                rankings_j = dataset_portfolio[dataset_j]
                corr_matrix[i][j] = stats.kendalltau(rankings_i, rankings_j)[0]

        figsize = max(0.31*len(datasets), 10)

        corr_df = pd.DataFrame(corr_matrix, index=datasets, columns=datasets)
        sns.clustermap(corr_df, cmap='viridis_r', figsize = (figsize, figsize))
        plt.savefig(os.path.join(output_main_dir, 'kendalltau_corr.svg'))

        return corr_df

def main_dataset_correlation(subgroup, output_main_dir = 'dataset_portfolio'):
    test_dataset_key = 'eval_dataset_'+str(subgroup)

    ### At this point add hard-coded test dataset results ###
    
    # FROM DOCS
    #models = list(MODELS.keys())
    #test_dataset_scores = EVAL_SCORES[test_dataset_key] # FROM DOCS

    # FROM INGESTION OUTPUT
    models = EVAL_CURVES[test_dataset_key]
    _, test_dataset_scores, _ = get_test_performances(os.path.join(output_main_dir, 'test_results'), subgroup) 
    
    #########################################################

    datasets = get_defaults_from_baseline(output_main_dir)

    performance_matrix = np.zeros([len(models), len(datasets)+1])
    for i, model_key in enumerate(models):
        output_dir = os.path.join(output_main_dir, str(subgroup), model_key)
        _, valid_accuracies, _ = get_performances(output_dir)

        for j, dataset in enumerate(datasets):
            performance_matrix[i][j] = valid_accuracies[dataset]

        performance_matrix[i][-1] = test_dataset_scores[model_key]

    datasets = datasets+[test_dataset_key]

    performance_df = pd.DataFrame(performance_matrix, index=models, columns=datasets)

    score_corr_df = score_corr(performance_df, datasets, os.path.join(output_main_dir, str(subgroup))) 
    ranking_corr_df = ranking_corr(performance_df, datasets, os.path.join(output_main_dir, str(subgroup)))

    ### Pick 2 for each test dataset according to one of the correlation methods ###
    
    corr_result = score_corr_df[test_dataset_key]
    corr_result = corr_result.sort_values(ascending=False)
    print(corr_result[1:10])

    corr_result = ranking_corr_df[test_dataset_key]
    corr_result = corr_result.sort_values(ascending=False)
    print(corr_result[1:10])

    ################################################################################

def ranking_curve_corr(metadataset_curves, test_curve_matrix, datasets, axis = None):
    corr = []
    p_values = []
    for dataset in datasets:
        curve_matrix = metadataset_curves[dataset]

        if axis == 0:
            cs = []
            ps = []
            for i in range(test_curve_matrix.shape[0]):
                c, p = stats.spearmanr(test_curve_matrix[i,:], curve_matrix[i,:])
                cs.append(c)
                ps.append(p)
        elif axis == 1:
            cs = []
            ps = []
            for i in range(test_curve_matrix.shape[1]):
                c, p = stats.spearmanr(test_curve_matrix[:,i], curve_matrix[:,i])
                cs.append(c)
                ps.append(p)
        else:
            cs, ps = stats.spearmanr(test_curve_matrix, curve_matrix, axis = None)
        
        corr.append(cs)
        p_values.append(ps)

    corr_result = pd.Series(corr, index=datasets)
    confidence_result = pd.Series(p_values, index=datasets)

    return corr_result, confidence_result

def get_incumbent(_list):
    curr_incumbent = _list[0]
    incumbent_list = [curr_incumbent]
    for el in _list[1:]:
        if el > curr_incumbent:
            curr_incumbent = el
        incumbent_list.append(curr_incumbent)

    return incumbent_list

def main_dataset_curve_correlation(subgroup, output_main_dir = 'dataset_portfolio', curve_len = 16, n_picks = 10):
    test_dataset_key = 'eval_dataset_'+str(subgroup)

    ### At this point add hard-coded test dataset results ###
    
    models = EVAL_CURVES[test_dataset_key]
    test_dataset_curves, _, _ = get_test_performances(os.path.join(output_main_dir, 'test_results'), subgroup) 

    #########################################################

    datasets = get_defaults_from_baseline(output_main_dir)

    metadataset_curves = {}
    performance_matrix = np.zeros([len(models), len(datasets)])
    for j, dataset in enumerate(datasets):
        curve_matrix = np.zeros([len(models), curve_len]) # epoch x score
        for i, model_key in enumerate(models):
            output_dir = os.path.join(output_main_dir, str(subgroup), model_key)
            valid_curves, valid_accuracies, _ = get_performances(output_dir)
            curve = valid_curves[dataset]
            curve_matrix[i] = get_incumbent(curve[:16])
            
            performance_matrix[i][j] = valid_accuracies[dataset]
            
        metadataset_curves[dataset] = curve_matrix

    performance_df = pd.DataFrame(performance_matrix, index=models, columns=datasets)

    test_curve_matrix = np.zeros([len(models), curve_len])
    for j, model_key in enumerate(models):
        curve = test_dataset_curves[model_key]
        test_curve_matrix[j] = get_incumbent(curve[:16])

    corr_result, confidence_result = ranking_curve_corr(metadataset_curves, test_curve_matrix, datasets) # by both epoch and model
    corr_result_by_time, _ = ranking_curve_corr(metadataset_curves, test_curve_matrix, datasets, 0) # by epoch

    corr_result = corr_result.sort_values(ascending=False)
    picks = corr_result.index[:n_picks]

    corr_result = corr_result[picks]
    confidence_result = confidence_result[picks]
    corr_result_by_time = corr_result_by_time[picks]
    
    output_dir = os.path.join(output_main_dir, str(subgroup))
    f = open(os.path.join(output_dir, 'correlation_results.txt'), 'w+')
    f.write('Correlations:\n')
    f.write(str(corr_result))
    f.write('\n\np-values:\n')
    f.write(str(confidence_result))
    f.close()

    performance_df[picks].to_csv(os.path.join(output_dir, 'performances.csv'))

    plt.figure(figsize = (10, 10))
    for label, corr_by_time in zip(picks, corr_result_by_time):
        plt.plot(corr_by_time, label = label)

    plt.xlabel('Epochs')
    plt.ylabel('Correlation Coef')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'correlation_by_epochs.svg'))

def perf_args_by_dataset():
    f = open('dev_meta_features/performance_run_by_dataset.args', 'w+')
    for i in [2, 1, 0]:
        input_dir = '/work/dlclarge2/ozturk-nascomp_track_3/meta_dataset_cosine_with_curation_no_similarity_eval_'+str(i)
        all_dataset_paths = get_dataset_paths(input_dir)
        for dpath in all_dataset_paths:
            for model_key in ['MixNet_XXL', 'DenseNet161', 'Gluon_xception65']:
                f.write(' '.join(['--model_key', str(model_key), '--dataset_subgroup', str(i), '--dataset_path', dpath]))
                f.write('\n')
    f.close()

def merge_performance_run_results(models = ["DenseNet161", "MixNet_XXL", "Gluon_xception65"], subgroups = [1, 2]):
    
    for model_key in models:
        for dataset_subgroup in subgroups:
            output_dir = os.path.join("dataset_portfolio", str(dataset_subgroup), model_key)
            result_files = [p for p in os.listdir(output_dir) if 'full_results_' in p]
            print(len(result_files))

            full_results = {}
            for f in result_files:
                results_path = os.path.join(output_dir, f)
                res = json.load(open(results_path,"r"))
                for name, res in res.items():
                    full_results[name] = res

            json.dump(full_results, open(os.path.join(output_dir, "full_results.json"),"w"))

            print("=== FINISHED EVALUATION ===")
            print(full_results)

def get_test_val_curves(subgroup, output_main_dir = 'dataset_portfolio', curve_len = 16):
    _dir = os.path.join(output_main_dir, 'test_results')
    _file = os.path.join(_dir, 'results_accumulated_'+str(subgroup)+'.txt')

    full_results = {k: {'training_time': 0, 'valid_accuracies': [0]*curve_len, 'best_val_score': 0, 'test_score': 0} for k in MODELS.keys()}

    with open(_file, 'r+') as f:
        results = f.read()

    results = results.split('training')[1:] # first row is empty
    for r in results:
        r = r.split('\n')
        
        model_key = r[0].strip()
        epochs = r[1:-1]
        valid_accuracies = []
        for e in epochs:
            e = e.strip().split()
            valid_accuracies.append(float(e[3]))

        while len(valid_accuracies) < curve_len:
            valid_accuracies.append(valid_accuracies[-1])
        
        full_results[model_key]['valid_accuracies'] = valid_accuracies
        full_results[model_key]['best_val_score'] = max(valid_accuracies)

    print(full_results)
    json.dump(full_results, open(os.path.join(_dir, "full_results_"+str(subgroup)+".json"),"w"))


if __name__ == "__main__":

    #perf_args_by_dataset()

    #embeddings_dir = "dev_meta_features/task2vec_results/meta_dataset_creation_resnet34/"
    #input_dir = '/work/dlclarge2/ozturk-nascomp_track_3/meta_dataset_cosine_with_curation_no_similarity_eval_2'

    #models = ["DenseNet121"]
    #subgroups = [1]
    #merge_performance_run_results(models, subgroups)

    #main_dataset_correlation(2, 'dataset_portfolio')
    #main_dataset_correlation(1, 'dataset_portfolio')
    
    main_dataset_curve_correlation(2, 'dataset_portfolio')
    main_dataset_curve_correlation(1, 'dataset_portfolio')
import os
import pickle
import json
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.append('dev_meta_features/task2vec/')
from task2vec import Task2Vec
from models import get_model
import task_similarity

from available_datasets import all_datasets, nascomp_dev, AUTODL_MAIN_DIR
import mf_utils as utils

def load_embeddings(output_dir):
    embeddings = pickle.load(open(os.path.join(output_dir, "embeddings.pkl"), "rb"))
    task_names = pickle.load(open(os.path.join(output_dir, "task_names.pkl"), "rb"))
    
    return embeddings, np.array(task_names)


def get_meta_features(embeddings, task_names):
    meta_features = dict()
    for i, (t, e) in enumerate(zip(task_names, embeddings)): 
        meta_features[t] = e.hessian

    return meta_features

def get_embeddings_dict(embeddings, task_names):
    embeddings_dict = dict()
    for (t, e) in zip(task_names, embeddings): 
        embeddings_dict[t] = e

    return embeddings_dict

def pdist_nascomp2metadataset(nascomp_embeddings, metadataset_embeddings, distance = 'cosine'):
    
    distance_matrix = task_similarity.cdist(nascomp_embeddings, metadataset_embeddings, distance=distance)
    devel_dataset_task_similarities = {t: distance_matrix[i] for i, t in enumerate(nascomp_dev)}
    
    return devel_dataset_task_similarities

    
def pick_from_metadataset(devel_dataset_task_similarities, metadataset_task_names, n = 10):

    def curate(labels, train_size_range = (1e4, 1e5), test_size_min = 1e3, max_resolution = 128, max_num_classes = 40):
        removed = []
        simple_mf_dict = dict()
        for name in labels:
            dataset_dir = os.path.join(AUTODL_MAIN_DIR, *name.split('-'))
            train_dataset, test_dataset, _ = utils.get_autodl_dataset(dataset_dir)
            
            train_size = train_dataset.get_metadata().size()
            test_size = test_dataset.get_metadata().size()
            _, h, w, _ = train_dataset.get_metadata().get_tensor_shape()
            num_classes = train_dataset.get_metadata().get_output_size()

            h = 224 if (h > 224 or h == -1) else h
            w = 224 if (w > 224 or w == -1) else w

            if (train_size_range[0] < train_size < train_size_range[1]) \
                and test_size > test_size_min \
                and h <= max_resolution and w <= max_resolution\
                and num_classes <= max_num_classes:
                r = False
            else:
                r = True
            
            removed.append(r)
            simple_mf_dict[name] = [train_size, test_size, num_classes, (h, w), r]
        
        return removed, simple_mf_dict
        

    devel_dataset_metadataset_picks = {t: [] for t in nascomp_dev+['portfolio']}
    removed, simple_mf_dict = curate(list(metadataset_task_names))
    for t, distances in devel_dataset_task_similarities.items():
        sorted_idx = np.argsort(distances)
        sorted_metadatasets = list(metadataset_task_names[i] for i in sorted_idx if not removed[i])
        print(len(sorted_metadatasets))
        if n == 'all':
            devel_dataset_metadataset_picks[t] = sorted_metadatasets
        else:
            devel_dataset_metadataset_picks[t] = sorted_metadatasets[:n]

    portfolio = []
    for v in devel_dataset_metadataset_picks.values():
        portfolio += v
    
    devel_dataset_metadataset_picks['portfolio'] = list(set(portfolio))


    
    return devel_dataset_metadataset_picks, simple_mf_dict

def save_analysis_result(devel_dataset_metadataset_picks, output_dir):
    
    result_dir = os.path.join(output_dir, "metadataset_info")
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)
    
    f = open(os.path.join(result_dir, "info.txt"), "w+")
    for t, picks in devel_dataset_metadataset_picks.items():
        f.write("Picks for {} ({}):\n".format(t, len(picks)))
        f.write("\n".join(picks))
        f.write("\n\n")
    f.close()

    json.dump(devel_dataset_metadataset_picks, open(os.path.join(result_dir, "info.json"),"w"))
        

def plot_distance_matrices(embeddings_dict, devel_dataset_metadataset_picks, distance_plots_path):
    
    for t in nascomp_dev:
        tasks = devel_dataset_metadataset_picks[t]
        tasks.append(t)
        task_specific_embeddings = [embeddings_dict[t] for t in tasks]
        task_similarity.plot_distance_matrix(task_specific_embeddings, tasks, savepath = os.path.join(distance_plots_path, t+"_picks_dist_mat.svg"))

    all_tasks = devel_dataset_metadataset_picks['portfolio']+nascomp_dev
    all_embeddings = [embeddings_dict[t] for t in all_tasks]
    task_similarity.plot_distance_matrix(all_embeddings, all_tasks, savepath = os.path.join(distance_plots_path, "all_picks_dist_mat.svg"))

    return all_embeddings

def plot_complexity_histograms(meta_features_dict, devel_dataset_metadataset_picks, complexity_plots_path):
    
    for t in nascomp_dev:
        tasks = devel_dataset_metadataset_picks[t]
        tasks.append(t)
        task_specific_meta_features = [meta_features_dict[t] for t in tasks]
        task_complexities = [np.linalg.norm(v, 1) for v in task_specific_meta_features]
        
        plt.figure(figsize = (10, 10))
        n_bins = max(int(np.sqrt(len(task_complexities))), 5)
        plt.hist(task_complexities, n_bins)
        plt.axvline(x = task_complexities[-1], color = 'r', linestyle = '-')
        plt.xticks([task_complexities[-1]], [t])
        plt.ylabel('# of datasets')
        plt.xlabel('Task complexity (1-norm of embeddings)')
        plt.savefig(os.path.join(complexity_plots_path, t+"_picks_complexity_hist.svg"))
        plt.close()

    all_tasks = devel_dataset_metadataset_picks['portfolio']+nascomp_dev
    all_meta_features = [meta_features_dict[t] for t in all_tasks]
    task_complexities = [np.linalg.norm(v, 1) for v in all_meta_features]

    plt.figure(figsize = (10, 10))
    n_bins = max(int(np.sqrt(len(task_complexities))), 5)
    plt.hist(task_complexities, n_bins)
    for c in task_complexities[-len(nascomp_dev):]:
        plt.axvline(x = c, color = 'r', linestyle = '-')
    plt.xticks(task_complexities[-len(nascomp_dev):], ['dev_0', 'dev_1', 'dev_2'])
    plt.ylabel('# of datasets')
    plt.xlabel('Task complexity (1-norm of embeddings)')
    plt.savefig(os.path.join(complexity_plots_path, "all_picks_complexity_hist.svg"))
    plt.close()

    return task_complexities


def analyze_test_scores(filepath, datasets_dir, meta_features_dict):
    results = json.load(open(filepath,"r"))

    result_dict = dict()
    for name, score in results.items():
        (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = utils.get_nascomp_dataset(os.path.join(datasets_dir, name))
        
        if name[-8:] == '_dataset':
            name = name[:-8]

        meta_features = meta_features_dict[name]
        task_complexity = np.linalg.norm(meta_features, 1)
        print(name, score)
        print(train_x.shape, valid_x.shape, test_x.shape)
        
        result_dict[name] = {
            'score': score,
            'simple_mf':{

                'train_size': train_x.shape[0],
                'valid_size': valid_x.shape[0],
                'test_size': test_x.shape[0],
                'channels': test_x.shape[1],
                'res_w': test_x.shape[2],
                'res_h': test_x.shape[3]
            },
            'complexity': task_complexity
            }

    return result_dict

def plot_simple_mf_analysis(portfolio, simple_mf_dict, simple_mf_plots_path):
    num_samples = []
    num_classes = []
    resolution = []
    avg_samples_per_class = []
    picked_num_samples = []
    picked_num_classes = []
    picked_resolution = []
    picked_avg_samples_per_class = []
    for name, simple_mf in simple_mf_dict.items():
        train_size, test_size, classes, (h, w), removed = simple_mf
        
        if removed:
            continue

        if name in portfolio:
            picked_num_samples.append(train_size)
            picked_num_classes.append(classes)
            picked_resolution.append((h+w)/2)
            picked_avg_samples_per_class.append(train_size/classes)
        else:
            num_samples.append(train_size)
            num_classes.append(classes)
            resolution.append((h+w)/2)
            avg_samples_per_class.append(train_size/classes)


    plt.figure(figsize = (10, 10))
    n_bins = 10
    plt.hist(num_samples, n_bins, label= 'non_picks', rwidth = 0.8, alpha = 0.5)
    plt.hist(picked_num_samples, n_bins, label= 'sim_picks', rwidth = 0.8, alpha = 0.5)
    plt.ylabel('# of datasets')
    plt.xlabel('Num samples')
    plt.legend()
    plt.savefig(os.path.join(simple_mf_plots_path, "num_samples.svg"))
    plt.close()

    plt.figure(figsize = (10, 10))
    n_bins = 10
    plt.hist(num_classes, n_bins, label= 'non_picks', rwidth = 0.8, alpha = 0.5)
    plt.hist(picked_num_classes, n_bins, label= 'sim_picks', rwidth = 0.8, alpha = 0.5)
    plt.ylabel('# of datasets')
    plt.xlabel('Num classes')
    plt.legend()
    plt.savefig(os.path.join(simple_mf_plots_path, "num_classes.svg"))
    plt.close()

    plt.figure(figsize = (10, 10))
    n_bins = 10
    plt.hist(resolution, n_bins, label= 'non_picks', rwidth = 0.8, alpha = 0.5)
    plt.hist(picked_resolution, n_bins, label= 'sim_picks', rwidth = 0.8, alpha = 0.5)
    plt.ylabel('# of datasets')
    plt.xlabel('Resolution')
    plt.legend()
    plt.savefig(os.path.join(simple_mf_plots_path, "resolution.svg"))
    plt.close()

    plt.figure(figsize = (10, 10))
    n_bins = 10
    plt.hist(avg_samples_per_class, n_bins, label= 'non_picks', rwidth = 0.8, alpha = 0.5)
    plt.hist(picked_avg_samples_per_class, n_bins, label= 'sim_picks', rwidth = 0.8, alpha = 0.5)
    plt.ylabel('# of datasets')
    plt.xlabel('Avg samples per class')
    plt.legend()
    plt.savefig(os.path.join(simple_mf_plots_path, "avg_samples_per_class.svg"))
    plt.close()


def meta_dataset_portfolio_creation(input_dir, distance_fns, n = 10, exp_suffix = ''):
    print('Analyzing dir {}'.format(input_dir))
    for distance_fn in distance_fns:
        output_dir = os.path.join(input_dir, distance_fn+'_'+exp_suffix)
        plots_path = os.path.join(output_dir, 'plots')
        complexity_plots_path = os.path.join(plots_path, 'complexities')
        distance_plots_path = os.path.join(plots_path, 'distances')
        simple_mf_plots_path = os.path.join(plots_path, 'metafeatures')

        if not os.path.isdir(distance_plots_path):
            os.makedirs(distance_plots_path)
        
        if not os.path.isdir(complexity_plots_path):
            os.makedirs(complexity_plots_path)

        if not os.path.isdir(simple_mf_plots_path):
            os.makedirs(simple_mf_plots_path)

        embeddings, task_names = load_embeddings(input_dir)
        embeddings_dict = get_embeddings_dict(embeddings, task_names)
        meta_features_dict = get_meta_features(embeddings, task_names)

        all_meta_features = [meta_features_dict[t] for t in task_names]
        task_complexities = [np.linalg.norm(v, 1) for v in all_meta_features]

        plt.figure(figsize = (20, 20))
        n_bins = max(int(np.sqrt(len(task_complexities))), 5)
        plt.hist(task_complexities, n_bins)
        for c in task_complexities[-len(nascomp_dev):]:
            plt.axvline(x = c, color = 'r', linestyle = '-')
           
        plt.xticks(task_complexities[-len(nascomp_dev):], ['dev_0', 'dev_1', 'dev_2'])
        plt.ylabel('# of datasets')
        plt.xlabel('Task complexity (1-norm of embeddings)')
        plt.savefig(os.path.join(plots_path, "complexity_hist.svg"))
        plt.close()

        nascomp_embeddings = embeddings[-len(nascomp_dev):]
        task_similarity.plot_distance_matrix(nascomp_embeddings, nascomp_dev, distance='cosine', savepath = os.path.join(plots_path, "devel_cosine_distance_mat.svg"))
        
        metadataset_embeddings = embeddings[:-len(nascomp_dev)]
        metadataset_task_names = task_names[:-len(nascomp_dev)]

        devel_dataset_task_similarities = pdist_nascomp2metadataset(nascomp_embeddings, metadataset_embeddings, distance_fn)
        devel_dataset_metadataset_picks, simple_mf_dict = pick_from_metadataset(devel_dataset_task_similarities, metadataset_task_names, n)
        print("Portfolio length for {} distance is {}".format(distance_fn, len(devel_dataset_metadataset_picks['portfolio'])))
        
        save_analysis_result(devel_dataset_metadataset_picks, output_dir)
        all_embeddings = plot_distance_matrices(embeddings_dict, devel_dataset_metadataset_picks, distance_plots_path)
        task_complexities = plot_complexity_histograms(meta_features_dict, devel_dataset_metadataset_picks, complexity_plots_path)
        plot_simple_mf_analysis(devel_dataset_metadataset_picks['portfolio'], simple_mf_dict, simple_mf_plots_path)

if __name__ == "__main__":
    
    input_dir = "dev_meta_features/task2vec_results/meta_dataset_creation_resnet34/"
    distance_fns = ['cosine']
    n = 'all'
    meta_dataset_portfolio_creation(input_dir, distance_fns, n, 'with_curation_no_similarity')
    
    '''
    embeddings, task_names = load_embeddings(input_dir)
    embeddings_dict = get_embeddings_dict(embeddings, task_names)
    meta_features_dict = get_meta_features(embeddings, task_names)

    datasets_dir = "/work/dlclarge2/ozturk-nascomp_track_3/meta_dataset_cosine"
    result_dict = analyze_test_scores(os.path.join(input_dir, "cosine/metadataset_info/scorescosine_baseline.json"), datasets_dir, meta_features_dict)
    curate_from_results(result_dict, output_dir = os.path.join(input_dir, "cosine/metadataset_info"))
    '''
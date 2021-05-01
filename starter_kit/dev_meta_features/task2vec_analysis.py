import os
import pickle
import json
import sys

import numpy as np

sys.path.append('task2vec/')
from task2vec import Task2Vec
from models import get_model
import task_similarity

from available_datasets import all_datasets, nascomp_dev

def load_embeddings(output_dir):
    embeddings = pickle.load(open(os.path.join(output_dir, "embeddings.pkl"), "rb"))
    task_names = pickle.load(open(os.path.join(output_dir, "task_names.pkl"), "rb"))
    
    return embeddings, np.array(task_names)


def get_meta_features(embeddings, task_names):
    meta_features = dict()
    for i, (t, e) in enumerate(zip(task_names, embeddings)): 
        meta_features[t] = {i: e.hessian[i] for i in range(len(e.hessian))}

    return meta_features


def pdist_nascomp2metadataset(nascomp_embeddings, metadataset_embeddings):
    
    distance_matrix = task_similarity.cdist(nascomp_embeddings, metadataset_embeddings, distance="cosine")
    devel_dataset_task_similarities = {t: distance_matrix[i] for i, t in enumerate(nascomp_dev)}
    
    return devel_dataset_task_similarities

    
def pick_from_metadataset(devel_dataset_task_similarities, n = 100):

    devel_dataset_metadataset_picks = {t: [] for t in nascomp_dev+['portfolio']}

    for t, distances in devel_dataset_task_similarities.items():
        sorted_idx = np.argsort(distances)
        sorted_metadatasets = metadataset_task_names[sorted_idx]
        devel_dataset_metadataset_picks[t] = list(sorted_metadatasets[:n])

    portfolio = []
    for v in devel_dataset_metadataset_picks.values():
        portfolio += v
    
    devel_dataset_metadataset_picks['portfolio'] = list(set(portfolio))

    print("Portfolio length is {}".format(len(devel_dataset_metadataset_picks['portfolio'])))
    
    return devel_dataset_metadataset_picks

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
        

def plot_distance_matrices(embeddings, task_names, devel_dataset_metadataset_picks, output_dir):
    
    for t in nascomp_dev:
        picks = devel_dataset_metadataset_picks[t]
        
        labels = [t]
        t_idx = np.where(task_names==t)[0][0]
        task_specific_embeddings = [embeddings[t_idx]]
        
        for p in picks:
            labels.append(p)
            p_idx = np.where(task_names==p)[0][0]
            task_specific_embeddings.append(embeddings[p_idx])

        task_similarity.plot_distance_matrix(task_specific_embeddings, labels, savepath = os.path.join(output_dir, t+"_picks_dist_mat.svg"))

    all_labels = devel_dataset_metadataset_picks['portfolio']+nascomp_dev
    all_embeddings = []
    for t in all_labels:
        t_idx = np.where(task_names==t)[0][0]
        all_embeddings.append(embeddings[t_idx])

    task_similarity.plot_distance_matrix(all_embeddings, all_labels, savepath = os.path.join(output_dir, "all_picks_dist_mat.svg"))


if __name__ == "__main__":
    output_dir = "task2vec_results/meta_dataset_creation/"

    embeddings, task_names = load_embeddings(output_dir)

    nascomp_embeddings = embeddings[-len(nascomp_dev):]
    task_similarity.plot_distance_matrix(nascomp_embeddings, nascomp_dev, savepath = os.path.join(output_dir, "devel_dist_mat.svg"))

    metadataset_embeddings = embeddings[:-len(nascomp_dev)]
    metadataset_task_names = task_names[:-len(nascomp_dev)]

    devel_dataset_task_similarities = pdist_nascomp2metadataset(nascomp_embeddings, metadataset_embeddings)
    devel_dataset_metadataset_picks = pick_from_metadataset(devel_dataset_task_similarities, 100)
    plot_distance_matrices(embeddings, task_names, devel_dataset_metadataset_picks, output_dir)
    save_analysis_result(devel_dataset_metadataset_picks, output_dir)

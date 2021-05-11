import argparse
import os
import pickle
from path import Path
import numpy as np
import json
import sys
import operator
import matplotlib.pyplot as plt

from utils import get_dataset_paths

parser = argparse.ArgumentParser(description='Train models.')
parser.add_argument('--data', type=str, help='Dataset name or path to datasets, if model is not trained on all datasets')
parser.add_argument('--save_dir', type=str, help='Dataset name or path to datasets, if model is not trained on all datasets')
parser.add_argument('-N', type=int, default=10, help='Get Top10 results based on score')
args = parser.parse_args()

sys.path.append('ingestion_program')
from nascomp.helpers import load_datasets

performance_dict = {}
for i in range(3):
    all_scores = {}
    for dataset_path in get_dataset_paths(args.data):
        print(20*'=')
        print(dataset_path)
        if dataset_path not in performance_dict.keys():
            performance_dict[dataset_path] = {}

        with open(os.path.join(dataset_path, 'dataset_metadata'), "r") as f:
            metadata = json.load(f)
        ref_y = np.load(os.path.join(dataset_path, 'test_y.npy'))
        
        
        score_dict = {}

        save_dir = os.path.join(args.save_dir, str(i), dataset_path[dataset_path.rfind('/')+1:])
        for key in os.listdir(save_dir):
            save_path = os.path.join(
                        save_dir,
                        f'{key}'
                    )

            with open(save_path, 'rb') as f:
                data = pickle.load(f)

            pred_y = data['test_predictions']
            acc = sum(ref_y == pred_y)/float(len(ref_y)) * 100
            score = sum(ref_y == pred_y)/float(len(ref_y)) * 100
            point_weighting = 10/(100 - metadata['benchmark'])
            score -= metadata['benchmark']
            score *= point_weighting
            k = key[:key.rfind('.')]
            score_dict[k] = score
            all_scores[k] = score + all_scores[k] if k in all_scores.keys() else score
            #print(k, score, v['best_val_score'], acc)
            if k not in performance_dict[dataset_path].keys():
                performance_dict[dataset_path][k] = []
            performance_dict[dataset_path][k].append(score)

        print(dict(sorted(score_dict.items(), key =operator.itemgetter(1), reverse = True)[:args.N]))

    print(20*'=')
    print('Overall')
    print(dict(sorted(all_scores.items(), key =operator.itemgetter(1), reverse = True)[:args.N]))
    print(20*'=')

print(20*'=')
# filter out data with only 1 value
performance_dict = {
    outer_k: {
        inner_k: inner_v
        for inner_k, inner_v in outer_v.items() if len(inner_v) > 1
    }
    for outer_k, outer_v in performance_dict.items()
}
keys = set()
for k,v in performance_dict.items():
    data = [val for _,val in v.items()]
    plt.boxplot(data, showmeans=True)
    plt.xticks([i+1 for i in range(len(v.keys()))], v.keys(), rotation='vertical')
    plt.tight_layout()
    plt.savefig(f'{k}.png')
    plt.close()

    for func in [np.min, np.mean, np.median, np.max]:
        vf = {_k:round(func(np.array(_v)), 4) for _k,_v in v.items()}
        print(k[k.rfind('/')+1:], func.__name__, dict(sorted(vf.items(), key =operator.itemgetter(1), reverse = True)[:args.N]))
    print(20*'=')
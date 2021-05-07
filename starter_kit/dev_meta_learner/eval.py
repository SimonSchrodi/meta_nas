import argparse
import os
import pickle
from path import Path
import numpy as np
import json
import sys
import operator

from utils import get_dataset_paths

parser = argparse.ArgumentParser(description='Train models.')
parser.add_argument('--data', type=str, help='Dataset name or path to datasets, if model is not trained on all datasets')
parser.add_argument('--save_dir', type=str, help='Dataset name or path to datasets, if model is not trained on all datasets')
parser.add_argument('-N', type=int, default=10, help='Get Top10 results based on score')
args = parser.parse_args()

sys.path.append('ingestion_program')
from nascomp.helpers import load_datasets

all_scores = {}
for dataset_path in get_dataset_paths(args.data):
    print(20*'=')
    print(dataset_path)
    with open(os.path.join(dataset_path, 'dataset_metadata'), "r") as f:
        metadata = json.load(f)
    ref_y = np.load(os.path.join(dataset_path, 'test_y.npy'))
    performance_dict = {}

    #print(os.listdir(args.save_dir))
    save_dir = os.path.join(args.save_dir, dataset_path[dataset_path.rfind('/')+1:])
    for key in os.listdir(save_dir):
        save_path = os.path.join(
                    save_dir,
                    f'{key}'
                )

        with open(save_path, 'rb') as f:
            data = pickle.load(f)

        performance_dict[key[:key.rfind('.')]] = data

    score_dict = {}
    for k, v in performance_dict.items():
        pred_y = v['test_predictions']
        acc = sum(ref_y == pred_y)/float(len(ref_y)) * 100
        score = sum(ref_y == pred_y)/float(len(ref_y)) * 100
        point_weighting = 10/(100 - metadata['benchmark'])
        score -= metadata['benchmark']
        score *= point_weighting
        score_dict[k] = score
        all_scores[k] = score + all_scores[k] if k in all_scores.keys() else score
        #print(k, score, v['best_val_score'], acc)

    print(dict(sorted(score_dict.items(), key =operator.itemgetter(1), reverse = True)[:args.N]))

print(20*'=')
print('Overall')
print(dict(sorted(all_scores.items(), key =operator.itemgetter(1), reverse = True)[:args.N]))
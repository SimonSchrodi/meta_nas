import argparse
import os
import pickle

from train import torch_evaluator

import sys
sys.path.append('dev_submission')
import nas_helpers
from models import *

sys.path.append('ingestion_program')
from nascomp.helpers import load_datasets

parser = argparse.ArgumentParser(description='Train models.')
parser.add_argument('--model', type=str, help='Model to train', required=True)
parser.add_argument('--data', type=str, help='Dataset name or path to datasets, if model is not trained on all datasets')
parser.add_argument('--save_dir', type=str, default=os.getcwd(), help='Dir where to save files')

def get_dataset_paths(data):
    def check_valid_dataset(dataset_path):
        needed_files = [
            'train_x.npy',
            'train_y.npy',
            'valid_x.npy',
            'valid_y.npy',
            'test_x.npy',
            'test_y.npy',
            'dataset_metadata'
        ]
        listdir = os.listdir(dataset_path)
        if not set(listdir) == set(needed_files):
            print(f'WARNING: Dataset path {dataset_path} is not valid!')
            return False
        return True

    if not os.path.isdir(data):
        raise Exception('No directoy!')
    
    listdir = os.listdir(data)
    if any([".npy" in d for d in listdir]):
        listdir = [data]
    else:
        listdir = [os.path.join(data, d) for d in listdir if os.path.isdir(os.path.join(data, d))]
    
    return [dataset_path for dataset_path in listdir if check_valid_dataset(dataset_path)]

def main():
    args = parser.parse_args()
    model = args.model

    assert os.path.isdir(args.save_dir)
    # get model
    assert args.model in model_portfolio
    model = model_portfolio[args.model]()
    # train model and save results to disc
    for dataset_path in get_dataset_paths(args.data):
        (train_x, train_y), (valid_x, valid_y), test_x, metadata = load_datasets(dataset_path)
        data = (train_x, train_y), (valid_x, valid_y), test_x
        model_specific = nas_helpers.reshape_model(model=model, channels=train_x.shape[1], n_classes=metadata['n_classes'])
        results = torch_evaluator(model_specific, data, metadata, n_epochs=64, full_train=True)
        
        save_path = os.path.join(
            args.save_dir,
            f'{dataset_path[dataset_path.rfind("/")+1:]}/{args.model}.pickle'
        )
        with open(save_path, 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()
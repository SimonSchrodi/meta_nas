import numpy as np
import os
import json

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
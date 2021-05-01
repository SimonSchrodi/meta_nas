import numpy as np
import os
import json

# === DATA LOADING HELPERS ====================================================
def get_dataset_paths(data_dir):
    paths = sorted([os.path.join(data_dir, d) for d in os.listdir(data_dir) if 'dataset' in d])
    return paths


def load_dataset_metadata(dataset_path):
    with open(os.path.join(dataset_path, 'dataset_metadata'), "r") as f:
        metadata = json.load(f)
    return metadata


# load dataset from location data/$dataset/
def load_datasets(data_path):
    train_x = np.load(os.path.join(data_path,'train_x.npy'))
    train_y = np.load(os.path.join(data_path,'train_y.npy'))
    valid_x = np.load(os.path.join(data_path,'valid_x.npy'))
    valid_y = np.load(os.path.join(data_path,'valid_y.npy'))
    test_x = np.load(os.path.join(data_path,'test_x.npy'))
    metadata = load_dataset_metadata(data_path)

    return (train_x, train_y), \
           (valid_x, valid_y), \
           (test_x), metadata
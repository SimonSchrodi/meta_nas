import os
import pickle
import sys
import numpy as np

sys.path.append(os.getcwd())
from available_datasets import all_datasets, GROUPS
from mf_utils import get_nascomp_dataset, nascomp_to_torch, logger

def load_dataset(_dir):
    train_dataset, valid_dataset, test_dataset = get_nascomp_dataset(_dir)
    return train_dataset, valid_dataset, test_dataset

def extract_embeddings_from_dataset(dataset):

    dataset, num_classes = nascomp_to_torch(dataset)

    sys.path.append('dev_meta_features/task2vec/')
    from task2vec import Task2Vec
    from models import get_model
    
    logger.info("Embedding")
    probe_network = get_model('resnet34', pretrained=True, num_classes=num_classes).cuda()
    task2vec = Task2Vec(probe_network, max_samples=10000, skip_layers=0, method = 'montecarlo', loader_opts = {'batch_size': 100})
    embedding, _ = task2vec.embed(dataset)
    logger.info("Embedding completed!")

    return embedding.hessian

def get_embeddings_of_dataset(embeddings_dir, _dir):
    import task2vec_analysis as t2v_a

    if _dir[-8:] == '_dataset':
        _dir = _dir[:-8]

    embeddings, task_names = t2v_a.load_embeddings(embeddings_dir)
    embeddings_dict = t2v_a.get_embeddings_dict(embeddings, task_names)

    return embeddings_dict[_dir].hessian

def extract_simple_mf_from_dataset(dataset):

    X, y = dataset

    num_samples, num_channels, width, height = X.shape
    num_classes = len(np.unique(y))

    return num_samples, num_classes, num_channels, width, height

def precompute_embeddings():
    pass

def precompute_simple_mf():
    pass

if __name__ == "__main__":

    metadataset_path = '/work/dlclarge2/ozturk-nascomp_track_3/meta_dataset_cosine_with_curation'
    dataset_dirs = os.listdir(metadataset_path)
    embeddings_dir = "dev_meta_features/task2vec_results/meta_dataset_creation_resnet34/"
    for _dir in dataset_dirs:
        train_dataset, _, _ = load_dataset(os.path.join(metadataset_path, _dir))
        task2vec_mf = extract_embeddings_from_dataset(train_dataset)
        task2vec_mf = find_embeddings_of_dataset(embeddings_dir, _dir)
        simple_mf = extract_simple_mf_from_dataset(train_dataset)

        print(len(task2vec_mf))
        print(simple_mf)
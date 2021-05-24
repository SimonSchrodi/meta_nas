import os
import pickle
import sys
from pathlib import Path

sys.path.append(os.getcwd())
from available_datasets import all_datasets, GROUPS
from mf_utils import get_autodl_dataset, autodl_to_torch, get_nascomp_dataset, nascomp_to_torch, logger, dump_meta_features_df_and_csv

sys.path.append('dev_meta_features/task2vec/')
from task2vec import Task2Vec
from models import get_model
import task_similarity

devel_datasets_dir = 'public_data_12-03-2021_13-33'
fails = []

def calculate_dataset_x_augmentation_embeddings(
    datasets_main_dir: Path, 
    dataset_names: list, 
    augmentations,
    probe: str,
    skip_layers: int,
    method: str, 
    max_samples: int):

    embeddings = []
    meta_features = dict()
    for n in augmentations:
        
        logger.info(f"Augmentation {n}")
        dataset_dirs = [os.path.join(datasets_main_dir, str(n), name) for name in dataset_names]
        
        for name, dataset_dir in zip(dataset_names, dataset_dirs):
            
            logger.info(f"Embedding {name}")

            if str(n)+'-'+name in fails:
                continue
            
            train_dataset, _, _ = get_autodl_dataset(dataset_dir)
            dataset, num_classes = autodl_to_torch(train_dataset)

            probe_network = get_model(probe, pretrained=True, num_classes=num_classes).cuda()
            task2vec = Task2Vec(probe_network, max_samples=max_samples, skip_layers=skip_layers, method = method, loader_opts = {'batch_size': 100})
            embedding, _ = task2vec.embed(dataset)
            embeddings.append(embedding)
            
            dataset_key = str(n)+'-'+name
            meta_features[dataset_key] = {i: embedding.hessian[i] for i in range(len(embedding.hessian))}

            logger.info(f"Embedding {name} completed!")


    for _dir in os.listdir(devel_datasets_dir):

        logger.info(f"Embedding {_dir}")

        train_dataset, _, _ = get_nascomp_dataset(os.path.join(devel_datasets_dir, _dir))
        dataset, num_classes = nascomp_to_torch(train_dataset)

        probe_network = get_model(probe, pretrained=True, num_classes=num_classes).cuda()
        task2vec = Task2Vec(probe_network, max_samples=max_samples, skip_layers=skip_layers, method = method, loader_opts = {'batch_size': 100})
        embedding, _ = task2vec.embed(dataset)
        embeddings.append(embedding)
        
        dataset_key = _dir
        meta_features[dataset_key] = {i: embedding.hessian[i] for i in range(len(embedding.hessian))}

        logger.info(f"Embedding {_dir} completed!")
    
    return embeddings, meta_features

def dump_embeddings(embeddings, task_names, output_dir):
    pickle.dump(embeddings, open(os.path.join(output_dir, 'embeddings.pkl'), 'wb'))
    pickle.dump(task_names, open(os.path.join(output_dir, 'task_names.pkl'), 'wb'))

def main(args):
    ### ALL DATASETS ALL AUGMENTATIONS ###

    dataset_names = GROUPS[args.dataset_group]

    embeddings, meta_features = \
    calculate_dataset_x_augmentation_embeddings(args.dataset_dir, 
                                                dataset_names, 
                                                range(args.n_augmentations),
                                                args.probe_network, 
                                                args.skip_layers,
                                                args.method,
                                                args.max_samples)
    
    task_names = [str(n)+'-'+d for d in dataset_names for n in range(args.n_augmentations) if str(n)+'-'+d not in fails]
    task_names += os.listdir(devel_datasets_dir)

    if args.plot_dist_mat:
        if not os.path.isdir(args.output_dir):
            os.makedirs(args.output_dir)
        task_similarity.plot_distance_matrix(embeddings, task_names, savepath = os.path.join(args.output_dir, 'dist_mat.svg'))

    return embeddings, task_names, meta_features

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser("Task2VecPipeline")
    parser.add_argument("--dataset_dir", type=Path, default = '/data/aad/image_datasets/augmented_datasets')
    parser.add_argument("--dataset_group", type=str, default = 'all') # all/training/validation
    parser.add_argument("--n_augmentations", type=int, default = 30)
    parser.add_argument("--output_dir", type=Path, default = 'dev_meta_features/task2vec_results/meta_dataset_creation_test/')
    parser.add_argument("--plot_dist_mat", type=bool, default = True)
    parser.add_argument("--probe_network", type=str, default = 'resnet34')
    parser.add_argument("--skip_layers", type=int, default = 0)
    parser.add_argument("--method", type=str, default = 'montecarlo')
    parser.add_argument("--max_samples", type=int, default = 10000)
    args, _ = parser.parse_known_args()

    #### ORIGINAL RUN SCRIPT ####
    embeddings, task_names, meta_features = main(args)
    dump_embeddings(embeddings, task_names, args.output_dir)
    dump_meta_features_df_and_csv(meta_features=meta_features, n_augmentations= args.n_augmentations, output_path=args.output_dir)





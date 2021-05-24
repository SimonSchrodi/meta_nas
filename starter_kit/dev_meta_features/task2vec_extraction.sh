#!/bin/bash

#SBATCH -p alldlc_gpu-rtx2080
#SBATCH --job-name mf-ext-nascomp
#SBATCH -o logs/t2v_%A-%a.%x.o
#SBATCH -e logs/t2v_%A-%a.%x.e

#SBATCH --mail-user=ozturk@informatik.uni-freiburg.de
#SBATCH --mail-type=END,FAIL

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 8

source /home/ozturk/anaconda3/bin/activate nascomp
pwd

#python dev_meta_features/task2vec_icgen.py --n_augmentations 1 --skip_layers 0 --max_samples 10000 --probe_network 'resnet18' \
#										--output_dir 'dev_meta_features/task2vec_results/meta_dataset_creation_resnet18_tr/'

python dev_meta_features/task2vec_icgen.py --n_augmentations 30 --skip_layers 0 --max_samples 10000 --probe_network 'resnet34' \
										--output_dir 'dev_meta_features/task2vec_results/meta_dataset_creation_resnet34_tr/'


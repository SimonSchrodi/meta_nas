#!/bin/bash

#SBATCH -p alldlc_gpu-rtx2080
#SBATCH --job-name mf-ext-nascomp
#SBATCH -o logs/t2v_%A-%a.%x.o
#SBATCH -e logs/t2v_%A-%a.%x.e

#SBATCH --mail-user=ozturk@informatik.uni-freiburg.de
#SBATCH --mail-type=END,FAIL

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 2

source /home/ozturk/anaconda3/bin/activate autodl
pwd

python -m task2vec_icgen --n_augmentations 30 --skip_layers 6 --max_samples 1024 \
						 --output_dir 'task2vec_results/meta_dataset_creation/'


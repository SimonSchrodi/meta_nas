#!/bin/bash

#SBATCH -p alldlc_gpu-rtx2080
#SBATCH --job-name it-models-datasets
#SBATCH -o logs/perf_%A-%a.%x.o
#SBATCH -e logs/perf_%A-%a.%x.e

#SBATCH --mail-user=ozturk@informatik.uni-freiburg.de
#SBATCH --mail-type=END,FAIL

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 8

#SBATCH -a 27-27

source /home/ozturk/anaconda3/bin/activate nascomp
pwd

ARGS_FILE=dev_meta_features/performance_run.args
TASK_SPECIFIC_ARGS=$(sed "${SLURM_ARRAY_TASK_ID}q;d" $ARGS_FILE)

echo $TASK_SPECIFIC_ARGS

python dev_meta_features/performance_run.py $TASK_SPECIFIC_ARGS


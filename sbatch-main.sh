#!/bin/bash
#SBATCH --gres=gpu:1 --constraint=gpu1080|gpup100
#SBATCH --job-name="TSC"
#SBATCH  -p publicgpu

module load python/Anaconda3
source activate transformers

cd ~/projects/TimeSeriesClassification

python3 main.py --exp_name $1 --ds_name $2 --aug_method $3 --model $4 --num_epochs $5 --batch_size $6 --num_iter $7
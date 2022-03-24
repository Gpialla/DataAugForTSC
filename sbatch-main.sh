#!/bin/bash
#SBATCH --gres=gpu:1 --constraint=gpu1080|gpup100
#SBATCH --job-name="OA_AO"
#SBATCH  -p grantgpu -A g2021a322g

module load python/Anaconda3
source activate transformers

cd ~/projects/TimeSeriesClassification

python3 main.py --exp_name $1 --ds_name $2 --aug_method $3 --aug_each_epch $4 --only_aug_data $5--multi_aug_method $6 --model $7 --num_epochs $8 --batch_size $9 --iter $10
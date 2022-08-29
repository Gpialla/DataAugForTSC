#!/bin/bash
#SBATCH --gres=gpu:1 --constraint=gpu1080|gpup100
#SBATCH --job-name="OA_AO"
#SBATCH  -p grantgpu -A g2021a322g

module load python/Anaconda3
source activate transformers

cd ~/projects/TimeSeriesClassification

python3 main.py --exp_name $1 --ds_name $2 --aug_each_epch $3 --only_aug_data $4--multi_aug_method $5 --model $6 --num_epochs $7 --batch_size $8 --iter $9

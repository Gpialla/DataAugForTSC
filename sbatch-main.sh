#!/bin/bash
#SBATCH --gres=gpu:1 --constraint=gpu1080|gpup100
#SBATCH --job-name="DA_P"
#SBATCH  -p grantgpu -A g2021a322g

module load python/Anaconda3
source activate transformers

cd ~/projects/DataAugForTSC

python3 main.py --exp_name $1 --archive_name $2 --archive_version $3 --ds_name $4 --aug_each_epch $5 --only_aug_data $6 --multi_aug_method $7 --model $8 --num_epochs $9 --batch_size ${10}

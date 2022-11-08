#!/bin/bash
#SBATCH --gres=gpu:1 --constraint=gpu1080|gpup100
#SBATCH --job-name="DA_P"
#SBATCH  -p grantgpu -A g2022a329g

module load python/Anaconda3
source activate transformers

cd ~/projects/DataAugForTSC

python3 main.py --archive DigitsRTD --model inception --num_epochs 1000 --batch_size 32 --iter 0

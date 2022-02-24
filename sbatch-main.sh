#!/bin/bash
#SBATCH --gres=gpu:1 --constraint=gpu1080|gpup100
#SBATCH --job-name="trsf_pialla"
#SBATCH  -p publicgpu

module load python/Anaconda3
source activate transformers

cd ~/projects/dl-4-tsc_mod

python3 main.py "$1" "$2" "$3" "$4"

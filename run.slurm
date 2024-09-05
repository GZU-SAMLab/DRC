#!/bin/sh
#SBATCH -J PVT_train
#SBATCH -o output/%j.out.log
#SBATCH -e output/%j.err.log
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --ntasks-per-node=20
#SBATCH --gres=gpu:1
python bce_train.py 64 20
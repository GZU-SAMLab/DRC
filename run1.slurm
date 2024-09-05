#!/bin/sh
#SBATCH -J PVT_train
#SBATCH -o output/log.out.%j
#SBATCH -e output/log.err.%j
#SBATCH --partition=gpu-a100-2
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --ntasks-per-node=20
#SBATCH --gres=gpu:1
python bce_SGD1.py --world_size=1 --node_rank=0 --master_port=29500 --batch_size=128 --nw=20 --addr_file='addr_train3.txt'
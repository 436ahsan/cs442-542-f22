#!/bin/bash

#SBATCH --partition=dualGPU
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --time=01:00:00
#SBATCH --job-name=hw6
#SBATCH --gres=gpu:2

source ~/.bashrc

module load cuda
module load openmpi
module load gcc/8.3.0-wbma

nvidia-smi

cd /users/bienz/cs-442-542-f20/heterogenenous
mpirun -n 16 ./hello_world
 

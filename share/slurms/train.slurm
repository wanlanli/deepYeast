#!/bin/bash
#SBATCH -J train_deepyeast_condatf210
#SBATCH -o out/out_%J
#SBATCH -e err/err_%J
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --cpus-per-task 2 
##SBATCH -w dnagpu002
#SBATCH --gres gpu:2
#SBATCH --time=24:0:0
#SBATCH --mem 64g

module purge
module load gcc/10.4.0
cd /work/FAC/FBM/DMF/smartin/cellfusion/wanlan/project/deepyeast
eval "$(conda shell.bash hook)"
conda activate tf210
#Run
python run_model_training.py --mode='train' --model_dir='../model_dir/' --config_file="./configs/config_wl.yaml" --num_gpus=2

#!/bin/bash
#SBATCH -J tf210M
#SBATCH -o out/out_%J
#SBATCH -e err/err_%J
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --cpus-per-task 16 
##########SBATCH -w dnagpu002
#SBATCH --gres gpu:1
#SBATCH --time=20:0:0
#SBATCH --mem 64g

runningnode=$(hostname -s)
user=$(whoami)
head=curnagl.dcsr.unil.ch
port=8880

#print tunnel jupyter log
echo -e "
Command to create ssh tunnel:
ssh -NfL ${port}:${runningnode}:${port} ${user}@${head}
-
Use a Browser on your local machine to go to:
localhost:${port}
"

module purge
module load gcc/10.4.0
eval "$(conda shell.bash hook)"
cd /work/FAC/FBM/DMF/smartin/cellfusion/wanlan
conda activate tf210
#Run Jupyter
jupyter lab --no-browser --port=${port} --ip=${runningnode}

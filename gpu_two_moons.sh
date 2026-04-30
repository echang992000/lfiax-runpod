#!/bin/bash

#SBATCH --job-name=2m_lfiax        ## name of the job.
#SBATCH -A eehui_lab_gpu                      ## account to charge
#SBATCH -p free-gpu                          ## partition/queue name "standard" for paid and "free/free-gpu" fo free
#SBATCH --gres=gpu:A30:1                   ## Specify 1 GPU of type V100
#SBATCH --error=./errors/error_%A_%a.txt    ## error log file name: %A is job id, %a is array task id
#SBATCH --output=./logs/out_%A_%a.out       ## output filename
#SBATCH --nodes=1                        ## number of nodes the job will use
#SBATCH --ntasks=1                       ## number of processes to launch for each array iteration
#SBATCH --cpus-per-task=1               ## number of cores the job needs
#SBATCH --mem-per-cpu=24G
#SBATCH --time=12:00:00                   ## time limit for each array task
#SBATCH --array=1                      ## number of array tasks
#SBATCH --mail-type=fail,end
#SBATCH --mail-user=vzaballa@uci.edu
                                ## $SLURM_ARRAY_TASK_ID takes values from 1 to 100 inclusive

## Activiating the conda environment
source ~/.bashrc

conda activate lfiax_idad

## Run the script
python two_moons.py seed=$SLURM_ARRAY_TASK_ID mi_params.N=32 mi_params.M=31 mi_params.eig_lambda=1e-2
python two_moons.py seed=$SLURM_ARRAY_TASK_ID mi_params.N=32 mi_params.M=31 mi_params.eig_lambda=1e-1
python two_moons.py seed=$SLURM_ARRAY_TASK_ID mi_params.N=32 mi_params.M=31 mi_params.eig_lambda=1e0
python two_moons.py seed=$SLURM_ARRAY_TASK_ID mi_params.N=32 mi_params.M=31 mi_params.eig_lambda=1e1
python two_moons.py seed=$SLURM_ARRAY_TASK_ID mi_params.N=32 mi_params.M=31 mi_params.eig_lambda=1e2
python two_moons.py seed=$SLURM_ARRAY_TASK_ID mi_params.N=64 mi_params.M=63 mi_params.eig_lambda=1e-2
python two_moons.py seed=$SLURM_ARRAY_TASK_ID mi_params.N=64 mi_params.M=63 mi_params.eig_lambda=1e-1
python two_moons.py seed=$SLURM_ARRAY_TASK_ID mi_params.N=64 mi_params.M=63 mi_params.eig_lambda=1e0
python two_moons.py seed=$SLURM_ARRAY_TASK_ID mi_params.N=64 mi_params.M=63 mi_params.eig_lambda=1e1
python two_moons.py seed=$SLURM_ARRAY_TASK_ID mi_params.N=64 mi_params.M=63 mi_params.eig_lambda=1e2
python two_moons.py seed=$SLURM_ARRAY_TASK_ID mi_params.N=128 mi_params.M=127 mi_params.eig_lambda=1e-2
python two_moons.py seed=$SLURM_ARRAY_TASK_ID mi_params.N=128 mi_params.M=127 mi_params.eig_lambda=1e-1
python two_moons.py seed=$SLURM_ARRAY_TASK_ID mi_params.N=128 mi_params.M=127 mi_params.eig_lambda=1e0
python two_moons.py seed=$SLURM_ARRAY_TASK_ID mi_params.N=128 mi_params.M=127 mi_params.eig_lambda=1e1
python two_moons.py seed=$SLURM_ARRAY_TASK_ID mi_params.N=128 mi_params.M=127 mi_params.eig_lambda=1e2
python two_moons.py seed=$SLURM_ARRAY_TASK_ID mi_params.N=256 mi_params.M=255 mi_params.eig_lambda=1e-2
python two_moons.py seed=$SLURM_ARRAY_TASK_ID mi_params.N=256 mi_params.M=255 mi_params.eig_lambda=1e-1
python two_moons.py seed=$SLURM_ARRAY_TASK_ID mi_params.N=256 mi_params.M=255 mi_params.eig_lambda=1e0
python two_moons.py seed=$SLURM_ARRAY_TASK_ID mi_params.N=256 mi_params.M=255 mi_params.eig_lambda=1e1
python two_moons.py seed=$SLURM_ARRAY_TASK_ID mi_params.N=256 mi_params.M=255 mi_params.eig_lambda=1e2
python two_moons.py seed=$SLURM_ARRAY_TASK_ID mi_params.N=512 mi_params.M=511 mi_params.eig_lambda=1e-2
python two_moons.py seed=$SLURM_ARRAY_TASK_ID mi_params.N=512 mi_params.M=511 mi_params.eig_lambda=1e-1
python two_moons.py seed=$SLURM_ARRAY_TASK_ID mi_params.N=512 mi_params.M=511 mi_params.eig_lambda=1e0
python two_moons.py seed=$SLURM_ARRAY_TASK_ID mi_params.N=512 mi_params.M=511 mi_params.eig_lambda=1e1
python two_moons.py seed=$SLURM_ARRAY_TASK_ID mi_params.N=512 mi_params.M=511 mi_params.eig_lambda=1e2

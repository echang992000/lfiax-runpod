#!/bin/bash -l

#SBATCH --job-name=lin_lfiax        ## name of the job.
#SBATCH -A eehui_lab # _gpu                      ## account to charge
#SBATCH -p standard                          ## partition/queue name "standard" for paid and "free" fo free
#SBATCH --error=./errors/error_%A_%a.txt    ## error log file name: %A is job id, %a is array task id
#SBATCH --output=./logs/out_%A_%a.out       ## output filename
#SBATCH --nodes=1                        ## number of nodes the job will use
#SBATCH --ntasks=1                       ## number of processes to launch for each array iteration
#SBATCH --cpus-per-task=1                ## number of cores the job needs
#SBATCH --time=20:00:00                   ## time limit for each array task
#SBATCH --array=1-2                      ## number of array tasks
#SBATCH --mail-type=fail,end
#SBATCH --mail-user=vzaballa@uci.edu
                                ## $SLURM_ARRAY_TASK_ID takes values from 1 to 100 inclusive

## Activiating the conda environment
source ~/.bashrc
module purge
module load mamba/24.3.0
. ~/.mycondainit-mamba-24.3.0
conda activate lfiax_idad

## Run the script
# python examples/linear_regression.py -m # ++seed=$SLURM_ARRAY_TASK_ID ++contrastive_sampling.M=10,11
python linear_regression.py ++seed=$SLURM_ARRAY_TASK_ID designs.num_xi=100

# python linear_regression_snpe.py ++seed=$SLURM_ARRAY_TASK_ID designs.num_xi=10 wandb.project='iclr_lfiax_linreg_snpe_design_scaled' optimization_params.eig_lambda=0.5 designs.scale_factor=11
# python linear_regression_snpe.py ++seed=$SLURM_ARRAY_TASK_ID designs.num_xi=10 wandb.project='iclr_lfiax_linreg_snpe_design_scaled' optimization_params.eig_lambda=0.5 designs.scale_factor=12
# python linear_regression_snpe.py ++seed=$SLURM_ARRAY_TASK_ID designs.num_xi=10 wandb.project='iclr_lfiax_linreg_snpe_design_scaled' optimization_params.eig_lambda=0.5 designs.scale_factor=13
# python linear_regression_snpe.py ++seed=$SLURM_ARRAY_TASK_ID designs.num_xi=10 wandb.project='iclr_lfiax_linreg_snpe_design_scaled' optimization_params.eig_lambda=0.5 designs.scale_factor=14
# python linear_regression_snpe.py ++seed=$SLURM_ARRAY_TASK_ID designs.num_xi=10 wandb.project='iclr_lfiax_linreg_snpe_design_scaled' optimization_params.eig_lambda=1.0
# python linear_regression_snpe.py ++seed=$SLURM_ARRAY_TASK_ID designs.num_xi=10 wandb.project='iclr_lfiax_linreg_snpe_design_scaled' optimization_params.eig_lambda=1.0 designs.scale_factor=11
# python linear_regression_snpe.py ++seed=$SLURM_ARRAY_TASK_ID designs.num_xi=10 wandb.project='iclr_lfiax_linreg_snpe_design_scaled' optimization_params.eig_lambda=1.0 designs.scale_factor=12
# python linear_regression_snpe.py ++seed=$SLURM_ARRAY_TASK_ID designs.num_xi=10 wandb.project='iclr_lfiax_linreg_snpe_design_scaled' optimization_params.eig_lambda=1.0 designs.scale_factor=13
# python linear_regression_snpe.py ++seed=$SLURM_ARRAY_TASK_ID designs.num_xi=10 wandb.project='iclr_lfiax_linreg_snpe_design_scaled' optimization_params.eig_lambda=1.0 designs.scale_factor=14

# python linear_regression_snpe.py ++seed=$SLURM_ARRAY_TASK_ID designs.num_xi=10 wandb.project='iclr_lfiax_linreg_snpe_design_scaled' optimization_params.eig_lambda=0.0
# python linear_regression_snpe.py ++seed=$SLURM_ARRAY_TASK_ID designs.num_xi=10 wandb.project='iclr_lfiax_linreg_snpe_design_scaled' optimization_params.eig_lambda=0.0 designs.scale_factor=11
# python linear_regression_snpe.py ++seed=$SLURM_ARRAY_TASK_ID designs.num_xi=10 wandb.project='iclr_lfiax_linreg_snpe_design_scaled' optimization_params.eig_lambda=0.0 designs.scale_factor=12
# python linear_regression_snpe.py ++seed=$SLURM_ARRAY_TASK_ID designs.num_xi=10 wandb.project='iclr_lfiax_linreg_snpe_design_scaled' optimization_params.eig_lambda=0.0 designs.scale_factor=13
# python linear_regression_snpe.py ++seed=$SLURM_ARRAY_TASK_ID designs.num_xi=10 wandb.project='iclr_lfiax_linreg_snpe_design_scaled' optimization_params.eig_lambda=0.0 designs.scale_factor=14

#python sbiDOEMAN/main_bma.py ++seed=$SLURM_ARRAY_TASK_ID ++num_design_rounds=5 ++BMP.model='onestep'
#python sbiDOEMAN/main_random.py ++seed=$SLURM_ARRAY_TASK_ID ++SDM.random=True ++num_design_rounds=5 ++BMP.model="onestep"
#python sbiDOEMAN/main_equidistant.py ++seed=$SLURM_ARRAY_TASK_ID ++SDM.control=True 

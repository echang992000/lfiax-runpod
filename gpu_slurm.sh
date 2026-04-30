#!/bin/bash -l

#SBATCH --job-name=sir_lfiax        ## name of the job.
#SBATCH -A eehui_lab_gpu                      ## account to charge
#SBATCH -p free-gpu                          ## partition/queue name "standard" for paid and "free/free-gpu" fo free
#SBATCH --gres=gpu:A30:1                   ## Specify 1 GPU of type V100
#SBATCH --error=./errors/error_%A_%a.txt    ## error log file name: %A is job id, %a is array task id
#SBATCH --output=./logs/out_%A_%a.out       ## output filename
#SBATCH --nodes=1                        ## number of nodes the job will use
#SBATCH --ntasks=1                       ## number of processes to launch for each array iteration
#SBATCH --cpus-per-task=1               ## number of cores the job needs
#SBATCH --mem-per-cpu=32G
#SBATCH --time=28:00:00                   ## time limit for each array task
#SBATCH --array=1                      ## number of array tasks
#SBATCH --mail-type=fail,end
#SBATCH --mail-user=vzaballa@uci.edu
                                ## $SLURM_ARRAY_TASK_ID takes values from 1 to 100 inclusive

## Activiating the conda environment
source ~/.bashrc
module purge
module load mamba/24.3.0
module load cuda/12.2.0
. ~/.mycondainit-mamba-24.3.0
conda activate lfiax_idad

## Run the script
# python examples/linear_regression.py -m # ++seed=$SLURM_ARRAY_TASK_ID ++contrastive_sampling.M=10,11
# python linear_regression.py seed=$SLURM_ARRAY_TASK_ID optimization_params.training_steps=10
# python sir.py seed=1 optimization_params.eig_lambda=1.0 flow_params.reset_flow=false

# =========================
# SIR baselines
# =========================

python sir.py seed=1 baseline.design_policy=random baseline.likelihood_objective=nle
python sir.py seed=1 baseline.design_policy=random baseline.likelihood_objective=infonce_lambda optimization_params.eig_lambda=0.01
python sir.py seed=1 baseline.design_policy=random baseline.likelihood_objective=infonce_lambda optimization_params.eig_lambda=0.1
python sir.py seed=1 baseline.design_policy=random baseline.likelihood_objective=infonce_lambda optimization_params.eig_lambda=1.0

python sir.py seed=1 baseline.design_policy=sobol baseline.likelihood_objective=nle
python sir.py seed=1 baseline.design_policy=sobol baseline.likelihood_objective=infonce_lambda optimization_params.eig_lambda=0.01
python sir.py seed=1 baseline.design_policy=sobol baseline.likelihood_objective=infonce_lambda optimization_params.eig_lambda=0.1
python sir.py seed=1 baseline.design_policy=sobol baseline.likelihood_objective=infonce_lambda optimization_params.eig_lambda=1.0

python sir.py seed=2 baseline.design_policy=random baseline.likelihood_objective=nle
python sir.py seed=2 baseline.design_policy=random baseline.likelihood_objective=infonce_lambda optimization_params.eig_lambda=0.01
python sir.py seed=2 baseline.design_policy=random baseline.likelihood_objective=infonce_lambda optimization_params.eig_lambda=0.1
python sir.py seed=2 baseline.design_policy=random baseline.likelihood_objective=infonce_lambda optimization_params.eig_lambda=1.0

python sir.py seed=2 baseline.design_policy=sobol baseline.likelihood_objective=nle
python sir.py seed=2 baseline.design_policy=sobol baseline.likelihood_objective=infonce_lambda optimization_params.eig_lambda=0.01
python sir.py seed=2 baseline.design_policy=sobol baseline.likelihood_objective=infonce_lambda optimization_params.eig_lambda=0.1
python sir.py seed=2 baseline.design_policy=sobol baseline.likelihood_objective=infonce_lambda optimization_params.eig_lambda=1.0

python sir.py seed=3 baseline.design_policy=random baseline.likelihood_objective=nle
python sir.py seed=3 baseline.design_policy=random baseline.likelihood_objective=infonce_lambda optimization_params.eig_lambda=0.01
python sir.py seed=3 baseline.design_policy=random baseline.likelihood_objective=infonce_lambda optimization_params.eig_lambda=0.1
python sir.py seed=3 baseline.design_policy=random baseline.likelihood_objective=infonce_lambda optimization_params.eig_lambda=1.0

python sir.py seed=3 baseline.design_policy=sobol baseline.likelihood_objective=nle
python sir.py seed=3 baseline.design_policy=sobol baseline.likelihood_objective=infonce_lambda optimization_params.eig_lambda=0.01
python sir.py seed=3 baseline.design_policy=sobol baseline.likelihood_objective=infonce_lambda optimization_params.eig_lambda=0.1
python sir.py seed=3 baseline.design_policy=sobol baseline.likelihood_objective=infonce_lambda optimization_params.eig_lambda=1.0
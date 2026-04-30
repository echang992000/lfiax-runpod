#!/bin/bash -l

#SBATCH --job-name=bmp_lfiax        ## name of the job.
#SBATCH -A eehui_lab_gpu                      ## account to charge
#SBATCH -p gpu                          ## partition/queue name "standard" for paid and "free/free-gpu" fo free
#SBATCH --gres=gpu:A30:1                   ## Specify 1 GPU of type V100
#SBATCH --error=./errors/error_%A_%a.txt    ## error log file name: %A is job id, %a is array task id
#SBATCH --output=./logs/out_%A_%a.out       ## output filename
#SBATCH --nodes=1                        ## number of nodes the job will use
#SBATCH --ntasks=1                       ## number of processes to launch for each array iteration
#SBATCH --cpus-per-task=1               ## number of cores the job needs
#SBATCH --mem-per-cpu=32G
#SBATCH --time=12:00:00                   ## time limit for each array task
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
# python BMP.py seed=$SLURM_ARRAY_TASK_ID flow_params.reset_flow=false optimization_params.eig_lambda=0.1 flow_params.mlp_hidden_size=128 flow_params.num_bins=8 flow_params.mlp_num_layers=2 contrastive_sampling.N=32 contrastive_sampling.M=31 optimization_params.refine_likelihood_rounds=0 post_optimization.vi_type=fKL
# =========================
# BMP baselines
# =========================

python BMP.py seed=1 baseline.design_policy=random baseline.likelihood_objective=nle
python BMP.py seed=1 baseline.design_policy=random baseline.likelihood_objective=infonce_lambda optimization_params.eig_lambda=0.01
python BMP.py seed=1 baseline.design_policy=random baseline.likelihood_objective=infonce_lambda optimization_params.eig_lambda=0.1
python BMP.py seed=1 baseline.design_policy=random baseline.likelihood_objective=infonce_lambda optimization_params.eig_lambda=1.0

# python BMP.py seed=1 baseline.design_policy=sobol baseline.likelihood_objective=nle
# python BMP.py seed=1 baseline.design_policy=sobol baseline.likelihood_objective=infonce_lambda optimization_params.eig_lambda=0.01
# python BMP.py seed=1 baseline.design_policy=sobol baseline.likelihood_objective=infonce_lambda optimization_params.eig_lambda=0.1
# python BMP.py seed=1 baseline.design_policy=sobol baseline.likelihood_objective=infonce_lambda optimization_params.eig_lambda=1.0

python BMP.py seed=2 baseline.design_policy=random baseline.likelihood_objective=nle
python BMP.py seed=2 baseline.design_policy=random baseline.likelihood_objective=infonce_lambda optimization_params.eig_lambda=0.01
python BMP.py seed=2 baseline.design_policy=random baseline.likelihood_objective=infonce_lambda optimization_params.eig_lambda=0.1
python BMP.py seed=2 baseline.design_policy=random baseline.likelihood_objective=infonce_lambda optimization_params.eig_lambda=1.0

# python BMP.py seed=2 baseline.design_policy=sobol baseline.likelihood_objective=nle
# python BMP.py seed=2 baseline.design_policy=sobol baseline.likelihood_objective=infonce_lambda optimization_params.eig_lambda=0.01
# python BMP.py seed=2 baseline.design_policy=sobol baseline.likelihood_objective=infonce_lambda optimization_params.eig_lambda=0.1
# python BMP.py seed=2 baseline.design_policy=sobol baseline.likelihood_objective=infonce_lambda optimization_params.eig_lambda=1.0

python BMP.py seed=3 baseline.design_policy=random baseline.likelihood_objective=nle
python BMP.py seed=3 baseline.design_policy=random baseline.likelihood_objective=infonce_lambda optimization_params.eig_lambda=0.01
python BMP.py seed=3 baseline.design_policy=random baseline.likelihood_objective=infonce_lambda optimization_params.eig_lambda=0.1
python BMP.py seed=3 baseline.design_policy=random baseline.likelihood_objective=infonce_lambda optimization_params.eig_lambda=1.0

# python BMP.py seed=3 baseline.design_policy=sobol baseline.likelihood_objective=nle
# python BMP.py seed=3 baseline.design_policy=sobol baseline.likelihood_objective=infonce_lambda optimization_params.eig_lambda=0.01
# python BMP.py seed=3 baseline.design_policy=sobol baseline.likelihood_objective=infonce_lambda optimization_params.eig_lambda=0.1
# python BMP.py seed=3 baseline.design_policy=sobol baseline.likelihood_objective=infonce_lambda optimization_params.eig_lambda=1.0


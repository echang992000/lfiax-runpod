#!/bin/bash -l

#SBATCH --job-name=2m_lfiax        ## name of the job.
#SBATCH -A eehui_lab_gpu                      ## account to charge
#SBATCH -p free                          ## partition/queue name "standard" for paid and "free/free-gpu" fo free
# #SBATCH --gres=gpu:A30:1                   ## Specify 1 GPU of type V100
#SBATCH --error=./errors/error_%A_%a.txt    ## error log file name: %A is job id, %a is array task id
#SBATCH --output=./logs/out_%A_%a.out       ## output filename
#SBATCH --nodes=1                        ## number of nodes the job will use
#SBATCH --ntasks=1                       ## number of processes to launch for each array iteration
#SBATCH --cpus-per-task=8              ## CPU EPIG scoring needs high memory; 24 * 24G = 576G
#SBATCH --mem-per-cpu=24G
#SBATCH --time=12:00:00                   ## time limit for each array task
#SBATCH --array=1-15                   ## 5 regularization strengths x 3 theta* set sizes
#SBATCH --mail-type=fail,end
#SBATCH --mail-user=vzaballa@uci.edu
                                ## $SLURM_ARRAY_TASK_ID takes values from 1 to 100 inclusive

## Activiating the conda environment
source ~/.bashrc
module purge
module load mamba/24.3.0
. ~/.mycondainit-mamba-24.3.0
conda activate lfiax_idad

export JAX_PLATFORMS=cpu
export XLA_FLAGS="--xla_force_host_platform_device_count=${SLURM_CPUS_PER_TASK}"

LAMBDA_VALUES=(1e-2 1e-1 1e0 1e1 1e2)
THETA_STAR_K_VALUES=(16 32 64)
SEED=${SEED:-1}

TASK_INDEX=$((SLURM_ARRAY_TASK_ID - 1))
NUM_K=${#THETA_STAR_K_VALUES[@]}
LAMBDA_INDEX=$((TASK_INDEX / NUM_K))
K_INDEX=$((TASK_INDEX % NUM_K))

EIG_LAMBDA=${LAMBDA_VALUES[$LAMBDA_INDEX]}
THETA_STAR_K=${THETA_STAR_K_VALUES[$K_INDEX]}

echo "Running two_moons_active_learning sweep:"
echo "  seed=${SEED}"
echo "  mi_params.eig_lambda=${EIG_LAMBDA}"
echo "  mi_params.top_k=${THETA_STAR_K}"

python two_moons_active_learning.py \
    seed=${SEED} \
    simulator.num_samples=2 \
    mi_params.N=1 \
    mi_params.M=512 \
    mi_params.eig_lambda=${EIG_LAMBDA} \
    mi_params.top_k=${THETA_STAR_K} \
    main_training.num_epochs=1000 \
    active_learn_rounds=8 \
    epig.num_candidates=100 \
    epig.K=64

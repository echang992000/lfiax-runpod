#!/bin/bash -l

#SBATCH --job-name=bmp_minebed
#SBATCH -A eehui_lab_gpu
#SBATCH -p free
# #SBATCH --gres=gpu:A30:1
#SBATCH --error=./errors/error_%A_%a.txt
#SBATCH --output=./logs/out_%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G
#SBATCH --time=12:00:00
#SBATCH --array=1-9
#SBATCH --mail-type=fail,end
#SBATCH --mail-user=vzaballa@uci.edu

source ~/.bashrc
module purge
module load mamba/24.3.0
module load cuda/12.2.0
. ~/.mycondainit-mamba-24.3.0
conda activate lfiax_idad

# Sensitivity grid:
#   sim_call_budget in {2, 4, 8}
#   seeds in {1, 2, 3}
# DATASIZE is derived in BMP_minebed.py as:
#   floor(minebed.total_sim_budget / minebed.sim_call_budget)
# BATCHSIZE defaults to DATASIZE.
# For this compute-matched sweep, use one initial BO point so small
# sim_call_budget values still leave room for BO iterations.
BUDGETS=(2 4 8)
SEEDS=(1 2 3)

TASK_INDEX=$((SLURM_ARRAY_TASK_ID - 1))
BUDGET_INDEX=$((TASK_INDEX % ${#BUDGETS[@]}))
SEED_INDEX=$((TASK_INDEX / ${#BUDGETS[@]}))

SIM_CALL_BUDGET=${BUDGETS[$BUDGET_INDEX]}
SEED=${SEEDS[$SEED_INDEX]}

python BMP_minebed.py \
  seed=${SEED} \
  experiment.design_rounds=1 \
  minebed.bo_init_num=1 \
  minebed.sim_call_budget=${SIM_CALL_BUDGET} \
  minebed.total_sim_budget=100000 \
  minebed.disable_gpyopt_early_stop=true \
  minebed.datasize=null \
  minebed.batchsize=null

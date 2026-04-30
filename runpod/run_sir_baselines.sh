#!/usr/bin/env bash
# Local equivalent of ``gpu_slurm.sh`` for a RunPod CUDA pod (or any
# single-GPU box). Strips the SLURM directives + remote conda
# activation; preserves every ``python sir.py …`` invocation.
#
# Overrides applied to every invocation:
#   * ``experiment.device=cuda``  — torch parts stay on GPU
#   * ``wandb.use_wandb=false``   — no remote logging credentials needed
#   * ``experiment.hpc=false``    — local file paths
#   * ``hydra.run.dir=...``       — keep one dir per (seed, policy, obj)
#
# All 3 seeds × {random, sobol} × {nle, infonce_lambda(λ=0.01,0.1,1.0)}
# = 24 invocations. At default ``training_steps=10_000`` × 2 design
# rounds, expect roughly 30-90 min per invocation on an A100/H100; an
# A6000/3090 will be 2-3× slower. Set ``LFIAX_TRAINING_STEPS`` in the
# environment to override (e.g. ``LFIAX_TRAINING_STEPS=2000`` for a
# faster smoke pass).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$REPO_ROOT"

if ! command -v python >/dev/null; then
  echo "python not on PATH — activate your pod environment first." >&2
  exit 1
fi

if [ ! -f "$REPO_ROOT/test_prior_sde_numpy.pkl" ]; then
  echo "[run] generating test_prior_sde_numpy.pkl"
  python "$REPO_ROOT/runpod/gen_test_pkl.py"
fi

TRAINING_STEPS="${LFIAX_TRAINING_STEPS:-10000}"
DESIGN_ROUNDS="${LFIAX_DESIGN_ROUNDS:-2}"
LOG_ROOT="${LFIAX_LOG_ROOT:-$REPO_ROOT/runpod_runs}"
mkdir -p "$LOG_ROOT"

echo "[run] training_steps=$TRAINING_STEPS  design_rounds=$DESIGN_ROUNDS"
echo "[run] log root=$LOG_ROOT"

run_one() {
  local seed=$1
  local policy=$2
  local objective=$3
  local lam=${4:-}
  local tag="seed${seed}_${policy}_${objective}${lam:+_lam${lam}}"
  local out_dir="$LOG_ROOT/$tag"
  mkdir -p "$out_dir"

  local extra_lam=""
  if [ -n "$lam" ]; then
    extra_lam="optimization_params.eig_lambda=${lam}"
  fi

  echo
  echo "[run] === $tag ==="
  python sir.py \
    seed=$seed \
    experiment.device=cuda \
    experiment.hpc=false \
    experiment.design_rounds=$DESIGN_ROUNDS \
    optimization_params.training_steps=$TRAINING_STEPS \
    wandb.use_wandb=false \
    baseline.design_policy=$policy \
    baseline.likelihood_objective=$objective \
    $extra_lam \
    "hydra.run.dir=$out_dir/hydra" \
    2>&1 | tee "$out_dir/stdout.log"
}

# Seeds × design policies × likelihood objectives — matches gpu_slurm.sh.
for seed in 1 2 3; do
  for policy in random sobol; do
    run_one "$seed" "$policy" "nle"
    run_one "$seed" "$policy" "infonce_lambda" 0.01
    run_one "$seed" "$policy" "infonce_lambda" 0.1
    run_one "$seed" "$policy" "infonce_lambda" 1.0
  done
done

echo
echo "[run] all 24 invocations complete. Results under $LOG_ROOT"

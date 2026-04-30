#!/usr/bin/env bash
# One-shot environment setup for a RunPod CUDA pod that already has
# Python and CUDA available (e.g. the default PyTorch 2.1.0 CUDA 12.1
# template). Equivalent to the Dockerfile but runs inplace.
#
# Usage (from a fresh pod):
#   git clone https://github.com/<your fork>/lfiax-code.git
#   cd lfiax-code
#   bash runpod/setup.sh
#
# What this does:
#   1. Pip-installs the JAX (CUDA), lfiax, harness, hydra, wandb stack.
#   2. Generates ``test_prior_sde_numpy.pkl`` that ``sir.py`` requires.
#   3. Verifies the JAX CUDA backend is live.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$REPO_ROOT"

echo "[setup] repo root: $REPO_ROOT"

python -m pip install --upgrade pip wheel
python -m pip install -r runpod/requirements.txt

# Editable installs. ``-e`` so any in-place edits to lfiax/agent-harness
# take effect without reinstall.
python -m pip install -e .
python -m pip install -e agent-harness

# Generate test pkl in repo root (sir.py loads from cwd).
python runpod/gen_test_pkl.py

# Sanity check: JAX should now see the CUDA device, and importing
# lfiax + cli-anything-lfiax should both succeed.
echo "[setup] verifying JAX CUDA backend..."
python - <<'PY'
import jax
print("jax version :", jax.__version__)
print("backend     :", jax.default_backend())
print("devices     :", jax.devices())
from lfiax.flows.nsf import make_nsf  # noqa: F401
from lfiax.utils.oed_losses import lf_pce_design_dist_sir  # noqa: F401
from cli_anything.lfiax.core.oed import describe_backend
print("oed describe:", describe_backend()["status"])
PY

echo "[setup] done. Run runpod/run_sir_baselines.sh to launch the sweep."

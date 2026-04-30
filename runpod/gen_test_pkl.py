"""Generate the ``test_prior_sde_numpy.pkl`` file that sir.py loads.

sir.py (line 514) opens ``test_prior_sde_numpy.pkl`` from cwd and
expects a dict with at least ``ts`` (used everywhere) plus the
``final_ys`` / ``theta_0`` / ``ys`` / ``prior_samples`` keys that the
upstream ``solve_sir_sdes`` produces. There is no generator for this
file in the lfiax-code repo, so we create one here from the same
Lognormal prior that ``sample_lognormal_with_log_probs`` uses.

Run from the repo root: ``python runpod/gen_test_pkl.py``.
"""
from __future__ import annotations

import os
import pickle as pkl
import sys

import numpy as np
import torch
import jax.random as jrandom

REPO_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from lfiax.utils.simulators import sample_lognormal_with_log_probs
from lfiax.utils.torch_utils import solve_sir_sdes


def main():
    out = os.path.join(REPO_ROOT, "test_prior_sde_numpy.pkl")
    if os.path.exists(out):
        print(f"already present: {out}")
        return 0

    num_samples = 256
    key = jrandom.PRNGKey(0)
    samples_jax, log_probs_jax = sample_lognormal_with_log_probs(key, num_samples)
    params = torch.from_numpy(np.asarray(samples_jax)).float()
    log_probs = torch.from_numpy(np.asarray(log_probs_jax)).float()

    # Pick the available torch device. RunPod CUDA pods see "cuda"; the
    # solve_sir_sdes routine accepts a device string. We don't push to
    # cuda here because solve_sir_sdes builds a fresh torch tensor
    # internally and `set_seed` is sensitive to device choice — keep on
    # cpu for reproducibility, the SDE solve is fast.
    save_dict = solve_sir_sdes(
        num_samples=num_samples,
        device="cpu",
        grid=10_000,
        save=False,
        savegrad=False,
        params=params,
        params_log_probs=log_probs,
        seed=[0],
    )

    # sir.py also accesses sde_dict['final_ys'] and sde_dict['theta_0']
    # when ``experiment.debug=True`` — supply both so debug mode works.
    save_dict["final_ys"] = save_dict["ys"][-1].numpy()
    save_dict["theta_0"] = save_dict["prior_samples"].numpy()

    with open(out, "wb") as fh:
        pkl.dump(save_dict, fh)
    print(f"wrote {out} (num_samples={save_dict['num_samples']}, ts shape="
          f"{tuple(save_dict['ts'].shape)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

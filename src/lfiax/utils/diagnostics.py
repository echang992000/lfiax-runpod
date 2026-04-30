from typing import Dict, Any
import torch
from sbi.diagnostics.lc2st import LC2ST
import jax.numpy as jnp

Array = jnp.ndarray

def lc2st_metrics(xs: Array,
                  thetas: Array,
                  posterior_samples: Array,
                  device: str = "cpu",
                  seed: int = 0) -> Dict[str, Any]:
    """Compute LC2ST statistic / p‑value / reject.

    Packed into a dict for easy WandB logging.
    """
    # Convert JAX ⇒ Torch once
    x_torch = torch.as_tensor(xs).to(device).float()
    thetas_t = torch.as_tensor(thetas).to(device).float()
    posts_t = torch.as_tensor(posterior_samples).to(device).float()

    lc2st = LC2ST(
        thetas=thetas_t,
        xs=x_torch.T,   # LC2ST expects (N, …)
        posterior_samples=posts_t,
        seed=seed,
        num_folds=1,
        num_ensemble=1,
        classifier="mlp",
        z_score=False,
        num_trials_null=100,
        permutation=True,
    )
    lc2st.train_under_null_hypothesis()
    lc2st.train_on_observed_data()

    theta_o = posts_t
    stat = lc2st.get_statistic_on_observed_data(theta_o=theta_o, x_o=x_torch)
    p_val = lc2st.p_value(theta_o=theta_o, x_o=x_torch)
    reject = lc2st.reject_test(theta_o=theta_o, x_o=x_torch, alpha=0.05)

    return float(stat), float(p_val), bool(reject)

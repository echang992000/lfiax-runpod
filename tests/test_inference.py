import pytest

jax = pytest.importorskip("jax")
blackjax = pytest.importorskip("blackjax")
distrax = pytest.importorskip("distrax")
import jax
import jax.numpy as jnp

from lfiax.utils.utils import run_mcmc


def test_run_mcmc_shapes():
    def logp(theta):
        return distrax.Normal(0.0, 1.0).log_prob(theta).sum()

    key = jax.random.PRNGKey(0)
    theta_0 = jnp.zeros((10,1))
    samples, lps = run_mcmc(key, logp, theta_0, 10, 5)
    assert samples.shape == (5,1)
    assert jnp.isfinite(lps).all()

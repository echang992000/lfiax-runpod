import pytest

jax = pytest.importorskip("jax")
haiku = pytest.importorskip("haiku")
import jax
import jax.numpy as jnp
import haiku as hk

from lfiax.flows.nsf import make_nsf
from lfiax.utils.oed_losses import lf_pce_design_dist_sir, lf_pce_design_dist_bmp


def _dummy_log_prob_setup():
    @hk.transform
    def log_prob(x, theta, xi):
        model = make_nsf(event_shape=(1,), num_layers=1, hidden_sizes=[2], num_bins=4, conditional=True)
        return model.log_prob(x, theta, xi)
    params = log_prob.init(jax.random.PRNGKey(0), jnp.ones((1,1)), jnp.ones((1,1)), jnp.ones((1,1)))
    return log_prob, params


def test_lf_pce_design_dist_sir_shapes():
    log_prob, params = _dummy_log_prob_setup()
    xi_params = {"xi_mu": jnp.zeros((1,)), "xi_stddev": jnp.ones((1,))}
    prng = jax.random.PRNGKey(0)
    final_ys = jnp.ones((2,1))
    ts = jnp.linspace(0,1,3)
    theta_0 = jnp.ones((1,1))
    loss, aux = lf_pce_design_dist_sir(params, xi_params, prng, final_ys, ts, theta_0, log_prob_fun=lambda p,k,x,t,xi: log_prob.apply(p,k,x,t,xi), N=1, M=1)
    assert jnp.isfinite(loss)


def test_lf_pce_design_dist_bmp_shapes():
    log_prob, params = _dummy_log_prob_setup()
    xi_params = {"xi_mu": jnp.zeros((1,)), "xi_stddev": jnp.ones((1,))}
    prng = jax.random.PRNGKey(0)
    design_key = jax.random.PRNGKey(1)
    theta_0 = jnp.ones((1,1))
    scaled_x = jnp.ones((1,1))
    loss, aux = lf_pce_design_dist_bmp(params, xi_params, None, None, prng, design_key, theta_0, scaled_x, lambda p,k,x,t,xi: log_prob.apply(p,k,x,t,xi), N=1, M=1)
    assert jnp.isfinite(loss)

import pytest

jax = pytest.importorskip("jax")
haiku = pytest.importorskip("haiku")
import jax
import jax.numpy as jnp
import haiku as hk

from lfiax.flows.nsf import make_nsf


def test_flow_initialization():
    @hk.transform
    def log_prob(x, theta, xi):
        model = make_nsf(event_shape=(1,), num_layers=1, hidden_sizes=[2], num_bins=4, conditional=True)
        return model.log_prob(x, theta, xi)

    params = log_prob.init(jax.random.PRNGKey(0), jnp.zeros((1,1)), jnp.zeros((1,1)), jnp.zeros((1,1)))
    out = log_prob.apply(params, jax.random.PRNGKey(1), jnp.zeros((1,1)), jnp.zeros((1,1)), jnp.zeros((1,1)))
    assert out.shape == (1,)
    assert jnp.isfinite(out).all()


def test_vector_flow_dropout_changes_log_prob_after_update():
    @hk.transform
    def log_prob(x, theta, xi):
        model = make_nsf(
            event_shape=(2,),
            num_layers=1,
            hidden_sizes=[8],
            num_bins=4,
            conditional=True,
            dropout_rate=0.5,
        )
        return model.log_prob(x, theta, xi)

    x = jnp.array([[0.25, 0.75], [0.35, 0.65]])
    theta = jnp.array([[0.1, -0.2], [0.2, -0.1]])
    xi = jnp.zeros((2, 0))
    params = log_prob.init(jax.random.PRNGKey(0), x, theta, xi)

    def loss_fn(params):
        return -jnp.mean(log_prob.apply(params, jax.random.PRNGKey(1), x, theta, xi))

    grads = jax.grad(loss_fn)(params)
    params = jax.tree_util.tree_map(lambda p, g: p - 0.1 * g, params, grads)

    out1 = log_prob.apply(params, jax.random.PRNGKey(2), x, theta, xi)
    out2 = log_prob.apply(params, jax.random.PRNGKey(3), x, theta, xi)

    assert out1.shape == (2,)
    assert out2.shape == (2,)
    assert jnp.isfinite(out1).all()
    assert jnp.isfinite(out2).all()
    assert not jnp.allclose(out1, out2)


def test_vector_flow_without_dropout_is_deterministic():
    @hk.transform
    def log_prob(x, theta, xi):
        model = make_nsf(
            event_shape=(2,),
            num_layers=1,
            hidden_sizes=[8],
            num_bins=4,
            conditional=True,
            dropout_rate=0.0,
        )
        return model.log_prob(x, theta, xi)

    x = jnp.array([[0.25, 0.75], [0.35, 0.65]])
    theta = jnp.array([[0.1, -0.2], [0.2, -0.1]])
    xi = jnp.zeros((2, 0))
    params = log_prob.init(jax.random.PRNGKey(0), x, theta, xi)

    out1 = log_prob.apply(params, jax.random.PRNGKey(2), x, theta, xi)
    out2 = log_prob.apply(params, jax.random.PRNGKey(3), x, theta, xi)

    assert jnp.allclose(out1, out2)

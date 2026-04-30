import pytest

numpy = pytest.importorskip("numpy")
import numpy as np

jax = pytest.importorskip("jax")
import jax.numpy as jnp

# bmp simulator
bmp_mod = pytest.importorskip("bmp_simulator.simulate_bmp")
from bmp_simulator.simulate_bmp import bmp_simulator

from lfiax.utils.simulators import simulate_sir

def test_bmp_simulator_shape():
    d = np.array([[0.1]])
    p = np.ones((1, 2))
    out = bmp_simulator(d, p)
    assert out.shape[0] == p.shape[0]
    assert out.shape[-1] == d.shape[0]


def test_simulate_sir_basic():
    xi = jnp.array([[5.0], [10.0]])
    ts = jnp.linspace(0, 20, 5)
    ys = jnp.tile(jnp.arange(5.0)[:, None], (1, 2))
    result, mean, std = simulate_sir(xi, ts, ys)
    assert result.shape == (2, 1)
    assert jnp.isfinite(mean).all()
    assert jnp.isfinite(std).all()

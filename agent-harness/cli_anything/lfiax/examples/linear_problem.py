"""Example BOED problem: scalar linear-Gaussian model.

User supplies two callables that cli-anything-lfiax will resolve at runtime:

    prior(key, n)                       -> theta  shape (n, theta_dim)
    simulator(theta, xi, key)           -> y      shape (n, y_dim)

Model:
    theta ~ N(0, 1)                     (theta_dim = 1)
    y | theta, xi = theta * xi + eps    (y_dim = 1, eps ~ N(0, 0.1^2))

JAX is imported inside the functions so this module can be introspected
without the JAX stack installed (e.g. by `oed validate`).

Used with examples/linear_spec.json.
"""
from __future__ import annotations


def prior(key, n):
    import jax.random as jrandom
    return jrandom.normal(key, shape=(int(n), 1))


def simulator(theta, xi, key):
    import jax.numpy as jnp
    import jax.random as jrandom
    theta = jnp.asarray(theta)
    xi = jnp.asarray(xi)
    if theta.ndim == 1:
        theta = theta[:, None]
    xi_b = jnp.broadcast_to(xi, (theta.shape[0], xi.shape[-1]))
    noise = 0.1 * jrandom.normal(key, shape=theta.shape)
    return theta * xi_b + noise

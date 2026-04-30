import os
import sys

import numpy as np
import jax
import jax.numpy as jnp
import jax.lax as lax
import jax.random as jrandom
from jax.scipy.special import logsumexp

from functools import partial
import distrax
import haiku as hk

from lfiax.utils.simulators import sim_linear_prior, sim_linear_data_vmap, sim_linear_prior_M_samples, simulate_sir
from lfiax.utils.utils import standard_scale, get_calibration_error_jax


from typing import Any, Callable, Tuple

Array = jnp.ndarray
PRNGKey = Array


def _safe_mean_terms(terms):
    mask = jnp.isnan(terms) | (terms == -jnp.inf) | (terms == jnp.inf)
    nonnan = jnp.sum(~mask, axis=0, dtype=jnp.float32)
    terms = jnp.where(mask, 0., terms)
    loss = terms / nonnan
    agg_loss = jnp.sum(loss)
    return agg_loss, loss


def _safe_sum(terms):
    mask = jnp.isnan(terms) | (terms == -jnp.inf) | (terms == jnp.inf)
    terms = jnp.where(mask, 0., terms)
    sum_terms = jnp.sum(terms)
    return sum_terms, terms


@jax.jit
def shuffle_samples(key, x, theta, xi):
    num_samples = x.shape[0]
    shuffled_indices = jax.random.permutation(key, num_samples)
    return x[shuffled_indices], theta[shuffled_indices], xi[shuffled_indices]


@jax.jit
def inverse_normalize_xi(normalized_x):
    # Inverse of the probit function (normal CDF)
    scaled_x = jax.scipy.stats.norm.cdf(normalized_x)
    # Rescale back to original range
    x = scaled_x * 100.0
    return x


@partial(jax.jit, static_argnums=[3,5,6])
def lf_pce_eig_scan_lin_reg_new(
                            flow_params: hk.Params, 
                            xi_params: hk.Params, 
                            prng_key: PRNGKey,
                            log_prob_fun: Callable, 
                            designs: Array, 
                            N: int=100, 
                            M: int=10, 
                            lam: float=0.5
    ):
    """
    Calculates LF-PCE loss using jax.lax.scan to accelerate.
    """
    keys = jrandom.split(prng_key, 1 + M)
    
    xi = jnp.broadcast_to(xi_params['xi'], (N, xi_params['xi'].shape[-1]))
    
    # simulate the outcomes before finding their log_probs
    # `designs` are combos of previous designs and proposed (non-scaled) designs
    x, theta_0, x_noiseless, noise = sim_linear_data_vmap(designs, N, keys[0])
    
    scaled_x = standard_scale(x)
    x_mean, x_std = jnp.mean(x), jnp.std(x) + 1e-10
    # If this is the wrong shape, grads don't flow :(
    if len(scaled_x.shape) > 2:
        scaled_x = scaled_x.squeeze(0)
    
    if scaled_x.shape[-1] == 1:
        def compute_marginal_lp(keys, log_prob_fun, M, theta_0, x, conditional_lp):
            def scan_fun(carry, i):
                contrastive_lps, theta = carry
                # theta = jnp.roll(theta_0, shift=1, axis=0)
                theta, _ = sim_linear_prior(N, keys[i % len(keys)])
                contrastive_lp = log_prob_fun(flow_params, x, theta, xi)
                contrastive_lp = jnp.logaddexp(contrastive_lps, contrastive_lp)
                return (contrastive_lp, theta), i + 1
            
            initial_carry = (conditional_lp, theta_0)

            result = jax.lax.scan(scan_fun, initial_carry, jnp.array(range(M)))
            return result[0][0]
        
        conditional_lp = log_prob_fun(flow_params, scaled_x, theta_0, xi)
    else:
        def compute_marginal_lp(keys, log_prob_fun, M, theta, x, conditional_lp):
            def scan_fun(carry, i):
                contrastive_lps, theta = carry
                # theta = jnp.roll(theta, shift=1, axis=0)
                theta, _ = sim_linear_prior(N, keys[i % len(keys)])
                contrastive_lp = jax.vmap(log_prob_fun, in_axes=(None, -1, None, -1))(
                    flow_params, x[:, jnp.newaxis], theta, xi[:, jnp.newaxis])
                contrastive_lp = jnp.sum(contrastive_lp, axis=0)
                contrastive_lp = jnp.logaddexp(contrastive_lps, contrastive_lp)
                return (contrastive_lp, theta), i + 1
            
            initial_carry = (conditional_lp, theta_0)

            result = jax.lax.scan(scan_fun, initial_carry, jnp.array(range(M)))
            return result[0][0]

        conditional_lp = jax.vmap(log_prob_fun, in_axes=(None, -1, None, -1))(
            flow_params, scaled_x[:, jnp.newaxis], theta_0, xi[:, jnp.newaxis])
        conditional_lp = jnp.sum(conditional_lp, axis=0)

    marginal_lp = compute_marginal_lp(
        keys[1:M+2], log_prob_fun, M, theta_0, scaled_x, conditional_lp
        ) - jnp.log(M + 1)
    
    # EIG = jnp.sum(conditional_lp - marginal_lp)
    EIG, EIGs = _safe_mean_terms(conditional_lp - marginal_lp)
    
    loss = EIG + lam * jnp.mean(conditional_lp)
    
    return -loss , (conditional_lp, theta_0, x, x_noiseless, noise, EIG, x_mean, x_std)


@partial(jax.jit, static_argnums=[8,9,10,11,12,13,14])
def lf_pce_design_dist_sir(
    flow_params: hk.Params, 
    xi_params_scaled: hk.Params,
    prng_key: PRNGKey,
    final_ys: Array,
    sde_dict_ts: Array,
    theta_0: Array,
    prev_data: Array = None,
    prev_designs: Array = None,
    log_prob_fun: Callable = None,
    N: int=100, 
    M: int=10, 
    lam: float=0.5,
    design_min: float=0.01,
    design_max: float=100.,
    use_design_dist: bool=True,
    ):
    """
    Calculates LF-PCE loss using jax.lax.scan to accelerate. Takes the previously
    simulated outputs from designs, broadcasts the xi design params, and uses
    the log_prob_fun to calculate the EIG and amortized density.

    Optionally has a probability distribution on designs.
    """
    prng_key, shuffle_key = jrandom.split(prng_key)
    keys = jrandom.split(prng_key, 2 + M)
    # Unnormalize designs to get proper output values
    if use_design_dist:
        xi_params = {k: jnp.exp(v) for k, v in xi_params_scaled.items() if k in ['xi_mu', 'xi_stddev']}
        # xi_params = {k: inverse_normalize_xi(v) for k, v in xi_params_scaled.items() if k in ['xi_mu', 'xi_stddev']}
        a, b = (design_min - xi_params['xi_mu']) / xi_params['xi_stddev'], (design_max - xi_params['xi_mu']) / xi_params['xi_stddev']
        d_sim = xi_params['xi_mu'] + xi_params['xi_stddev'] * jrandom.truncated_normal(keys[0], a, b, shape=(N,1))
    else:
        d_sim = jnp.broadcast_to(xi_params_scaled['xi_mu'], (N,1))
    
    sim_x, x_mean, x_std = simulate_sir(d_sim,
                                        sde_dict_ts,
                                        final_ys)
    
    # Scale designs for conditional flow
    xi = d_sim

    # manually shuffle the theta_0 and corresponding data points
    sim_x, theta_0, xi = shuffle_samples(shuffle_key, sim_x, theta_0, xi)

    if len(sim_x.shape) > 2:
        sim_x = sim_x.squeeze(0)

    # Prepare data by concatenating previous observations if any
    if prev_data is not None:
        x_combined = jnp.concatenate([prev_data, sim_x], axis=1)
        xi_combined = jnp.concatenate([prev_designs, xi], axis=1)
        xi = xi_combined
    else:
        x_combined = sim_x
    
    if prev_data is None:
        def compute_marginal_lp(keys, log_prob_fun, M, theta_0, x, conditional_lp):
            def scan_fun(carry, i):
                contrastive_lps, theta = carry
                theta = jnp.roll(theta, shift=1, axis=0)
                contrastive_lp = log_prob_fun(flow_params, keys[i], x, theta, xi)
                contrastive_lp = jnp.logaddexp(contrastive_lps, contrastive_lp)
                return (contrastive_lp, theta), i + 1
            
            initial_carry = (conditional_lp, theta_0)

            result = jax.lax.scan(scan_fun, initial_carry, jnp.array(range(M)))
            return result[0][0]
        conditional_lp = log_prob_fun(flow_params, keys[0], sim_x, theta_0, xi)
    else:
        def compute_marginal_lp(keys, log_prob_fun, M, theta_0, x, conditional_lp):
            def scan_fun(carry, i):
                prng_keys = jrandom.split(keys[i], num=x.shape[1])
                contrastive_lps, theta = carry
                theta = jnp.roll(theta, shift=1, axis=0)
                contrastive_lp = jax.vmap(log_prob_fun, in_axes=(None, 0, -1, None, -1))(
                    flow_params, prng_keys, x[:, jnp.newaxis], theta, xi[:, jnp.newaxis])
                contrastive_lp = jnp.sum(contrastive_lp, axis=0)
                contrastive_lp = jnp.logaddexp(contrastive_lps, contrastive_lp)
                return (contrastive_lp, theta), i + 1
            
            initial_carry = (conditional_lp, theta_0)

            result = jax.lax.scan(scan_fun, initial_carry, jnp.array(range(M)))
            return result[0][0]
        
        prng_keys = jrandom.split(keys[0], num=x_combined.shape[1])
        conditional_lp = jax.vmap(log_prob_fun, in_axes=(None, 0, -1, None, -1))(
            flow_params, prng_keys, x_combined[:, jnp.newaxis], theta_0, xi[:, jnp.newaxis])
        conditional_lp = jnp.sum(conditional_lp, axis=0)
    
    marginal_lp = compute_marginal_lp(
        keys[:M+1], log_prob_fun, M, theta_0, x_combined, conditional_lp
        ) - jnp.log(M + 1)
    
    EIG, EIGs = _safe_mean_terms(conditional_lp - marginal_lp)
    
    loss = EIG + lam * jnp.mean(conditional_lp)
    
    return -loss , (conditional_lp, EIG, EIGs, x_mean, x_std, xi)


@partial(jax.jit, static_argnums=[7,8,9,10,11,12,13,14,15,16])
def lf_ace_design_dist_sir(flow_params: hk.Params, 
                           post_params: hk.Params, 
                           xi_params_scaled: hk.Params,
                           prng_key: PRNGKey,
                           final_ys: Array,
                            sde_dict_ts: Array,
                            theta_0: Array, 
                            likelihood_lp_fun: Callable, 
                            prior_lp_fun: Callable,
                            prior_sample_fun: Callable,
                            post_lp_fun: Callable,
                            post_sample_fun: Callable,
                            N: int=100, 
                            M: int=10, 
                            lam: float=0.5,
                            design_min: float=0.01,
                            design_max: float=100.,
                            sbc_samples: int=32,
                            sbc_lambda: float=1.,
                            ):
    """
    Calculates LF-ACE loss using jax.lax.scan to accelerate. Only requires a likelihood
    and posterior. 

    The "prior_lp_fun" needs to be passed in with the 
    """
    keys = jrandom.split(prng_key, 3 + M)

    # Unnormalize designs to get proper output values
    xi_params = {k: jnp.multiply(v, 100.) for k, v in xi_params_scaled.items() if k in ['xi_mu', 'xi_stddev']}
    a, b = (design_min - xi_params['xi_mu']) / xi_params['xi_stddev'], (design_max - xi_params['xi_mu']) / xi_params['xi_stddev']
    d_sim = xi_params['xi_mu'] + xi_params['xi_stddev'] * jrandom.truncated_normal(keys[0], a, b, shape=(N,1))
    
    # Scale for conditional flow
    xi = d_sim/100.

    scaled_x, x_mean, x_std = simulate_sir(d_sim,
                                           sde_dict_ts, 
                                           final_ys/100.)
    
    if len(scaled_x.shape) > 2:
        scaled_x = scaled_x.squeeze(0)
    
    def compute_marginal_lp(keys, M, theta_0, x, conditional_lp):
        def scan_fun(carry, i):
            contrastive_lps = carry
            theta, _ = post_sample_fun(post_params, keys[i], N, x)
            contrastive_prior = prior_lp_fun(theta)
            contrastive_likelihood = likelihood_lp_fun(flow_params, x, theta, xi)
            contrastive_posterior = post_lp_fun(
                # jax.lax.stop_gradient(post_params), theta, x)
                post_params, theta, x)
            contrastive_lp = (contrastive_prior + contrastive_likelihood) - contrastive_posterior
            contrastive_lp = jnp.logaddexp(contrastive_lps, contrastive_lp)
            return (contrastive_lp), i + 1
        
        contrastive_prior = prior_lp_fun(theta_0)
        contrastive_likelihood = likelihood_lp_fun(flow_params, x, theta_0, xi)
        contrastive_posterior = post_lp_fun(
            # jax.lax.stop_gradient(post_params), theta_0, x)
            post_params, theta_0, x)
        conditional_lp = contrastive_prior + contrastive_likelihood - contrastive_posterior
        initial_carry = (conditional_lp)

        result = jax.lax.scan(scan_fun, initial_carry, jnp.array(range(M)))
        return result[0]
    
    conditional_lp = likelihood_lp_fun(flow_params, scaled_x, theta_0, xi)
    
    # BUG: This should be shape [512,1]
    marginal_lp = compute_marginal_lp(
        keys[1:M+1], M, theta_0, scaled_x, conditional_lp
        ) - jnp.log(M + 1)
    
    EIG, EIGs = _safe_mean_terms(conditional_lp - marginal_lp)
    
    loss = EIG + lam * jnp.mean(conditional_lp)# + sbc_lambda * sbc_regularization

    theta, contrastive_posterior = post_sample_fun(post_params, keys[0], N, scaled_x)
    prior_lp = prior_lp_fun(theta_0)
    contrastive_prior = prior_lp_fun(theta)
    contrastive_likelihood = likelihood_lp_fun(flow_params, scaled_x, theta_0, xi)
    posterior_lp = post_lp_fun(
        jax.lax.stop_gradient(post_params), theta_0, scaled_x)
    contrastive_lp = (prior_lp + contrastive_likelihood) - posterior_lp
    prior_post_diff = prior_lp - posterior_lp
    prior_post_cont = contrastive_prior - contrastive_posterior
    # best_design_i = jnp.argmax(EIGs)
    # jax.debug.breakpoint()
    
    return -loss , (conditional_lp, EIG, EIGs, x_mean, x_std, d_sim, prior_post_diff, prior_post_cont)


@partial(jax.jit, static_argnums=[8,9,10,11,12])
def lf_pce_design_dist_bmp(
    flow_params: hk.Params, 
    xi_params_scaled: hk.Params,
    static_designs: Array,
    static_outputs: Array,
    prng_key: PRNGKey,
    design_prng_key: PRNGKey,
    theta_0: Array, 
    scaled_x: Array,
    log_prob_fun: Callable,
    N: int=100, 
    M: int=10, 
    lam: float=0.5,
    train: bool=True,
    ):
    """
    Calculates LF-PCE loss using jax.lax.scan to accelerate. Takes the previously
    simulated outputs from designs, broadcasts the xi design params, and uses
    the log_prob_fun to calculate the EIG and amortized density.

    Has a probability distribution on designs. Can set the bounds of the truncated normal.

    static_designs: Pass in 
    """
    prng_key, shuffle_key = jrandom.split(prng_key)
    keys = jrandom.split(prng_key, 2 + M)

    if train:
        # Unnormalize designs to get proper design dist values
        xi_params = {k: jnp.exp(v) for k, v in xi_params_scaled.items() if k in ['xi_mu', 'xi_stddev']}
        a, b = (0. - xi_params['xi_mu']) / xi_params['xi_stddev'], (1000. - xi_params['xi_mu']) / xi_params['xi_stddev']
        d_sim = xi_params['xi_mu'] + xi_params['xi_stddev'] * jrandom.truncated_normal(design_prng_key, a, b, shape=(N,1))
        
        # If combining with previous static designs
        xi = d_sim
    else:
        xi = jnp.broadcast_to(xi_params_scaled['xi_mu'], (N,1))
    
    # scaled_x now shape (1, N)
    if len(scaled_x.shape) > 2:
        scaled_x = scaled_x.squeeze(0)

    # TODO: standardize the use of naming x's as sim_x since scaled_x is an artifact and normalization happens in log_prob
    scaled_x, theta_0, xi = shuffle_samples(shuffle_key, scaled_x, theta_0, xi)
        
    if static_designs is None:
        def compute_marginal_lp(
                keys, log_prob_fun, M, theta_0, x, conditional_lp):
            def scan_fun(carry, i):
                contrastive_lps, theta = carry
                theta = jnp.roll(theta, shift=1, axis=0)
                contrastive_lp = log_prob_fun(flow_params, scaled_x, theta, xi)
                contrastive_lp = jnp.logaddexp(contrastive_lps, contrastive_lp)
                return (contrastive_lp, theta), i + 1
            
            initial_carry = (conditional_lp, theta_0)

            result = jax.lax.scan(scan_fun, initial_carry, jnp.array(range(M)))
            return result[0][0]
    
        conditional_lp = log_prob_fun(flow_params, scaled_x, theta_0, xi)
        marginal_lp = compute_marginal_lp(
            keys[1:M+1], log_prob_fun, M, theta_0, scaled_x, conditional_lp
            ) - jnp.log(M + 1)
    else:
        def compute_marginal_lp(
                keys, log_prob_fun, M, theta_0, conditional_lp):
            def scan_fun(carry, i):
                contrastive_lps, theta = carry
                theta = jnp.roll(theta, shift=1, axis=0)
                contrastive_lp = log_prob_fun(
                    flow_params, new_static_outputs[:, None], theta, new_static_designs[:, None]).T.sum(1)
                contrastive_lp = jnp.logaddexp(contrastive_lps, contrastive_lp)
                return (contrastive_lp, theta), i + 1
            
            initial_carry = (conditional_lp, theta_0)
            result = jax.lax.scan(scan_fun, initial_carry, jnp.array(range(M)))
            return result[0][0]

        new_static_outputs = jnp.concatenate([static_outputs, scaled_x], axis=1)
        new_static_designs = jnp.concatenate([static_designs, xi], axis=1)

        vmap_fun = jax.vmap(log_prob_fun, in_axes=(None, -1, None, -1))
        conditional_lp = vmap_fun(
            flow_params, new_static_outputs[:, None], theta_0, new_static_designs[:, None]).T.sum(1)
    
        marginal_lp = compute_marginal_lp(
            keys[1:M+1], vmap_fun, M, theta_0, conditional_lp
            ) - jnp.log(M + 1)
    
    EIG, EIGs = _safe_mean_terms(conditional_lp - marginal_lp)
    
    loss = EIG + lam * jnp.mean(conditional_lp)
    
    return -loss , (conditional_lp, EIG, EIGs, xi)


@partial(jax.jit, static_argnums=[5,6,7,8])
def lf_pce_eig_scan(flow_params: hk.Params, 
                    static_designs: Array,
                    prng_key: PRNGKey,
                    scaled_x: Array,
                    theta_0: Array, 
                    log_prob_fun: Callable, 
                    N: int=100, 
                    M: int=10, 
                    lam: float=0.5,
                    ):
    """
    Calculates LF-PCE loss using jax.lax.scan to accelerate. Takes the previously
    simulated outputs from designs, broadcasts the xi design params, and uses
    the log_prob_fun to calculate the EIG and amortized density.

    """
    keys = jrandom.split(prng_key, 2 + M)

    xi = static_designs
    
    if len(scaled_x.shape) > 2:
        scaled_x = scaled_x.squeeze(0)
    
    if scaled_x.shape[-1] == 1:
        def compute_marginal_lp(
                keys, log_prob_fun, M, theta_0, x, conditional_lp):
            def scan_fun(carry, i):
                contrastive_lps, theta = carry
                theta = jnp.roll(theta, shift=1, axis=0)
                contrastive_lp = log_prob_fun(flow_params, x, theta, xi)
                contrastive_lp = jnp.logaddexp(contrastive_lps, contrastive_lp)
                return (contrastive_lp, theta), i + 1
            
            initial_carry = (conditional_lp, theta_0)

            result = jax.lax.scan(scan_fun, initial_carry, jnp.array(range(M)))
            return result[0][0]
        
        conditional_lp = log_prob_fun(flow_params, scaled_x, theta_0, xi)
    else:
        def compute_marginal_lp(
                keys, log_prob_fun, M, theta_0, x, conditional_lp):
            def scan_fun(carry, i):
                contrastive_lps, theta = carry
                theta = jnp.roll(theta, shift=1, axis=0)
                contrastive_lp = jax.vmap(log_prob_fun, in_axes=(None, -1, None, -1))(
                    flow_params, x[:, jnp.newaxis], theta, xi[:, jnp.newaxis])
                contrastive_lp = jnp.sum(contrastive_lp, axis=0)
                contrastive_lp = jnp.logaddexp(contrastive_lps, contrastive_lp)
                return (contrastive_lp, theta), i + 1
            
            initial_carry = (conditional_lp, theta_0)

            result = jax.lax.scan(scan_fun, initial_carry, jnp.array(range(M)))
            return result[0][0]
        
        # BUG: Issue here with concatenation of a static design with optimizeable design
        conditional_lps = jax.vmap(log_prob_fun, in_axes=(None, -1, None, -1))(
            flow_params, scaled_x[:, jnp.newaxis], theta_0, xi[:, jnp.newaxis])
        conditional_lp = jnp.sum(conditional_lps, axis=0)

    marginal_lp = compute_marginal_lp(
        keys[1:M+1], log_prob_fun, M, theta_0, scaled_x, conditional_lp
        ) - jnp.log(M + 1)

    EIG, EIGs = _safe_mean_terms(conditional_lp - marginal_lp)
    
    loss = EIG + lam * jnp.mean(conditional_lp)
    
    return -loss , (conditional_lp, EIG)


def _logmeanexp(a: Array, axis: int) -> Array:
    return logsumexp(a, axis=axis) - jnp.log(a.shape[axis])


@partial(jax.jit, static_argnums=[5, 6, 7, 8])
def lf_epig_scan(
    flow_params: hk.Params,
    static_designs: Array,
    prng_key: PRNGKey,
    theta_candidates: Array,   # (C, theta_dim) or (theta_dim,)
    theta_targets: Array,      # (J, theta_dim) or (theta_dim,)
    log_prob_fun: Callable,    # (params, y, theta, xi, dropout_key) -> log p(y|theta,xi)
    sample_fun: Callable,      # (params, theta, xi, sample_key, dropout_key) -> y ~ p_{phi(dropout)}(.|theta,xi)
    K: int = 32,               # number of dropout "model particles" phi^(k)
    S: int = 1,                # # (y, y*) pairs per model particle (usually 1 is fine)
) -> Tuple[Array, Array]:
    """
    EPIG acquisition score for simulator-parameter active learning using ONLY likelihood evals,
    with epistemic uncertainty approximated via MC-dropout.

    EPIG(theta) = E_{theta* ~ p*(theta*)} KL( p(y,y*|theta,theta*) || p(y|theta)p(y*|theta*) )

    Dropout key indexes the model particle: phi^(k).
    We sample (y, y*) from the JOINT predictive by using the *same* dropout particle k for both.

    Returns:
        loss:    -EPIG scores (so you can minimize)
        scores:  EPIG scores
    """
    xi = static_designs

    # Normalize shapes
    squeeze_cand = False
    if theta_candidates.ndim == 1:
        theta_candidates = theta_candidates[jnp.newaxis, :]
        squeeze_cand = True
    if theta_targets.ndim == 1:
        theta_targets = theta_targets[jnp.newaxis, :]

    C = theta_candidates.shape[0]
    J = theta_targets.shape[0]
    G = K * S  # total joint samples per target: (k,s)

    cand_keys = jrandom.split(prng_key, C)

    def epig_one_candidate(theta_c: Array, key_c: PRNGKey) -> Array:
        # One key per candidate -> split into (dropout particles, y sampling, per-target sampling)
        key_drop, key_y, key_t = jrandom.split(key_c, 3)

        # Dropout particles: each key defines one "phi^(k)"
        drop_keys = jrandom.split(key_drop, K)  # (K, 2)

        # --- Sample y from each dropout particle at theta_c ---
        y_keys = jrandom.split(key_y, G).reshape(K, S, 2)  # (K,S,2)

        def _sample_S(theta: Array, drop_key: PRNGKey, keys_S: Array) -> Array:
            # keys_S: (S,2)
            return jax.vmap(lambda sk: sample_fun(flow_params, theta, xi, sk, drop_key))(keys_S)

        # y_samples: (K,S, y_dim...)
        y_samples = jax.vmap(lambda dk, ks: _sample_S(theta_c, dk, ks))(drop_keys, y_keys)
        y_flat = y_samples.reshape((G,) + y_samples.shape[2:])  # (G, y_dim...)

        # logp_y: (K_components, G_samples)
        def _lp_component_y(drop_key_component: PRNGKey) -> Array:
            return jax.vmap(lambda y: log_prob_fun(flow_params, y, theta_c, xi, drop_key_component))(y_flat)

        logp_y = jax.vmap(_lp_component_y)(drop_keys)  # (K, G)
        log_marg_y = _logmeanexp(logp_y, axis=0)       # (G,)

        # --- For each target theta*, estimate KL(joint || product) using the same dropout particles ---
        target_keys = jrandom.split(key_t, J)  # one RNG per target (controls only base-noise for y*)

        def kl_one_target(theta_star: Array, key_star: PRNGKey) -> Array:
            # Sample y* from each dropout particle at theta_star
            ystar_keys = jrandom.split(key_star, G).reshape(K, S, 2)  # (K,S,2)
            ystar_samples = jax.vmap(lambda dk, ks: _sample_S(theta_star, dk, ks))(drop_keys, ystar_keys)
            ystar_flat = ystar_samples.reshape((G,) + ystar_samples.shape[2:])  # (G, y_dim...)

            # logp_star: (K_components, G_samples)
            def _lp_component_star(drop_key_component: PRNGKey) -> Array:
                return jax.vmap(lambda y: log_prob_fun(flow_params, y, theta_star, xi, drop_key_component))(ystar_flat)

            logp_star = jax.vmap(_lp_component_star)(drop_keys)  # (K, G)

            log_marg_star = _logmeanexp(logp_star, axis=0)       # (G,)
            log_joint = _logmeanexp(logp_y + logp_star, axis=0)  # (G,)

            # MC estimate of KL via samples from joint predictive
            r = log_joint - log_marg_y - log_marg_star           # (G,)
            return jnp.mean(r)

        kl_targets = jax.vmap(kl_one_target)(theta_targets, target_keys)  # (J,)
        return jnp.mean(kl_targets)  # EPIG(theta_c)

    scores = jax.vmap(epig_one_candidate)(theta_candidates, cand_keys)  # (C,)

    if squeeze_cand:
        scores = scores.squeeze(0)

    return -scores, scores


@partial(jax.jit, static_argnums=[3, 5, 6])
def lf_pce_eig_scan_lin_reg(
    flow_params: hk.Params,
    xi_params: hk.Params,
    prng_key: PRNGKey,
    log_prob_fun: Callable,
    designs: Array,
    N: int = 100,
    M: int = 10,
    lam: float = 0.5,
):
    """
    Calculates LF-PCE loss using jax.lax.scan to accelerate.
    """

    def compute_marginal_lp(keys, log_prob_fun, M, N, x, conditional_lp):
        def scan_fun(contrastive_lps, i):
            theta, _ = sim_linear_prior(N, keys[i + 1])
            contrastive_lp = log_prob_fun(flow_params, x, theta, xi)
            return jnp.logaddexp(contrastive_lps, contrastive_lp), i + 1

        result = jax.lax.scan(scan_fun, conditional_lp, jnp.array(range(M)))
        return result[0]

    keys = jrandom.split(prng_key, 1 + M)

    xi = jnp.broadcast_to(xi_params["xi"], (N, xi_params["xi"].shape[-1]))

    # simulate the outcomes before finding their log_probs
    # `designs` are combos of previous designs and proposed (non-scaled) designs
    x, theta_0, x_noiseless, noise = sim_linear_data_vmap(designs, N, keys[0])
    
    scaled_x = standard_scale(x)
    x_mean, x_std = jnp.mean(x), jnp.std(x) + 1e-10
    # If this is the wrong shape, grads don't flow :(
    if len(scaled_x.shape) > 2:
        scaled_x = scaled_x.squeeze(0)
    # Shape is [N,len(x)]
    conditional_lp = log_prob_fun(flow_params, scaled_x, theta_0, xi)
    marginal_lp = compute_marginal_lp(
        keys[1 : M + 1], log_prob_fun, M, N, scaled_x, conditional_lp
    ) - jnp.log(M + 1)

    EIG, EIGs = _safe_mean_terms(conditional_lp - marginal_lp)

    loss = EIG + lam * jnp.mean(conditional_lp)

    return -loss, (conditional_lp, theta_0, x, x_noiseless, noise, EIG, x_mean, x_std)


@partial(jax.jit, static_argnums=[3,6,7,8])
def snpe_c(post_params: hk.Params, xi_params: hk.Params, prng_key: PRNGKey, 
           prior: Callable, scaled_x: Array, theta_0: Array,
           post_log_prob_fun: Callable, N: int=100, M: int=10, lam: float=0.5):
    """
    Calculates NP-PCE loss using jax.lax.scan to accelerate. Requires a likelihood
    log_prob function and a prior. Will use to calculate the EIG and amortized density.
    """
    def compute_snpe_marginal_lp(keys, prior, post_log_prob_fun, M, N, x, conditional_lp):
        def scan_fun(contrastive_lps, i):
            # TODO: Make sample_shape adapt to passed prior instead of pre-specified shape.
            # TODO: Conditional statement if prior is a flow
            thetas, prior_lp = prior.sample_and_log_prob(seed=keys[i+1], sample_shape=(N,))
            contrastive_lp = post_log_prob_fun(post_params, thetas, x, xi)
            contrastive_lp = contrastive_lp - prior_lp
            return jnp.logaddexp(contrastive_lps, contrastive_lp), i + 1

        result = jax.lax.scan(scan_fun, conditional_lp, jnp.array(range(M)))
        return result[0]
    
    keys = jrandom.split(prng_key, 2 + M)
    
    # Broadcast scaled xi design params & initial priors
    xi = jnp.broadcast_to(xi_params['xi'], (N, xi_params['xi'].shape[-1]))
    
    if len(scaled_x.shape) > 2:
        scaled_x = scaled_x.squeeze(0)
    
    conditional_lp = post_log_prob_fun(post_params, theta_0, scaled_x, xi)
    prior_lp = prior.log_prob(theta_0)
    conditional_lp = conditional_lp - prior_lp
    marginal_lp = compute_snpe_marginal_lp(
        keys[1:M+1], prior, post_log_prob_fun, M, N, scaled_x, conditional_lp
        ) - jnp.log(M+1)
    
    # EIG = jnp.sum(conditional_lp - marginal_lp)
    EIG, EIGs = _safe_mean_terms(conditional_lp - marginal_lp)

    loss = EIG + lam * jnp.mean(conditional_lp - prior_lp)
    
    return -loss , (conditional_lp, EIG)


# TODO: update static argnums
@partial(jax.jit, static_argnums=[4, 5, 6, 7, 9, 10])
def lf_ace_eig_scan(flow_params: hk.Params, post_params: hk.Params, xi_params: hk.Params,
           prng_key: PRNGKey, scaled_x: Array, theta_0: Array, prior: Callable,
           log_prob_fun: Callable, post_log_prob_fun: Callable, 
           post_sample_fun: Callable, designs: Array, N: int=100, M: int=10,):
    """
    Calculates snpe-c using a posterior and prior. Requires a posterior and prior
    estimate. Will use all three to calculate the EIG. This takes a vectorized
    approach for readability and GPU compatability.
    """
    def compute_snpe_marginal_lp(keys, log_prob_fun, M, N, x, conditional_lp):
        def scan_fun(contrastive_lps, i):
            # TODO: Make sample_shape adapt to passed prior instead of pre-specified shape.
            theta = post_sample_fun.sample(seed=keys[i+1], sample_shape=(N,2))
            contrastive_lp = log_prob_fun(flow_params, x, theta, xi)
            # TODO conditional statement if prior is a flow
            prior_lp = prior.log_prob(theta)
            numerator = jnp.logaddexp(prior_lp, contrastive_lp)
            post_lp = post_log_prob_fun(post_params, theta, x)
            contrastive_lp = jnp.logaddexp(numerator, - post_lp)
            return jnp.logaddexp(contrastive_lps, contrastive_lp), i + 1

        result = jax.lax.scan(scan_fun, conditional_lp, jnp.array(range(M)))
        return result[0]
    
    keys = jrandom.split(prng_key, 2 + 3 * M)
    
    # Broadcast xi design params & initial priors
    xi = jnp.broadcast_to(xi_params['xi'], (N, xi_params['xi'].shape[-1]))
    
    if len(scaled_x.shape) > 2:
        scaled_x = scaled_x.squeeze(0)

    conditional_lp = log_prob_fun(flow_params, scaled_x, theta_0, xi)
    marginal_lp = compute_snpe_marginal_lp(
        keys[1:M+1], log_prob_fun, M, N, scaled_x, conditional_lp
        ) - jnp.log(M+1)

    EIG, EIGs = _safe_mean_terms(conditional_lp - marginal_lp)

    loss = EIG + jnp.mean(conditional_lp)
    
    return -loss , (conditional_lp, EIG)


@partial(jax.jit, static_argnums=[2,3])
def lfi_pce_eig_fori(params: hk.Params, prng_key: PRNGKey, N: int=100, M: int=10, **kwargs):
    """
    Calculates PCE loss using jax.lax.fori_loop to accelerate. Slightly slower than scan.
    More readable than scan.
    TODO: refactor arguments.
    """
    def compute_marginal(M, num_samples, key, flow_params, x, xi_broadcast, conditional_lp):
        def loop_body_fun2(i, carry):
            contrastive_lps = carry
            theta, _ = sim_linear_prior(num_samples, keys[i + 1])
            contrastive_lp = log_prob.apply(flow_params, x, theta, xi_broadcast)
            contrastive_lps += jnp.exp(contrastive_lp)
            return contrastive_lps
        conditional_lps = jax.lax.fori_loop(0, M, loop_body_fun2, conditional_lp)
        return jnp.log(conditional_lps)
    
    keys = jrandom.split(prng_key, 3 + M)
    xi = params['xi']
    flow_params = {k: v for k, v in params.items() if k != 'xi'}

    # simulate the outcomes before finding their log_probs
    x, theta_0 = sim_linear_data_vmap(d_sim, num_samples, keys[0])
    xi_broadcast = jnp.broadcast_to(xi, (num_samples, len(xi)))

    conditional_lp = log_prob.apply(flow_params, x, theta_0, xi_broadcast)

    # conditional_lp could be the initial starting state that is added upon... 
    marginal_lp = compute_marginal(
        M, num_samples, key, flow_params, x, xi_broadcast, conditional_lp
        ) - jnp.log(M+1)

    return - sum(conditional_lp - marginal_lp) - jnp.mean(conditional_lp)


@partial(jax.jit, static_argnums=[2,3])
def lfi_pce_eig_vmap_distrax(params: hk.Params, prng_key: PRNGKey, N: int=100, M: int=10, **kwargs):
    """
    Calculates PCE loss using vmap inherent to `distrax` distributions. May be faster
    than scan on GPUs.
    TODO: refactor arguments.
    """
    keys = jrandom.split(prng_key, 2)
    xi = params['xi']
    flow_params = {k: v for k, v in params.items() if k != 'xi'}

    # simulate the outcomes before finding their log_probs
    x, theta_0 = sim_linear_data_vmap(d_sim, num_samples, keys[0])

    xi_broadcast = jnp.broadcast_to(xi, (num_samples, len(xi)))

    conditional_lp = log_prob.apply(flow_params, x, theta_0, xi_broadcast)

    # TODO: Make function that returns M x num_samples priors
    thetas, log_probs = sim_linear_prior_M_samples(num_samples=num_samples, M=M, key=keys[1])
    
    # conditional_lp could be the initial starting state that is added upon... 
    contrastive_lps = jax.vmap(lambda theta: log_prob.apply(params, x, theta, xi_broadcast))(thetas)
    marginal_log_prbs = jnp.concatenate((jax_lexpand(conditional_lp, 1), jnp.array(contrastive_lps)))
    marginal_lp = jax.nn.logsumexp(marginal_log_prbs, 0) - math.log(M + 1)
    # marginal_lp = compute_marginal_lp3(M, num_samples, key, flow_params, x, xi_broadcast, conditional_lp)

    return - sum(conditional_lp - marginal_lp) - jnp.mean(conditional_lp)


@partial(jax.jit, static_argnums=[2,3])
def lfi_pce_eig_vmap_manual(params: hk.Params, prng_key: PRNGKey, N: int=100, M: int=10, **kwargs):
    """
    Calculates PCE loss using explicit vmap of `distrax` distributions. May potentially
    be more stable than using `ditrax` implicit version as of 2/9/23. May be faster
    than scan on GPUs.
    TODO: refactor arguments.
    """
    keys = jrandom.split(prng_key, M + 1)
    xi = params['xi']
    flow_params = {k: v for k, v in params.items() if k != 'xi'}

    # simulate the outcomes before finding their log_probs
    x, theta_0 = sim_linear_data_vmap(d_sim, num_samples, keys[0])

    xi_broadcast = jnp.broadcast_to(xi, (num_samples, len(xi)))

    conditional_lp = log_prob.apply(flow_params, x, theta_0, xi_broadcast)

    thetas, log_probs = jax.vmap(partial(sim_linear_prior, num_samples))(keys[1:M+1])
    
    # conditional_lp could be the initial starting state that is added upon... 
    contrastive_lps = jax.vmap(lambda theta: log_prob.apply(params, x, theta, xi_broadcast))(thetas)
    marginal_log_prbs = jnp.concatenate((jax_lexpand(conditional_lp, 1), jnp.array(contrastive_lps)))
    marginal_lp = jax.nn.logsumexp(marginal_log_prbs, 0) - math.log(M + 1)
    # marginal_lp = compute_marginal_lp3(M, num_samples, key, flow_params, x, xi_broadcast, conditional_lp)

    return - sum(conditional_lp - marginal_lp) - jnp.mean(conditional_lp)

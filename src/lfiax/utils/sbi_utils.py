from functools import partial

import jax
import jax.numpy as jnp
import jax.lax as lax
import jax.random as jrandom
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.stats import binom
from scipy.stats import t
import distrax
from lfiax.flows.nsf import make_nsf

from typing import List, Optional, Tuple, Union

import haiku as hk
import numpy as np
from tqdm.auto import tqdm

from typing import (
    Any,
    Callable,
    Sequence,
    Union,
    Iterator,
    Mapping,
    Optional,
    Tuple,
)

Array = jnp.ndarray
Batch = Mapping[str, np.ndarray]
PRNGKey = Array


# ------------ SBI training helper functions ------------
def early_stopping_condition(current_loss, best_loss, no_improvement_epochs, early_stopping_limit=20):
    if current_loss < best_loss:
        return current_loss, 0, False
    elif no_improvement_epochs >= early_stopping_limit:
        return best_loss, no_improvement_epochs, True
    else:
        return best_loss, no_improvement_epochs + 1, False

# ------------ SBC helper functions ------------
def constrained_to_unconstrained(x):
    """Map from [-1, 1] to unconstrained space."""
    x_clipped = jnp.clip(x, -0.9999999, 0.9999999)
    return jax.scipy.special.logit((x_clipped + 1) / 2)

def unconstrained_to_constrained(x):
    """Map from unconstrained space to [-1, 1]."""
    return 2 * jax.scipy.special.expit(x) - 1

def constrained_to_unconstrained_logdetjac(x):
    """Log determinant of the Jacobian for the constrained to unconstrained transformation."""
    jac_diag = jax.vmap(jax.grad(constrained_to_unconstrained))(x)
    return jnp.sum(jnp.log(jnp.abs(jac_diag)))

def unconstrained_to_constrained_logdetjac(x):
    """Log determinant of the Jacobian for the unconstrained to constrained transformation."""
    jac_diag = jax.vmap(jax.grad(unconstrained_to_constrained))(x)
    return jnp.sum(jnp.log(jnp.abs(jac_diag)))

# ------------ Posterior checks ------------
def run_sbc_two_moons(
    prior_samples,
    observed_data,
    log_likelihood_fn,
    log_prior_fn,
    mcmc_fn,
    num_adapt_steps,
    num_mcmc_samples,
    theta_train_mean,
    theta_train_std,
    params,
    prng_seq,
    mcmc_posterior,
    alpha: float = 0.05, 
    show_progress=True,
    **kwargs
):
    """
    Run SBC for the given prior samples and observed data.

    Args:
        prior_samples: Array of shape (num_simulations, theta_dim)
        observed_data: Array of shape (num_simulations, x_dim)
        log_likelihood_fn: Function that computes log-likelihood given params, x, theta
        log_prior_fn: Function that computes log-prior given theta
        mcmc_fn: Function that runs MCMC given posterior function and returns samples
        num_adapt_steps: Number of adaptation steps for MCMC
        num_mcmc_samples: Number of MCMC samples to draw
        theta_train_mean: Mean of theta used for normalization
        theta_train_std: Std of theta used for normalization
        params: Parameters for the log_likelihood_fn (e.g., neural network parameters)
        prng_seq: PRNG key or sequence
        show_progress: Whether to show a progress bar
        **kwargs: Additional arguments for mcmc_fn

    Returns:
        ranks: Array of ranks (num_simulations,)
    """

    num_simulations = prior_samples.shape[0]

    def sbc_body_fn(carry, i):
        ranks, prng_seq = carry

        theta_i = prior_samples[i]
        x_i = observed_data[i]
        mcmc_posterior_i = partial(mcmc_posterior, x_obs=x_i)

        # Run MCMC to get posterior samples
        theta_0 = theta_i  # NOTE: can choose a different initial position if desired
        posterior_samples, post_lps = mcmc_fn(
            prng_seq,
            mcmc_posterior_i,
            theta_0,
            num_adapt_steps,
            num_mcmc_samples,
            **kwargs
        )
        
        # Compute the rank for each dimension
        ranks_i = jnp.sum(posterior_samples.mean(1) < theta_i.mean(), axis=0)
        # Sum ranks across dimensions or handle multi-dimensional theta
        ranks = ranks.at[i].set(ranks_i.sum())

        # Optionally update prng_seq if needed (you can update RNG per step)
        prng_seq = jax.random.split(prng_seq)[0]

        return (ranks, prng_seq), i  # Carry the updated ranks and prng_seq

    # Initial state
    initial_ranks = jnp.zeros(num_simulations, dtype=int)
    carry_initial = (initial_ranks, prng_seq)

    # Run lax.scan
    carry_final, _ = jax.lax.scan(sbc_body_fn, carry_initial, jnp.arange(num_simulations))

    # Extract final ranks
    final_ranks, final_prng_seq = carry_final

    # use ranks to calculate a lot of the metrics to return
    ranks_sorted = jnp.sort(final_ranks)
    empirical_coverage = ranks_sorted / num_mcmc_samples
    levels = (np.arange(1, num_simulations + 1) - 0.5) / num_simulations

    lower_bounds = []
    upper_bounds = []

    for p in levels:
        # Compute the confidence interval for a binomial proportion
        lower = binom.ppf(alpha / 2, num_simulations, p) / num_simulations
        upper = binom.ppf(1 - alpha / 2, num_simulations, p) / num_simulations
        lower_bounds.append(lower)
        upper_bounds.append(upper)

    # Convert to numpy arrays
    lower_bounds = np.array(lower_bounds)
    upper_bounds = np.array(upper_bounds)

    # Ensure bounds are within [0,1]
    lower_bounds = np.clip(lower_bounds, 0, 1)
    upper_bounds = np.clip(upper_bounds, 0, 1)

    return final_ranks, empirical_coverage, levels, lower_bounds, upper_bounds

def run_sbc(
    prior_samples,
    observed_data,
    mcmc_fn,
    num_warmup_steps,
    num_mcmc_samples,
    prng_seq,
    prior_lp,
    log_likelihood_fn,
    mcmc_posterior,
    alpha: float = 0.05, 
    **kwargs
):
    """
    Run SBC for the given prior samples and observed data.

    Args:
        prior_samples: Array of shape (num_simulations, theta_dim)
        observed_data: Array of shape (num_simulations, x_dim)
        mcmc_fn: Function that runs MCMC given posterior function and returns samples
        num_warmup_steps: Number of warmup steps for MCMC
        num_mcmc_samples: Number of MCMC samples to draw
        params: Parameters for the log_likelihood_fn (e.g., neural network parameters)
        prng_seq: PRNG key or sequence
        show_progress: Whether to show a progress bar
        **kwargs: Additional arguments for mcmc_fn

    Returns:
        ranks: Array of ranks (num_simulations,)
    """
    num_simulations = prior_samples.shape[0]
    observed_data = jnp.array(observed_data)
    def sbc_body_fn(carry, i):
        ranks, prng_seq = carry

        theta_i = prior_samples[i]
        x_i = observed_data[i][None, None, :]
        log_likelihood_fn_i = partial(log_likelihood_fn, x=x_i)
        mcmc_posterior_i = partial(mcmc_posterior, x=x_i)

        # Run MCMC to get posterior samples
        theta_0 = theta_i[None, :]
        posterior_samples, _ = mcmc_fn(
            prng_seq,
            prior_lp,
            log_likelihood_fn_i,
            mcmc_posterior_i,
            theta_0,
            num_mcmc_samples,
            num_warmup_steps,
            **kwargs
        )
        
        # Compute the rank for each dimension
        ranks_i = jnp.sum(posterior_samples.mean(1) < theta_i.mean(), axis=0)
        # Sum ranks across dimensions or handle multi-dimensional theta
        ranks = ranks.at[i].set(ranks_i.sum())

        # Optionally update prng_seq if needed (you can update RNG per step)
        prng_seq = jax.random.split(prng_seq)[0]

        return (ranks, prng_seq), i

    # Initial state
    initial_ranks = jnp.zeros(num_simulations, dtype=int)
    carry_initial = (initial_ranks, prng_seq)

    carry_final, _ = jax.lax.scan(sbc_body_fn, carry_initial, jnp.arange(num_simulations))

    # Extract final ranks
    final_ranks, _ = carry_final

    # use ranks to calculate a lot of the metrics to return
    ranks_sorted = jnp.sort(final_ranks)
    empirical_coverage = ranks_sorted / num_mcmc_samples
    levels = (np.arange(1, num_simulations + 1) - 0.5) / num_simulations

    lower_bounds = []
    upper_bounds = []

    for p in levels:
        # Compute the confidence interval for a binomial proportion
        lower = binom.ppf(alpha / 2, num_simulations, p) / num_simulations
        upper = binom.ppf(1 - alpha / 2, num_simulations, p) / num_simulations
        lower_bounds.append(lower)
        upper_bounds.append(upper)

    # Convert to numpy arrays
    lower_bounds = np.array(lower_bounds)
    upper_bounds = np.array(upper_bounds)

    # Ensure bounds are within [0,1]
    lower_bounds = np.clip(lower_bounds, 0, 1)
    upper_bounds = np.clip(upper_bounds, 0, 1)

    return final_ranks, empirical_coverage, levels, lower_bounds, upper_bounds



def ks_test(sample1, sample2):
    '''Two-sample KS-test.'''
    sample1_sorted = jnp.sort(sample1)
    sample2_sorted = jnp.sort(sample2)
    sample1_size = sample1.shape[0]
    sample2_size = sample2.shape[0]
    
    data_all = jnp.concatenate([sample1_sorted, sample2_sorted])
    group_indicator = jnp.concatenate([jnp.zeros(sample1_size), jnp.ones(sample2_size)])
    index_sorted = jnp.argsort(data_all)
    
    group_sorted = group_indicator[index_sorted]
    d_plus = jnp.where(group_sorted == 1, 1 / sample2_size, 0)
    d_minus = jnp.where(group_sorted == 0, 1 / sample1_size, 0)
    
    cdf_diff = jnp.cumsum(d_plus - d_minus)
    ks_statistic = jnp.max(jnp.abs(cdf_diff))
    
    # Compute p-value using asymptotic distribution
    n = sample1_size * sample2_size / (sample1_size + sample2_size)
    p_value = np.exp(-2 * n * ks_statistic**2)
    
    return ks_statistic, p_value


def c2st_accuracy(ranks: jnp.ndarray, uniforms: jnp.ndarray, num_folds: int = 5) -> float:
    """
    Perform the Classifier 2-Sample Test (C2ST) using a logistic regression classifier and
    compute the average cross-validated accuracy.

    Args:
        ranks: A JAX numpy array containing the first set of data samples (ranks in SBC).
        uniforms: A JAX numpy array containing the second set of data samples (uniform samples in SBC).
        num_folds: The number of folds to use for cross-validation (default is 5).

    Returns:
        The average cross-validated accuracy of the classifier on the two datasets.
    """
    # Combine the data and create labels
    data_combined = jnp.concatenate([ranks, uniforms])[:, None]
    labels = jnp.concatenate([jnp.zeros(ranks.shape[0]), jnp.ones(uniforms.shape[0])])

    # Define logistic regression model using Haiku
    def logistic_regression_fn(x):
        return hk.Sequential([hk.Linear(1), jax.nn.sigmoid])(x)

    logistic_regression = hk.without_apply_rng(hk.transform(logistic_regression_fn))

    # Cross-validation
    kf = KFold(n_splits=num_folds, shuffle=True)
    accuracy_scores = []

    for train_indices, val_indices in kf.split(data_combined):
        # Split the data into training and validation sets
        x_train, x_val = data_combined[train_indices], data_combined[val_indices]
        y_train, y_val = labels[train_indices], labels[val_indices]

        # Initialize parameters
        params = logistic_regression.init(jrandom.PRNGKey(42), x_train)

        # Define the loss function
        def loss_fn(params, x, y):
            logits = logistic_regression.apply(params, x)
            return -jnp.mean(y * jnp.log(logits) + (1 - y) * jnp.log(1 - logits))

        # Define the gradient function
        grad_fn = jax.value_and_grad(loss_fn)

        # Define the optimizer
        opt = optax.adam(0.01)
        opt_state = opt.init(params)

        # Train the logistic regression model
        for _ in range(500):
            loss, grads = grad_fn(params, x_train, y_train)
            updates, opt_state = opt.update(grads, opt_state)
            params = optax.apply_updates(params, updates)

        # Compute accuracy on the validation set
        logits_val = logistic_regression.apply(params, x_val)
        preds_val = jnp.round(logits_val).squeeze()
        accuracy = jnp.mean(preds_val == y_val)
        accuracy_scores.append(accuracy)

    return jnp.mean(jnp.array(accuracy_scores))


def expected_coverage_probability(sbc_ranks: jnp.ndarray, alpha: float) -> float:
    """
    Calculate the Expected Coverage Probability (ECP) for a given value of alpha.

    Args:
        sbc_ranks: A JAX numpy array containing the SBC ranks.
        alpha: A float value between 0 and 1.

    Returns:
        The Expected Coverage Probability (ECP) as a float.
    """
    num_simulations = sbc_ranks.shape[0]
    num_ranks_exceeding_alpha = jnp.sum(sbc_ranks / num_simulations >= alpha)
    ecp = num_ranks_exceeding_alpha / num_simulations
    return ecp


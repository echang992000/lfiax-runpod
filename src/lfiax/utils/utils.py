from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import jax.lax as lax
import jax.random as jrandom
import blackjax
import blackjax.smc.resampling as resampling
from blackjax.smc import extend_params
import tensorflow_probability.substrates.jax as tfp
import haiku as hk
import distrax
from lfiax.flows.nsf import make_nsf
# from .fast_soft_sort.jax_ops import soft_sort

import matplotlib.pyplot as plt

from typing import List, Optional, Tuple, Union

import tensorflow_datasets as tfds

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


# ------------ training helpers ------------
@jax.jit
def shuffle_samples(key, x, theta, xi):
    num_samples = x.shape[0]
    shuffled_indices = jrandom.permutation(key, num_samples)
    return x[shuffled_indices], theta[shuffled_indices], xi[shuffled_indices]

def split_data_for_validation_jax(x_sbi, thetas_sbi, sbi_d, prng_key, validation_fraction=0.1):
    """
    Splits the data into training and validation sets using JAX.

    Parameters:
    - x_sbi: array-like, shape (n_samples, ...)
    - thetas_sbi: array-like, shape (n_samples, ...)
    - sbi_d: array-like, shape (n_samples, ...)
    - prng_key: JAX PRNGKey for random shuffling
    - validation_fraction: float, fraction of data to use for validation (default 0.1)

    Returns:
    - x_sbi: array-like, training data (90% of original x_sbi)
    - thetas_sbi: array-like, training data (90% of original thetas_sbi)
    - sbi_d: array-like, training data (90% of original sbi_d)
    - x_sbi_val: array-like, validation data (10% of original x_sbi)
    - thetas_sbi_val: array-like, validation data (10% of original thetas_sbi)
    - sbi_d_val: array-like, validation data (10% of original sbi_d)
    """
    # Get the number of samples
    n_samples = x_sbi.shape[0]

    # Create shuffled indices using JAX
    indices = jnp.arange(n_samples)
    prng_key, subkey = jrandom.split(prng_key)
    shuffled_indices = jrandom.permutation(subkey, indices)

    # Determine the split index
    split_idx = int(n_samples * (1 - validation_fraction))

    # Split indices into training and validation
    train_indices = shuffled_indices[:split_idx]
    val_indices = shuffled_indices[split_idx:]

    # Split the data accordingly
    x_sbi_train = x_sbi[train_indices]
    thetas_sbi_train = thetas_sbi[train_indices]
    sbi_d_train = sbi_d[train_indices]

    x_sbi_val = x_sbi[val_indices]
    thetas_sbi_val = thetas_sbi[val_indices]
    sbi_d_val = sbi_d[val_indices]

    # Overwrite the input variables with the training data
    x_sbi = x_sbi_train
    thetas_sbi = thetas_sbi_train
    sbi_d = sbi_d_train

    return x_sbi, thetas_sbi, sbi_d, x_sbi_val, thetas_sbi_val, sbi_d_val


# ------------ mcmc transforms helpers ------------
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

# ------------ SBC calibration functions --------------
@jax.custom_vjp
def ste_hard_tanh(x):
    return jnp.where(x > 0, 1.0, 0.0)

def ste_hard_tanh_fwd(x):
    # Forward pass returns the primal output and an empty tuple for residuals
    primal_out = jnp.where(x > 0, 1.0, 0.0)
    residuals = ()  # No residuals needed in this case
    return primal_out, residuals

def ste_hard_tanh_bwd(residuals, tangents):
    # Backward pass returns the gradient with respect to the input
    grad_x = jax.nn.hard_tanh(tangents)
    return (grad_x,)


ste_hard_tanh.defvjp(ste_hard_tanh_fwd, ste_hard_tanh_bwd)


def batched_get_logq_for_ranks_jax(model, prior_samples, x, n_samples, prng_key):
    # prior_samples, prior_lps = prior_sample(prng_key, n_samples)
    # # NOTE: Is this really the same as the likelihood ratio? might wanna add prior_lps
    logq = model(prior_samples)
    return logq


def get_ranks_jax(model, sbc_model, theta, x, prior_samples, prior_lps, n_samples, prng_key):
    """The passed in x and theta should always return the 'true' value for the
    first round of the process bc that's what's used in the ste_hard_tanh calculation.
    That, or, you can calculate it separately outside and then use that. 
    
    The 'model' is the posterior log_prob."""
    keys = jrandom.split(prng_key, num=len(x))
    batched_fn = jax.vmap(
        batched_get_logq_for_ranks_jax,
        in_axes=(None, None, 0, None, 0)
    )
    logq_o = model(theta, x)
    sbc_model = lambda theta: model(theta, x)
    logq_n = sbc_model(prior_samples)
    logp_o = prior_lps
    # logq_n, logp_o = batched_fn(sbc_model, prior_samples, x, n_samples, keys)
    rankings = ste_hard_tanh(logq_o[:, None] - logq_n)
    res = jax.nn.logsumexp(
        (logq_n - logp_o) + rankings, axis=1
        ) - jax.nn.logsumexp(logq_n - logp_o, axis=1)
    # TODO: See that the prior logprob IS is justified
    return res


def get_ranks_jax_refactored(model, theta, x, prior_samples, prior_lps):
    """Compute the rank-based SBC calibration error.
    
    model: the posterior log_prob function
    theta: thetas from the batch (true values)
    x: observations from the batch
    prior_samples: pre-sampled prior values
    prior_lps: log probabilities of the prior samples
    """
    # Compute the log probabilities of the batch data under the current posterior model
    logq_o = model(theta, x)
    
    # Compute log probabilities for the pre-sampled priors
    def evaluate_prior_for_x(single_x):
        x_broadcasted = jnp.broadcast_to(single_x, (prior_samples.shape[0], single_x.shape[0]))
        return model(prior_samples, x_broadcasted)

    logq_n = jax.vmap(evaluate_prior_for_x, in_axes=0)(x)
    
    # Rankings calculation using STE hard tanh
    rankings = ste_hard_tanh(logq_o[:, None] - logq_n)
    
    # Calculate SBC result using logsumexp for stability
    res = jax.nn.logsumexp((logq_n - prior_lps) + rankings, axis=1) - \
          jax.nn.logsumexp(logq_n - prior_lps, axis=1)
    
    return res


def get_coverage_jax(ranks):
    # TODO: Double-check this. Might be flipped if the rankings are flipped.
    alpha = soft_sort(ranks[None, :])
    levels = jnp.linspace(0.0, 1.0, alpha.shape[-1] + 2)[1:-1]
    return levels, jnp.flip(alpha, axis=0)


def get_calibration_error_jax(
        model, theta, x, prior_samples, prior_lps, calibration=1):
    """
    model: posterior model log_prob evaluation. Essentially the likelihood.
    x: observed data points (don't need)
    theta: originally-generated thetas from the model using observed values. (just
      posterior samples for VI posterior).
    prior: prior that you can sample and assess lp from.
    n_samples: ?
    calibration: whether to do 
    """
    ranks = get_ranks_jax_refactored(
        model, theta, x, prior_samples, prior_lps)
    coverage, expected = get_coverage_jax(ranks)
    if calibration == 0:
        return jnp.mean(jnp.square(jax.nn.relu(expected - coverage)))
    elif calibration == 1:
        return jnp.mean(jnp.square(coverage - expected))
    else:
        return jnp.mean(
            jnp.square(
                (1 - calibration) * jax.nn.relu(expected - coverage)
                + calibration * (coverage - expected)
            )
        )

# ------------ VI Loss function ------------
def vi_fkl_sbc_post_loss(
        params: hk.Params, 
        N: int,
        prng_seq: hk.PRNGSequence, 
        likelihood_lp_fun,
        post_sample_fun,
        post_lp_fun: Callable,
        prior_lp_fun,
        ) -> Array:
    """Loss function for the VI model."""
    # sample and return log probs of samples from vi posterior
    post_samples, _ = post_sample_fun(params, prng_seq, N)
    post_lps = post_lp_fun(post_samples)
    prng_key, subkey = jrandom.split(prng_seq)
    _, likelihood_log_probs = likelihood_lp_fun(post_samples, subkey)
    prior_log_probs = prior_lp_fun(post_samples)
    joint_lps = likelihood_log_probs + prior_log_probs
    
    # cacluate logsumexp of the joint - posterior log probs
    log_weights = joint_lps - post_lps
    weights = jnp.exp(log_weights)
    weights_norm = weights.sum()
    loss = -jnp.sum((weights/weights_norm) * post_lps)

    return loss

def vi_fkl_post_loss(
                params: hk.Params, 
                N: int,
                prng_seq: hk.PRNGSequence, 
                likelihood_log_prob_fun: Callable,
                prior_lp_fun: Callable,
                post_sample_fun: Callable,
                post_lp_fun: Callable,
                ) -> Array:
    """Loss function for the VI model."""
    # sample and return log probs of samples from vi posterior
    post_samples, _ = post_sample_fun(params, prng_seq, N)
    post_lps = post_lp_fun(
        jax.lax.stop_gradient(params), post_samples)
    likelihood_log_probs = likelihood_log_prob_fun(post_samples)
    prior_log_probs = prior_lp_fun(post_samples)
    joint_log_probs = likelihood_log_probs + prior_log_probs

    # cacluate logsumexp of the joint - posterior log probs
    log_weights = joint_log_probs - post_lps
    weights = jnp.exp(log_weights)
    weights_norm = weights.sum()
    # log_weight_norm = jax.scipy.special.logsumexp(log_weights)
    # loss = jnp.sum(jnp.exp(log_weights - log_weight_norm) * (joint_log_probs - post_log_probs))
    loss = -jnp.sum((weights/weights_norm) * post_lps)
    return loss

def vi_post_iwelbo(
        params: hk.Params,
        N: int,
        keys: Array,
        likelihood_log_prob_fun: Callable,
        prior_log_prob_fun: Callable,
        post_sample_fun: Callable,
        post_lp: Callable,
        ) -> Array:
    # Making N=K square for convenience
    K = N
    vmap_post_sample = jax.vmap(post_sample_fun, in_axes=(None, 0, None))
    post_samples, _ = vmap_post_sample(
        params,
        keys,
        K
        )
    vmap_post_lp = jax.vmap(post_lp, in_axes=(None, 0))
    post_lps = vmap_post_lp(
        jax.lax.stop_gradient(params),
        post_samples
    )
    # to do importance sampling, need to draw a set of samples for each data point
    # key, subkey = jrandom.split(keys[0])
    likelihood_lps = jax.vmap(likelihood_log_prob_fun, in_axes=0)(post_samples)
    prior_lps = jax.vmap(prior_log_prob_fun, in_axes=0)(post_samples)
    joint_lps = likelihood_lps + prior_lps
    log_weights = joint_lps - post_lps # these are the "elbo particles"
    # importance sample them
    imp_weights = jnp.exp(log_weights - jax.scipy.special.logsumexp(log_weights, axis=-1))
    surr_loss = -(imp_weights * log_weights).sum(-1).mean(0)
    return surr_loss


# ------------ VI Sampling ------------
@partial(jax.jit, static_argnums=[2, 3, 4])
def generate_vi_post_samples(params, keys, post_sample_fun, total_samples, K):
    def scan_fun(carry, prng_key):
        post_samples, post_log_probs = post_sample_fun(params, prng_key, K)
        idx = jrandom.categorical(prng_key, post_log_probs)
        selected_log_prob = post_log_probs[idx]
        selected_sample = post_samples[idx]
        return carry, (selected_sample, selected_log_prob)

    _, samples_and_log_probs = jax.lax.scan(scan_fun, None, keys, length=total_samples)
    samples, log_probs = samples_and_log_probs
    return samples, log_probs


# ------------- MCMC Sampling functions ------------
@jax.jit
def prior_to_standard_normal(theta):
    theta_loc = jnp.log(jnp.array([0.5, 0.1]))
    theta_covmat = jnp.eye(2) * 0.5 ** 2  # Covariance matrix
    std_devs = jnp.sqrt(jnp.diag(theta_covmat))  # Standard deviations [0.5, 0.5]
    z = (jnp.log(theta) - theta_loc) / std_devs
    return z

@jax.jit
def prior_lp_logdetjac(z):
    theta_loc = jnp.log(jnp.array([0.5, 0.1]))
    theta_covmat = jnp.eye(2) * 0.5 ** 2
    std_devs = jnp.sqrt(jnp.diag(theta_covmat))
    log_theta = z * std_devs + theta_loc
    theta = jnp.exp(log_theta)
    logdetjac = jnp.sum(jnp.log(std_devs)) + jnp.sum(jnp.log(theta))
    return logdetjac.reshape(-1, 1)

@jax.jit
def standard_normal_to_prior(z):
    theta_loc = jnp.log(jnp.array([0.5, 0.1]))
    theta_covmat = jnp.eye(2) * 0.5 ** 2  # Covariance matrix
    std_devs = jnp.sqrt(jnp.diag(theta_covmat))  # Standard deviations [0.5, 0.5]
    # Inverse transformation
    log_theta = z * std_devs + theta_loc
    theta = jnp.exp(log_theta)
    return theta

def run_mcmc(prng_seq, mcmc_posterior, theta_0, num_adapt_steps, num_mcmc_samples):
    """
    Runs MCMC using the NUTS algorithm provided by the BlackJAX library.

    Parameters:
    prng_seq (iterable): Pseudo-random number generator sequence.
    mcmc_posterior (callable): Function that calculates the proportional posterior log probability.
    theta_0 (np.ndarray): Initial values for the theta parameters.
    num_adapt_steps (int): Number of steps to use for the window adaptation during warmup.
    num_mcmc_samples (int): Number of MCMC samples to generate.

    Returns:
    np.ndarray: Array of MCMC samples.
    """
    rng_key = prng_seq
    initial_position = prior_to_standard_normal(theta_0.mean(axis=0))[None, :]

    # Warmup phase with window adaptation
    warmup = blackjax.window_adaptation(blackjax.nuts, mcmc_posterior)
    rng_key, warmup_key, sample_key = jrandom.split(rng_key, 3)
    (state, parameters), _ = warmup.run(warmup_key, initial_position, num_steps=num_adapt_steps)

    # Define the NUTS kernel using the adapted parameters
    kernel = blackjax.nuts(mcmc_posterior, **parameters).step

    # Sampling loop
    def inference_loop(rng_key, kernel, initial_state, num_samples):
        @jax.jit
        def one_step(state, rng_key):
            state, _ = kernel(rng_key, state)
            return state, state

        keys = jrandom.split(rng_key, num_samples)
        _, states = jax.lax.scan(one_step, initial_state, keys)
        return states

    states = inference_loop(sample_key, kernel, state, num_mcmc_samples)
    mcmc_samples = states.position.squeeze()
    
    return standard_normal_to_prior(mcmc_samples), states.logdensity


def smc_inference_loop(rng_key, smc_kernel, initial_state):
    """Run the temepered SMC algorithm.

    We run the adaptive algorithm until the tempering parameter lambda reaches the value
    lambda=1.
    """
    def cond(carry):
        i, state, _k = carry
        return state.lmbda < 1

    def one_step(carry):
        i, state, k = carry
        k, subk = jrandom.split(k, 2)
        state, _ = smc_kernel(subk, state)
        return i + 1, state, k
    
    n_iter, final_state, _ = jax.lax.while_loop(
        cond, one_step, (0, initial_state, rng_key)
    )

    return n_iter, final_state

def run_mcmc_smc(prng_seq, prior_lp, loglikelihood, mcmc_posterior, theta_0, num_adapt_steps, num_mcmc_samples):
    """
    Runs MCMC using the SMC algorithm provided by the BlackJAX library.

    Parameters:
    prng_seq (iterable): Pseudo-random number generator sequence.
    mcmc_posterior (callable): Function that calculates the posterior log probability.
    theta_0 (np.ndarray): Initial values for the theta parameters.
    num_adapt_steps (int): Number of steps to use for the window adaptation during warmup.
    num_mcmc_samples (int): Number of MCMC samples to generate.

    Returns:
    np.ndarray: Array of MCMC samples.
    """
    rng_key = prng_seq
    initial_position = prior_to_standard_normal(theta_0.mean(axis=0))[None, :]

    theta_dim = theta_0.shape[-1]
    warmup = blackjax.window_adaptation(blackjax.nuts, mcmc_posterior)
    rng_key, warmup_key, sample_key = jrandom.split(rng_key, 3)
    initial_position = prior_to_standard_normal(theta_0.mean(axis=0))[None, :]
    (_, parameters), _ = warmup.run(warmup_key, initial_position, num_steps=1_000)

    # HMC parameters
    # inv_mass_matrix = jnp.ones(theta_dim).squeeze()  # Identity mass matrix
    step_size = float(parameters["step_size"])
    inv_mass_matrix = parameters["inverse_mass_matrix"]

    hmc_parameters = dict(
        step_size=step_size, inverse_mass_matrix=inv_mass_matrix, num_integration_steps=1
    )

    # Set up the tempered SMC algorithm
    tempered = blackjax.adaptive_tempered_smc(
        prior_lp,
        loglikelihood,
        blackjax.hmc.build_kernel(),
        blackjax.hmc.init,
        # extend_params(num_mcmc_samples, hmc_parameters),
        extend_params(hmc_parameters),
        resampling.systematic,
        0.5,
        num_mcmc_steps=1
    )

    # Initialize particles
    rng_key, init_key, sample_key = jrandom.split(rng_key, 3)
    initial_particles = jrandom.normal(
        init_key,
        (num_mcmc_samples, theta_dim)
    )
    initial_smc_state = tempered.init(initial_particles)
    
    # Run the SMC inference loop
    n_iter, smc_samples = smc_inference_loop(
        sample_key, tempered.step, initial_smc_state)
    
    lps = smc_samples.weights
    smc_samples = np.array(jax.tree_util.tree_leaves(smc_samples.particles)[0])

    # Return samples and log weights
    return standard_normal_to_prior(smc_samples), lps

def run_mcmc_smc_bmp(prng_seq, prior_lp, loglikelihood, mcmc_posterior, theta_0, num_mcmc_samples,
                     num_warmup_steps=1_000, target_ess=0.5):
    """
    Runs MCMC using the SMC algorithm provided by the BlackJAX library.

    Parameters:
    prng_seq (iterable): Pseudo-random number generator sequence.
    mcmc_posterior (callable): Function that calculates the posterior log probability.
    theta_0 (np.ndarray): Initial values for the theta parameters.
    num_adapt_steps (int): Number of steps to use for the window adaptation during warmup.
    num_mcmc_samples (int): Number of MCMC samples to generate.

    Returns:
    np.ndarray: Array of MCMC samples.
    """
    rng_key = prng_seq

    theta_dim = theta_0.shape[-1]
    warmup = blackjax.window_adaptation(blackjax.nuts, mcmc_posterior)
    rng_key, warmup_key, sample_key = jrandom.split(rng_key, 3)
    initial_position = jax.scipy.stats.norm.ppf(theta_0.mean(axis=0))[None, :]
    (_, parameters), _ = warmup.run(warmup_key, initial_position, num_steps=num_warmup_steps)
    
    # HMC parameters
    step_size = parameters["step_size"]
    inv_mass_matrix = parameters["inverse_mass_matrix"]

    hmc_parameters = dict(
        step_size=step_size, inverse_mass_matrix=inv_mass_matrix, num_integration_steps=1
    )

    # Set up the tempered SMC algorithm
    tempered = blackjax.adaptive_tempered_smc(
        prior_lp,
        loglikelihood,
        blackjax.hmc.build_kernel(),
        blackjax.hmc.init,
        extend_params(num_mcmc_samples, hmc_parameters),
        # extend_params(hmc_parameters),
        resampling.systematic,
        target_ess,
        num_mcmc_steps=1
    )

    # Initialize particles
    rng_key, init_key, sample_key = jrandom.split(rng_key, 3)
    initial_particles = jrandom.normal(
        init_key,
        (num_mcmc_samples, theta_dim)
    )
    initial_smc_state = tempered.init(initial_particles)
    
    # Run the SMC inference loop
    n_iter, smc_samples = smc_inference_loop(
        sample_key, tempered.step, initial_smc_state)
    
    lps = smc_samples.weights
    smc_samples = smc_samples.particles

    # Return samples and log weights
    return jax.scipy.stats.norm.cdf(smc_samples), lps


@jax.jit
def probit_transform(x):
    """Probit transform: inverse CDF of standard normal distribution."""
    return jax.scipy.stats.norm.ppf(x)

@jax.jit
def inverse_probit_transform(x):
    """Inverse of probit transform: CDF of standard normal distribution."""
    return jax.scipy.stats.norm.cdf(x)

@jax.jit
def inverse_probit_logdetjac(x):
    """NOTE: this is different from the one in the BMP.py file."""
    jac_diag = jax.vmap(jax.grad(jax.scipy.stats.norm.cdf))(x.reshape(-1))
    jac_diag = jac_diag.reshape(x.shape)
    log_abs_jac = jnp.log(jnp.abs(jac_diag))
    logdetjac = jnp.sum(log_abs_jac, axis=tuple(range(1, log_abs_jac.ndim)))
    return logdetjac.reshape(-1, 1)

def run_mcmc_bmp(prng_seq, mcmc_posterior, theta_0, num_adapt_steps, num_mcmc_samples):
    """
    Runs MCMC using the NUTS algorithm provided by the BlackJAX library.

    Parameters:
    prng_seq (iterable): Pseudo-random number generator sequence.
    mcmc_posterior (callable): Function that calculates the posterior log probability.
    theta_0 (np.ndarray): Initial values for the theta parameters.
    num_adapt_steps (int): Number of steps to use for the window adaptation during warmup.
    num_mcmc_samples (int): Number of MCMC samples to generate.

    Returns:
    np.ndarray: Array of MCMC samples.
    """
    rng_key = prng_seq
    initial_position = probit_transform(theta_0.mean(axis=0))[None, :]

    # Warmup phase with window adaptation
    warmup = blackjax.window_adaptation(blackjax.nuts, mcmc_posterior)
    rng_key, warmup_key, sample_key = jrandom.split(rng_key, 3)
    (state, parameters), _ = warmup.run(warmup_key, initial_position, num_steps=num_adapt_steps)

    # Define the NUTS kernel using the adapted parameters
    kernel = blackjax.nuts(mcmc_posterior, **parameters).step

    # Sampling loop
    def inference_loop(rng_key, kernel, initial_state, num_samples):
        @jax.jit
        def one_step(state, rng_key):
            state, _ = kernel(rng_key, state)
            return state, state

        keys = jrandom.split(rng_key, num_samples)
        _, states = jax.lax.scan(one_step, initial_state, keys)
        return states

    states = inference_loop(sample_key, kernel, state, num_mcmc_samples)
    mcmc_samples = states.position.squeeze()
    
    return inverse_probit_transform(mcmc_samples), states.logdensity


def run_mcmc_two_moons(prng_seq, mcmc_posterior, theta_init, num_adapt_steps, num_mcmc_samples):
    """
    Runs MCMC using the NUTS algorithm with transformed theta values in an unconstrained domain.

    Parameters:
    - prng_seq: Pseudo-random number generator sequence.
    - mcmc_posterior: Function that calculates the posterior log probability.
    - theta_0: Initial values for the theta parameters (in constrained space, range [-1, 1]).
    - num_adapt_steps: Number of steps for the window adaptation during warmup.
    - num_mcmc_samples: Number of MCMC samples to generate.

    Returns:
    - Array of MCMC samples in the constrained space.
    - Log densities for the MCMC samples.
    """
    rng_key = prng_seq
    initial_position = constrained_to_unconstrained(theta_init)

    # Warmup phase with window adaptation
    warmup = blackjax.window_adaptation(blackjax.nuts, mcmc_posterior)
    rng_key, warmup_key, sample_key = jax.random.split(rng_key, 3)
    (state, parameters), _ = warmup.run(warmup_key, initial_position, num_steps=num_adapt_steps)

    # Define the NUTS kernel using the adapted parameters
    kernel = blackjax.nuts(mcmc_posterior, **parameters).step

    # Sampling loop
    def inference_loop(rng_key, kernel, initial_state, num_samples):
        @jax.jit
        def one_step(state, rng_key):
            state, _ = kernel(rng_key, state)
            return state, state

        keys = jax.random.split(rng_key, num_samples)
        _, states = jax.lax.scan(one_step, initial_state, keys)
        return states

    states = inference_loop(sample_key, kernel, state, num_mcmc_samples)

    # Transform MCMC samples back to constrained space
    mcmc_samples_unconstrained = states.position.squeeze()

    # Apply map from unconstrained space to [0, 1]
    mcmc_samples = unconstrained_to_constrained(mcmc_samples_unconstrained)

    # correct logprob with the logdetjac
    logdetjac = jax.vmap(unconstrained_to_constrained_logdetjac, in_axes=[0])(mcmc_samples_unconstrained)

    return mcmc_samples, states.logdensity + logdetjac


# ------------ Data Transforms ------------
def create_lognormal_to_gaussian_bijectors(loc, scale_diag):
    log_bijector = tfp.bijectors.Log()
    scale_bijectors = [tfp.bijectors.Scale(scale=1.0 / jnp.sqrt(s)) for s in scale_diag]
    shift_bijectors = [distrax.Shift(shift=0) for m in loc]
    per_dim_bijectors = [distrax.Chain([shift, scale]) for shift, scale in zip(shift_bijectors, scale_bijectors)]
    bijector_chain = distrax.Block(distrax.Chain(per_dim_bijectors + [log_bijector]), ndims=1)
    return bijector_chain

@jax.jit
def inverse_standard_scale(scaled_x, shift, scale):
    return (scaled_x * scale) + shift

@jax.jit
def standard_scale(x):
    def single_column_fn(x):
        mean = jnp.mean(x)
        std = jnp.std(x) + 1e-10
        return (x - mean) / std
        
    def multi_column_fn(x):
        mean = jnp.mean(x, axis=0, keepdims=True)
        std = jnp.std(x, axis=0, keepdims=True) + 1e-10
        return (x - mean) / std
        
    scaled_x = jax.lax.cond(
        x.shape[-1] == 1,
        single_column_fn,
        multi_column_fn,
        x
    )
    return scaled_x


def jax_lexpand(A, *dimensions):
    """Expand tensor, adding new dimensions on left."""
    if jnp.isscalar(A):
        A = A * jnp.ones(dimensions)
        return A
    shape = tuple(dimensions) + A.shape
    A = A[jnp.newaxis, ...]
    A = jnp.broadcast_to(A, shape)
    return A


def load_dataset(split: tfds.Split, batch_size: int) -> Iterator[Batch]:
    """Helper function for loading and preparing tfds splits."""
    ds = split
    ds = ds.shuffle(buffer_size=10 * batch_size)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=1000)
    ds = ds.repeat()
    return iter(tfds.as_numpy(ds))


def prepare_tf_dataset(batch: Batch, prng_key: Optional[PRNGKey] = None) -> Array:
    """[Legacy] Helper function for preparing tfds splits for use in fliax."""
    # TODO: add length arguments to function.
    # Batch is [y, thetas, d]
    data = batch.astype(np.float32)
    x = data[:, :len_x]
    cond_data = data[:, len_x:]
    theta = cond_data[:, :-len_x]
    d = cond_data[:, -len_x:-len_xi]
    xi = cond_data[:, -len_xi:]
    return x, theta, d, xi


# ------------ Likelihood computations ------------
def sir_update(log_likelihood_fn, prior_samples, prior_log_probs, 
               prng_key, likelihood_params, x_obs, xi):
    '''
    Update the prior samples using the likelihood function.

    Args:
        log_likelihood_fn: The likelihood function to use.
        prior_samples: The prior samples to update.
        prior_log_probs: The log probabilities of the prior samples.
        prng_key: The PRNG key to use.
        likelihood_params: The parameters of the likelihood function.
        x_obs: The observed data.
        xi: The conditional information.
    Returns:
        posterior_samples: The updated posterior samples.
        posterior_weights: The updated posterior weights.
    '''
    # TODO: Need to update this for product likelihood method
    log_likelihoods = log_likelihood_fn.apply(likelihood_params, x_obs, prior_samples, xi)
    
    # Update the importance weights
    new_log_weights = prior_log_probs + log_likelihoods
    
    # Normalize the weights
    max_log_weight = jnp.max(new_log_weights)
    log_weights_shifted = new_log_weights - max_log_weight
    unnormalized_weights = jnp.exp(log_weights_shifted)
    
    # Resample with the updated weights
    posterior_weights = unnormalized_weights / jnp.sum(unnormalized_weights)
    posterior_samples = jrandom.choice(prng_key, prior_samples, shape=(len(prior_samples),), replace=True, p=posterior_weights)
    
    return posterior_samples, posterior_weights


@partial(jax.jit, static_argnums=[0])
def sir_update_prod_likelihood(
        log_likelihood_fn: Callable,
        prior_samples: Array,
        prior_log_probs: Array,
        prng_key: PRNGKey,
        likelihood_params: hk.Params, 
        x_obs: Array,
        xi: Array,
        ):
    # TODO: Need to update this for product likelihood method
    log_prob_fun = lambda params, x, theta, xi: log_likelihood_fn.apply(
                params, x, theta, xi)
    
    log_likelihoods = jax.vmap(log_prob_fun, in_axes=(None, -1, None, -1))(
                    likelihood_params,
                    x_obs[:, jnp.newaxis],
                    prior_samples, 
                    xi[:, jnp.newaxis])
    log_likelihoods = jnp.sum(log_likelihoods, axis=0)
    
    # Update the importance weights
    new_log_weights = prior_log_probs + log_likelihoods
    
    # Normalize the weights
    max_log_weight = jnp.max(new_log_weights)
    log_weights_shifted = new_log_weights - max_log_weight
    unnormalized_weights = jnp.exp(log_weights_shifted)
    
    # Resample with the updated weights
    posterior_weights = unnormalized_weights / jnp.sum(unnormalized_weights)
    
    posterior_samples = jrandom.choice(
        prng_key, 
        prior_samples, 
        shape=(len(prior_samples),), 
        replace=True, 
        p=posterior_weights)
    
    return posterior_samples, new_log_weights


def sir_update_prod_likelihood_bespoke(
        log_likelihood_fn, 
        prior_samples, 
        prior_log_probs, 
        prng_key, 
        likelihood_params, 
        x_obs,
        xi,
        x_obs_vmap_axis=0,
        xi_vmap_axis=0,
        **kwargs):
    """
    The kwargs is for params_dict that is passed in recursive rejection sampling. Not used,
    but accepted.
    """
    log_prob_fun = lambda params, x, theta, xi: log_likelihood_fn.apply(
                params, x, theta, xi)
    
    log_likelihoods = jax.vmap(log_prob_fun, in_axes=(None, x_obs_vmap_axis, None, xi_vmap_axis))(
                    likelihood_params,
                    x_obs, 
                    prior_samples, 
                    xi)
    
    # Product of each likelihood for each data point
    log_likelihoods = jnp.sum(log_likelihoods, axis=0)
    
    # Update the importance weights
    new_log_weights = log_likelihoods
    
    # Normalize the weights
    max_log_weight = jnp.max(new_log_weights)
    log_weights_shifted = new_log_weights - max_log_weight
    unnormalized_weights = jnp.exp(log_weights_shifted)
    
    # Resample with the updated weights
    posterior_weights = unnormalized_weights / jnp.sum(unnormalized_weights)
    
    posterior_samples = jrandom.choice(
        prng_key, 
        prior_samples, 
        shape=(len(prior_samples),), 
        replace=True, 
        p=posterior_weights)
    
    return posterior_samples, new_log_weights

@partial(jax.jit, static_argnums=[0,5,6])
def prod_likelihood(
        log_likelihood_fn: hk.Transformed,
        theta,
        likelihood_params,
        x_obs,
        xi,
        x_obs_vmap_axis=-1,
        xi_vmap_axis=-1):
    """
    Takes the log likelihood, its params, prior samples, prior log probs, 
    prng_key, likelihood params, observed data, and xi values. Returns the
    importance-sampled posterior samples and relative log weights.
    """
    log_prob_fun = lambda params, x, theta, xi: log_likelihood_fn.apply(
                params, x, theta, xi)
    
    log_likelihoods = jax.vmap(log_prob_fun, in_axes=(None, x_obs_vmap_axis, None, xi_vmap_axis))(
                    likelihood_params,
                    x_obs[:, jnp.newaxis],
                    theta,
                    xi[:, jnp.newaxis]
                    )

    log_likelihoods = jnp.sum(log_likelihoods, axis=0)

    return log_likelihoods


@partial(jax.jit, static_argnums=(0,))
def compute_total_likelihood(log_likelihood_fn: hk.Transformed,
                             theta,
                             params_dict,
                             x_obs,
                             xi):
    """
    This function computes the total likelihood over multiple design rounds.
    It uses a different set of parameters for each round, formatted as:
    'design_round_{design_round}_flow_params'.
    """
    log_likelihood = jnp.array(0.0)
    for key, params in params_dict.items():
        round_index = int(key.split('_')[-1])
        log_likelihood += log_likelihood_fn.apply(params, x_obs[:, round_index, None], theta, xi[:, round_index, None])
    return log_likelihood


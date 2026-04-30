from tqdm import tqdm

import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax

def pretrain_likelihood(
    prng_seq,
    design_round: int,
    design_min: float,
    design_max: float,
    xi_params: dict,
    flow_params,
    priors,
    post_samples,
    simulator_fn,
    static_outputs,
    d,
    N: int,
    sbi_prior_samples: int,
    refine_likelihood_rounds: int,
    optimizer,
    log_prob_apply_fn,
    shuffle_samples_fn,
    use_wandb: bool = False,
    wandb_prefix: str = "",
):
    """
    Pretrain a likelihood model using truncated Xi design sampling and given prior/posterior samples.
    
    Parameters
    ----------
    prng_seq : iterator
        Iterator of PRNGKeys (e.g., something like `while True: yield jax.random.split(...)`).
    design_round : int
        Current design round (0 => uses prior, >0 => uses posterior samples).
    design_min : float
        Minimum xi value for truncated normal.
    design_max : float
        Maximum xi value for truncated normal.
    xi_params : dict
        Dictionary with keys 'xi_mu' and 'xi_stddev' for the truncated normal distribution.
    flow_params : hk.Params
        Current parameters of the flow model (likelihood).
    priors : object
        Object containing a method `sample_and_log_prob(seed, sample_shape)` returning (samples, log_probs).
    post_samples : jnp.ndarray
        Posterior samples (only used for design_round > 0).
    simulator_fn : Callable
        Simulator function: simulator_fn(design, theta) -> x.
    static_outputs : jnp.ndarray
        Observed/known data to be concatenated with simulated data if `d` is not None.
    d : jnp.ndarray or None
        Additional design array. If None, simulation uses just sampled d_sim; otherwise it is concatenated.
    N : int
        Size or dimension used for truncated normal shape computations (here used with `N//2`).
    sbi_prior_samples : int
        Number of samples to draw from the prior when design_round == 0.
    refine_likelihood_rounds : int
        Number of optimization (refinement) steps.
    optimizer : optax.GradientTransformation
        Optax optimizer for training steps.
    log_prob_apply_fn : Callable
        A function that applies the log_prob (e.g., `log_prob.apply(params, prng_key, x, theta, xi)`).
    shuffle_samples_fn : Callable
        A function that shuffles (x, theta, xi) given a random key. 
        Signature: shuffle_samples_fn(prng_key, x, theta, xi) -> (x, theta, xi).
    use_wandb : bool
        Whether to log to Weights & Biases.
    wandb_prefix : str
        Prefix for wandb logging keys (e.g. "boed_").
    
    Returns
    -------
    flow_params : hk.Params
        Updated parameters of the flow (likelihood) after refinement.
    """

    # ---------------------------------------------------------
    # Generate Xi sample for design (truncated normal)
    # ---------------------------------------------------------
    design_prng_key = next(prng_seq)
    a = (design_min - xi_params['xi_mu']) / xi_params['xi_stddev']
    b = (design_max - xi_params['xi_mu']) / xi_params['xi_stddev']
    
    # shape=(1, N//2) as in your original code
    d_sim = (xi_params['xi_mu'] +
             xi_params['xi_stddev'] *
             jrandom.truncated_normal(design_prng_key, a, b, shape=(1, N // 2)))

    # ---------------------------------------------------------
    # Sample (theta) for NLL
    # ---------------------------------------------------------
    if design_round == 0:
        # from prior
        nll_theta, _ = priors.sample_and_log_prob(
            seed=next(prng_seq), sample_shape=(sbi_prior_samples, 2)
        )
        nll_theta = nll_theta.squeeze()
    else:
        # from posterior
        nll_theta = post_samples.squeeze()

    # ---------------------------------------------------------
    # Simulate x based on xi value and theta
    # ---------------------------------------------------------
    if d is None:
        # purely simulated x
        x_refine = simulator_fn(d_sim, nll_theta)
    else:
        # combine new design with existing one
        x_refine = simulator_fn(d_sim[:1, :], nll_theta)
        d_sim = jnp.concatenate((d_sim, d[:1, :]), axis=1)
        # concatenate with known (static) outputs
        x_refine = jnp.concatenate((x_refine, static_outputs), axis=1)

    # ---------------------------------------------------------
    # Define negative log-likelihood loss
    # ---------------------------------------------------------
    def nll_loss_fn(params, prng_key, x, theta, xi):
        # We'll define an inline helper that calls log_prob.apply
        def single_log_prob(p, key, xx, tt, dd):
            return log_prob_apply_fn(p, key, xx, tt, dd)

        # shuffle samples
        x_shuf, theta_shuf, xi_shuf = shuffle_samples_fn(prng_key, x, theta, xi)

        # vmapped log_prob calls
        log_likelihoods = jax.vmap(
            single_log_prob, 
            in_axes=(None, None, -1, None, -1)
        )(
            params,
            prng_key,
            x_shuf[:, jnp.newaxis],
            theta_shuf,
            xi_shuf[:, jnp.newaxis],
        )

        # sum across the "sample axis" => shape (n_samples,) before sum
        log_likelihoods = jnp.sum(log_likelihoods, axis=0)
        nll = -jnp.mean(log_likelihoods)
        return nll

    # ---------------------------------------------------------
    # Define single update step (JIT-compiled)
    # ---------------------------------------------------------
    @jax.jit
    def update(params, prng_key, opt_state, x, theta, xi):
        loss, grads = jax.value_and_grad(nll_loss_fn)(params, prng_key, x, theta, xi)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    # ---------------------------------------------------------
    # Train the flow using the new simulated data
    # ---------------------------------------------------------
    rng = next(prng_seq)
    opt_state = optimizer.init(flow_params)

    d_sim_new = jnp.broadcast_to(d_sim, (len(nll_theta), d_sim.shape[-1]))

    progress_bar = tqdm(range(refine_likelihood_rounds), desc="Training SBI")
    for _ in progress_bar:
        rng, step_rng = jax.random.split(rng)
        flow_params, opt_state, loss = update(
            flow_params, step_rng, opt_state, x_refine, nll_theta, d_sim_new
        )
        progress_bar.set_postfix(loss=loss.item(), refresh=False)

        # Optionally log to wandb
        if use_wandb:
            import wandb
            wandb.log({f"{wandb_prefix}{design_round}/preprocess_loss": loss})

    return flow_params

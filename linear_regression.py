import omegaconf
import hydra
from hydra.core.hydra_config import HydraConfig
import wandb
import os
import csv, time
import pickle as pkl
import math
import random
from tqdm import tqdm

import jax
import jax.numpy as jnp
import jax.lax as lax
import jax.random as jrandom
# from jax.test_util import check_grads

import numpy as np
from functools import partial
import optax
import distrax
import haiku as hk

import tensorflow as tf
import tensorflow_datasets as tfds

from lfiax.flows.nsf import make_nsf
from lfiax.utils.oed_losses import lf_pce_eig_scan_lin_reg, lf_pce_eig_scan
from lfiax.utils.simulators import sim_linear_data_vmap, sim_linear_data_vmap_theta, sim_linear_jax
from lfiax.utils.utils import standard_scale

from typing import (
    Any,
    Dict,
    Iterator,
    Mapping,
    Optional,
    Tuple,
    Callable,
)

Array = jnp.ndarray
PRNGKey = Array
Batch = Mapping[str, np.ndarray]
OptState = Any


# TODO: Make prior outside of the simulator so you can sample and pass it around
def make_lin_reg_prior():
    theta_shape = (2,)

    mu = jnp.zeros(theta_shape)
    sigma = (3**2) * jnp.ones(theta_shape)

    prior = distrax.Independent(
        distrax.MultivariateNormalDiag(mu, sigma)
    )
    return prior


@partial(jax.jit, static_argnums=[0])
def linear_update_likelihood(
        log_likelihood_fn: hk.Transformed,
        prior_samples: Array,
        prior_log_probs: Array,
        prng_key: PRNGKey,
        likelihood_params: hk.Params,
        x_obs: Array,
        xi: Array,
):
    log_likelihoods = log_likelihood_fn.apply(
        likelihood_params,
        x_obs,
        prior_samples,
        xi,
    )

    new_log_weights = prior_log_probs + log_likelihoods
    max_log_weight = jnp.max(new_log_weights)
    log_weights_shifted = new_log_weights - max_log_weight
    unnormalized_weights = jnp.exp(log_weights_shifted)
    posterior_weights = unnormalized_weights / jnp.sum(unnormalized_weights)

    posterior_samples = jrandom.choice(
        prng_key,
        prior_samples,
        shape=(len(prior_samples),),
        replace=True,
        p=posterior_weights,
    )

    return posterior_samples, new_log_weights


def _scalar(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    else:
        value = jax.device_get(value)
    return float(np.asarray(value).reshape(-1)[0])


def _bool_scalar(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    else:
        value = jax.device_get(value)
    return bool(np.asarray(value).reshape(-1)[0])


def _compute_lc2st_metrics(
        xs: Array,
        thetas: Array,
        posterior_samples: Array,
        x_obs: Array,
        seed: int,
        device: str,
        alpha: float = 0.05,
) -> Dict[str, Any]:
    try:
        from sbi.diagnostics.lc2st import LC2ST
        import torch
    except ImportError as exc:
        raise ImportError(
            "LC2ST diagnostics require `sbi` and `torch` in the run environment."
        ) from exc

    xs_np = np.asarray(jax.device_get(xs), dtype=np.float32)
    thetas_np = np.asarray(jax.device_get(thetas), dtype=np.float32)
    posts_np = np.asarray(jax.device_get(posterior_samples), dtype=np.float32)
    x_obs_np = np.asarray(jax.device_get(x_obs), dtype=np.float32)

    if xs_np.ndim == 1:
        xs_np = xs_np[:, None]
    if thetas_np.ndim == 1:
        thetas_np = thetas_np[:, None]
    if posts_np.ndim == 1:
        posts_np = posts_np[:, None]
    if x_obs_np.ndim == 1:
        x_obs_np = x_obs_np[None, :]

    n = min(xs_np.shape[0], thetas_np.shape[0], posts_np.shape[0])
    if n < 2:
        raise ValueError("LC2ST needs at least two samples.")

    torch_device = torch.device(
        "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
    xs_torch = torch.as_tensor(xs_np[:n], device=torch_device).float()
    thetas_torch = torch.as_tensor(thetas_np[:n], device=torch_device).float()
    posts_torch = torch.as_tensor(posts_np[:n], device=torch_device).float()
    x_obs_torch = torch.as_tensor(x_obs_np, device=torch_device).float()

    lc2st = LC2ST(
        thetas=thetas_torch,
        xs=xs_torch,
        posterior_samples=posts_torch,
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

    statistic = lc2st.get_statistic_on_observed_data(
        theta_o=posts_torch, x_o=x_obs_torch)
    p_value = lc2st.p_value(theta_o=posts_torch, x_o=x_obs_torch)
    reject = lc2st.reject_test(theta_o=posts_torch, x_o=x_obs_torch, alpha=alpha)
    del lc2st

    return {
        "lc2st_statistic": _scalar(statistic),
        "lc2st_p_value": _scalar(p_value),
        "lc2st_reject": _bool_scalar(reject),
        "lc2st_alpha": alpha,
    }


class Workspace:
    def __init__(self, cfg):
        self.cfg = cfg
        
        if cfg.wandb.use_wandb:
            wandb.config = omegaconf.OmegaConf.to_container(
                cfg, resolve=True, throw_on_missing=True
                )
            wandb.config.update(wandb.config)
            wandb.init(
                entity=self.cfg.wandb.entity, 
                project=self.cfg.wandb.project, 
                config=wandb.config
                )
        
        # Number of design rounds to perform optimization
        self.design_rounds = self.cfg.experiment.design_rounds
        self.refine_rounds = self.cfg.experiment.refine_rounds
        self.sbi_prior_samples = self.cfg.experiment.sbi_prior_samples
        self.sbi_train_steps = self.cfg.experiment.sbi_train_steps
        self.device = self.cfg.experiment.device
        self.hpc = self.cfg.experiment.hpc

        if self.hpc:
            self.work_dir = "/pub/vzaballa/lfiax_data"
            print(f'workspace: {self.work_dir}')

            current_time = time.localtime()
            current_time_str = f"{current_time.tm_year}.{current_time.tm_mon:02d}.{current_time.tm_mday:02d}.{current_time.tm_hour:02d}.{current_time.tm_min:02d}"

            eig_lambda_str = str(cfg.optimization_params.eig_lambda).replace(".", "-")
            file_name = f"eig_lambda_{eig_lambda_str}"
            self.subdir = os.path.join(
                self.work_dir,  # Change here from os.getcwd() to self.work_dir
                "lin_reg_new", 
                file_name, 
                str(cfg.designs.num_xi), 
                str(cfg.seed), 
                current_time_str
            )
            os.makedirs(self.subdir, exist_ok=True)
        else:
            # Work around since hydra logging is erring
            self.work_dir = os.getcwd()
            print(f'workspace: {self.work_dir}')

            current_time = time.localtime()
            current_time_str = f"{current_time.tm_year}.{current_time.tm_mon:02d}.{current_time.tm_mday:02d}.{current_time.tm_hour:02d}.{current_time.tm_min:02d}"
            
            eig_lambda_str = str(cfg.optimization_params.eig_lambda).replace(".", "-")
            file_name = f"eig_lambda_{eig_lambda_str}"
            self.subdir = os.path.join(os.getcwd(), "lin_reg", file_name, str(cfg.designs.num_xi), str(cfg.seed), current_time_str)
            os.makedirs(self.subdir, exist_ok=True)

        self.seed = self.cfg.seed
        rng = jrandom.PRNGKey(self.seed)
        
        if self.cfg.designs.num_xi is not None:
            if self.cfg.designs.d is None:
                self.d = jnp.array([])
                self.xi = jrandom.uniform(rng, shape=(self.cfg.designs.num_xi,), minval=-10, maxval=10)
                self.d_sim = self.xi
            else:
                self.d = jnp.array([self.cfg.designs.d])
                self.xi = jrandom.uniform(rng, shape=(self.cfg.designs.num_xi,), minval=-10, maxval=10)
                self.d_sim = jnp.concatenate((self.d, self.xi[jnp.newaxis, :]), axis=1)
        else:
            # Case when not optimizing designs
            if self.cfg.designs.d is None:
                self.d = jnp.array([])
                self.xi = jnp.array([self.cfg.designs.xi])
                self.d_sim = self.xi
            elif self.cfg.designs.xi is None:
                self.d = jnp.array([self.cfg.designs.d])
                self.xi = jnp.array([])
                self.d_sim = self.d
            else:
                self.d = jnp.array([self.cfg.designs.d])
                self.xi = jnp.array([self.cfg.designs.xi])
                self.d_sim = jnp.concatenate((self.d, self.xi), axis=1)
        
        # Bunch of event shapes needed for various functions
        len_xi = self.xi.shape[-1]
        self.xi_shape = (len_xi,)
        # self.xi_shape = (1,)
        self.theta_shape = (2,)
        self.EVENT_SHAPE = (self.d_sim.shape[-1],)
        # self.EVENT_SHAPE = (1,)
        EVENT_DIM = self.cfg.param_shapes.event_dim

        # contrastive sampling parameters
        self.M = self.cfg.contrastive_sampling.M
        self.N = self.cfg.contrastive_sampling.N

        # likelihood flow's params
        flow_num_layers = self.cfg.flow_params.num_layers
        mlp_num_layers = self.cfg.flow_params.mlp_num_layers
        hidden_size = self.cfg.flow_params.mlp_hidden_size
        num_bins = self.cfg.flow_params.num_bins

        # Optimization parameters
        self.learning_rate = self.cfg.optimization_params.learning_rate
        self.xi_lr_init = self.cfg.optimization_params.xi_learning_rate
        self.training_steps = self.cfg.optimization_params.training_steps
        self.xi_optimizer = self.cfg.optimization_params.xi_optimizer
        self.xi_scheduler = self.cfg.optimization_params.xi_scheduler
        self.xi_lr_end = self.cfg.optimization_params.xi_lr_end
        self.eig_lambda = self.cfg.optimization_params.eig_lambda
        self.eig_ema_decay = self.cfg.optimization_params.get("eig_ema_decay", 0.95)

        # Scheduler to use
        if self.xi_scheduler == "None":
            self.schedule = self.xi_lr_init
        elif self.xi_scheduler == "CosineDecay":
            lr_values = self.cfg.optimization_params.lr_values
            restarts = self.cfg.optimization_params.restarts
            decay_steps = self.training_steps / restarts
            def cosine_decay_multirestart_schedules(
                    lr_values, decay_steps, restarts, alpha=0.0, exponent=1.0):
                schedules = []
                boundaries = []
                for i in range(restarts):
                    lr = lr_values[i % len(lr_values)]
                    d = decay_steps * (i + 1)
                    s = optax.cosine_decay_schedule(
                        lr, decay_steps, alpha=alpha)
                    schedules.append(s)
                    boundaries.append(d)
                return optax.join_schedules(schedules, boundaries)

            self.schedule = cosine_decay_multirestart_schedules(
                lr_values, decay_steps, restarts, alpha=self.xi_lr_end)
        else:
            raise AssertionError("Specified unsupported scheduler.")

        @hk.without_apply_rng
        @hk.transform
        def log_prob(x: Array, theta: Array, xi: Array) -> Array:
            '''Up to user to appropriately scale their inputs :).'''
            # TODO: Pass more nsf parameters from config.yaml
            model = make_nsf(
                event_shape=self.EVENT_SHAPE,
                num_layers=flow_num_layers,
                hidden_sizes=[hidden_size] * mlp_num_layers,
                num_bins=num_bins,
                standardize_theta=True,
                use_resnet=True,
                conditional=True
            )
            return model.log_prob(x, theta, xi)
        
        self.log_prob = log_prob

        # Simulator function
        self.simulator = sim_linear_data_vmap_theta

    def run(self) -> Callable:
        logf, writer = self._init_logging()
        tic = time.time()
        
        @partial(jax.jit, static_argnums=[5, 6, 8])
        def update_pce(
            flow_params: hk.Params,
            xi_params: hk.Params,
            prng_key: PRNGKey,
            opt_state: OptState,
            opt_state_xi: OptState,
            N: int,
            M: int,
            designs: Array,
            lam: float,
        ) -> Tuple[hk.Params, OptState]:
            """Single SGD update step."""
            log_prob_fun = lambda params, x, theta, xi: self.log_prob.apply(
                params, x, theta, xi
            )

            (
                loss,
                (conditional_lp, theta_0, x, x_noiseless, noise, EIG, x_mean, x_std),
            ), grads = jax.value_and_grad(
                lf_pce_eig_scan_lin_reg, argnums=[0, 1], has_aux=True
            )(
                flow_params,
                xi_params,
                prng_key,
                log_prob_fun,
                designs,
                N=N,
                M=M,
                lam=lam,
            )

            updates, new_opt_state = optimizer.update(grads[0], opt_state)
            xi_updates, xi_new_opt_state = optimizer2.update(grads[1], opt_state_xi)

            new_params = optax.apply_updates(flow_params, updates)
            new_xi_params = optax.apply_updates(xi_params, xi_updates)
            
            return (
                new_params,
                new_xi_params,
                new_opt_state,
                xi_new_opt_state,
                loss,
                grads[1],
                xi_updates,
                conditional_lp,
                theta_0,
                x,
                x_noiseless,
                noise,
                EIG,
                x_mean,
                x_std,
            )

        # Initialize the net's params
        prng_seq = hk.PRNGSequence(self.seed)
        params = self.log_prob.init(
            next(prng_seq),
            np.zeros((1, *self.EVENT_SHAPE)),
            np.zeros((1, *self.theta_shape)),
            np.zeros((1, *self.xi_shape)),
        )
        
        optimizer = optax.adam(self.learning_rate)
        opt_state = optimizer.init(params)

        if self.xi_optimizer == "Adam":
            optimizer2 = optax.adam(learning_rate=self.schedule, b2=0.95)
        else:
            raise ValueError(f"Xi optimizer type {self.xi_optimizer} not recognized.")
        
        # Initialize designs
        # This could be initialized by a distribution of designs!
        params['xi'] = self.xi
        xi_params = {key: value for key, value in params.items() if key == 'xi'}
        # Normalize xi values for optimizer
        design_min = -10.
        design_max = 10.
        norm_type = self.cfg.designs.norm_type
        scale_factor = self.cfg.designs.scale_factor
        df = self.cfg.designs.df

        if norm_type == "inf":
            scale_factor = float(jnp.max(jnp.array([jnp.abs(design_min), jnp.abs(design_max)])))
            xi_params_scaled = {}
            xi_params_scaled['xi'] = jnp.divide(xi_params['xi'], scale_factor)
        else:
            raise ValueError(f"Norm type {norm_type} not recognized.")
        
        opt_state_xi = optimizer2.init(xi_params_scaled)
        flow_params = {key: value for key, value in params.items() if key != 'xi'}
        
        priors = make_lin_reg_prior()
        round_diagnostics = []

        # ----- Start SBI-BOED -----
        for design_round in range(self.design_rounds):
            print(f"Shape of xi params{xi_params['xi'].shape} in round {design_round}")
            eig_ema = None
            best_eig_ema = -float("inf")
            best_eig_ema_step = -1
            best_eig_ema_xi = None
            best_eig_ema_raw = None
            for step in range(self.training_steps):
                tic = time.time()

                # Sample theta_0 from prior (last round's posterior)
                if design_round == 0:
                    theta_0, theta_0_log_prob = priors.sample_and_log_prob(
                        seed=next(prng_seq), sample_shape=(self.N,))
                else:
                    d_sim = jnp.broadcast_to(
                        self.d/10., 
                        (len(post_samples), self.d.shape[-1])
                        )
                    theta_0, theta_0_log_prob = linear_update_likelihood(
                        self.log_prob, 
                        post_samples, # This is posterior from previous SBI round
                        post_log_probs,
                        next(prng_seq), 
                        prior_params, 
                        x_obs_scale, # Should be same batch size as post_samples
                        d_sim # Should be same feature size as post_samples
                        )
                
                # Simulate data from the simulator using theta_0
                x, _ , _ = self.simulator(self.d_sim, theta_0, next(prng_seq))
                scaled_x = standard_scale(x)
                x_mean, x_std = jnp.mean(x), jnp.std(x) + 1e-10
                
                # Optimize the designs using theta_0
                # flow_params, xi_params_scaled, opt_state, opt_state_xi, loss, xi_grads, xi_updates, _, EIG = update_pce(
                #     flow_params, xi_params_scaled, self.d, next(prng_seq), opt_state, opt_state_xi, scaled_x, N=self.N, M=self.M, theta_0=theta_0, lam=self.eig_lambda
                # )
                (
                flow_params,
                xi_params_scaled,
                opt_state,
                opt_state_xi,
                loss,
                xi_grads,
                xi_updates,
                conditional_lp,
                theta_0,
                x,
                x_noiseless,
                noise,
                EIG,
                x_mean,
                x_std,
                ) =  update_pce(
                    flow_params, xi_params_scaled, next(prng_seq), opt_state, opt_state_xi, N=self.N, M=self.M, designs=self.d_sim, lam=self.eig_lambda
                )
                
                if jnp.any(jnp.isnan(xi_grads['xi'])):
                    print("Gradients contain NaNs. Breaking out of loop.")
                    break
                
                if norm_type == "inf":
                    # Setting bounds on the designs
                    xi_params_scaled['xi'] = jnp.clip(
                        xi_params_scaled['xi'], 
                        a_min=jnp.divide(design_min, scale_factor), 
                        a_max=jnp.divide(design_max, scale_factor)
                        )
                    xi_params['xi'] = jnp.multiply(xi_params_scaled['xi'], scale_factor)
                else:
                    raise ValueError(f"Norm type {norm_type} not recognized.")

                # Update d_sim vector for new simulations
                if jnp.size(self.d) == 0:
                    # No previous designs
                    self.d_sim = xi_params['xi']
                elif jnp.size(self.xi) == 0:
                    # case where you don't have an optimizable input xi
                    self.d_sim =self.d_sim
                else:
                    # Case where you have both d and xi
                    self.d_sim = jnp.concatenate((self.d, xi_params['xi']), axis=0)

                eig_value = _scalar(EIG)
                if eig_ema is None:
                    eig_ema = eig_value
                else:
                    eig_ema = (
                        self.eig_ema_decay * eig_ema
                        + (1.0 - self.eig_ema_decay) * eig_value
                    )
                if eig_ema > best_eig_ema:
                    best_eig_ema = eig_ema
                    best_eig_ema_step = step
                    best_eig_ema_raw = eig_value
                    best_eig_ema_xi = jax.device_get(xi_params['xi'])
                
                run_time = time.time()-tic

                # print every 10 steps
                # if step % 10 == 0:
                print(f"STEP: {step:5d}; Xi: {xi_params['xi']}; \
                    Xi Updates: {xi_updates['xi']}; Loss: {loss}; EIG: {EIG}; \
                        EIG EMA: {eig_ema}; Best EIG EMA: {best_eig_ema}; \
                        Run time: {run_time}")
                
                writer.writerow({
                    'STEP': step,
                    'Xi': xi_params['xi'],
                    'Loss': loss,
                    'EIG': eig_value,
                    'EIG_EMA': eig_ema,
                    'best_EIG_EMA': best_eig_ema,
                    'best_EIG_EMA_step': best_eig_ema_step,
                    'time':float(run_time),
                    'seed': self.seed,
                    'lambda': self.eig_lambda,
                    'design_round': design_round,
                })
                logf.flush()

                if self.cfg.wandb.use_wandb:
                    wandb.log({
                        "loss": loss,
                        "EIG": eig_value,
                        "EIG_EMA": eig_ema,
                        "best_EIG_EMA": best_eig_ema,
                        "xi": xi_params['xi'], 
                        "xi_grads": xi_grads['xi'], 
                        })
                    
            # Make observation from noisy linear model
            true_theta = jnp.array([[5,2]])
            x_obs, _, _ = sim_linear_data_vmap_theta(self.d_sim, true_theta, next(prng_seq))
            # TODO: Track various rounds' observations
            # IDEA: This mean and stddev changes over rounds, but that could be okay
            x_obs_scale = (x_obs - x_mean) / x_std

            # ----- Perform SBI to refine the likelihood ----- 
            for r in range(self.cfg.experiment.refine_rounds):
                if r == 0:
                    prior_samples, prior_log_prob = theta_0, theta_0_log_prob
                else:
                    # Shape of these post_samples enough?
                    prior_samples, prior_log_prob = post_samples, post_log_probs
                
                print(f"Shape of prior samples: {prior_samples.shape} in SBI round {r}")

                xi_sim = jnp.broadcast_to(
                    self.d_sim/10., 
                    (len(prior_samples), self.d_sim.shape[-1])
                    )
                
                x_obs_scale = jnp.broadcast_to(
                    x_obs_scale, (len(prior_samples), x_obs_scale.shape[-1]))
                
                # Can put this in a loop to refine posterior
                post_samples, post_log_probs = linear_update_likelihood(
                    self.log_prob,
                    prior_samples, 
                    prior_log_prob, 
                    next(prng_seq), 
                    flow_params, 
                    x_obs_scale, 
                    xi_sim
                    )
                
                # Simulate data using the posterior
                x_post, _, _ = self.simulator(self.d_sim, post_samples, next(prng_seq))
                scaled_x_post = standard_scale(x_post)
                
                # Append data to simulated data history
                if r == 0:
                    data_history = scaled_x_post
                    theta_history = post_samples
                else:
                    data_history = jnp.concatenate([data_history, scaled_x_post], axis=0)
                    theta_history = jnp.concatenate([theta_history, post_samples], axis=0)
                    xi_sim = jnp.broadcast_to(
                        self.d_sim/10., 
                        (len(prior_samples)*(r+1), self.d_sim.shape[-1])
                        )
                
                # SBI update function - loss that wraps global variables theta_history & xi_sim
                def sbi_loss_fn(params: hk.Params, batch: Batch) -> Array:
                    log_prob_fun = lambda params, x, theta, xi: self.log_prob.apply(
                        params, x, theta, xi)
                        
                    log_likelihoods = jax.vmap(log_prob_fun, in_axes=(None, -1, None, -1))(
                                    params,
                                    batch[:, jnp.newaxis],
                                    theta_history, 
                                    xi_sim[:, jnp.newaxis])
                    log_likelihoods = jnp.sum(log_likelihoods, axis=0)
                    loss = -jnp.mean(log_likelihoods)
                    return loss
                
                # Note: remember you're just using the old optimizer here
                @jax.jit
                def update(params: hk.Params,
                            opt_state: OptState,
                            batch: Batch) -> Tuple[hk.Params, OptState]:
                    """Single SGD update step."""
                    loss, grads = jax.value_and_grad(sbi_loss_fn)(params, batch)
                    updates, new_opt_state = optimizer.update(grads, opt_state)
                    new_params = optax.apply_updates(params, updates)
                    return new_params, new_opt_state, loss
                
                # Early stopping variables
                best_loss = float('inf')
                no_improvement_epochs = 0
                early_stopping_limit = 20

                # TODO: Maybe implement mini batching here if run into memory issues

                # Refine the likelihood by training
                for _ in tqdm(range(self.sbi_train_steps), desc="Training SBI"):
                    flow_params, opt_state, loss = update(
                        flow_params, opt_state, data_history)
                    
                    # Check for improvement in validation loss
                    if loss < best_loss:
                        best_loss = loss
                        no_improvement_epochs = 0  # Reset the counter
                    else:
                        no_improvement_epochs += 1  # Increment the counter
                    
                    # Check for early stopping condition
                    if no_improvement_epochs >= early_stopping_limit:
                        print(f"Early stopping triggered. No improvement in validation loss for {early_stopping_limit} epochs.")
                        break

                    if self.cfg.wandb.use_wandb:
                        wandb.log({"SBI_loss": loss})

            if self.refine_rounds == 0:
                prior_samples, prior_log_prob = theta_0, theta_0_log_prob

            xi_sim = jnp.broadcast_to(
                self.d_sim/10.,
                (len(prior_samples), self.d_sim.shape[-1])
                )
            x_obs_scale = jnp.broadcast_to(
                x_obs_scale, (len(prior_samples), x_obs_scale.shape[-1]))
            post_samples, post_log_probs = linear_update_likelihood(
                self.log_prob,
                prior_samples,
                prior_log_prob,
                next(prng_seq),
                flow_params,
                x_obs_scale,
                xi_sim
                )

            x_post, _, _ = self.simulator(self.d_sim, post_samples, next(prng_seq))
            median_distance = jnp.median(
                jnp.linalg.norm(x_post - x_obs, ord=2, axis=-1))
            lc2st_xs, _, _ = self.simulator(self.d_sim, prior_samples, next(prng_seq))
            lc2st_metrics = _compute_lc2st_metrics(
                lc2st_xs,
                prior_samples,
                post_samples,
                x_obs,
                seed=self.seed,
                device=self.device,
            )
            median_distance_value = _scalar(median_distance)
            diagnostic_row = {
                'STEP': 'diagnostics',
                'Xi': xi_params['xi'],
                'Loss': '',
                'EIG': best_eig_ema_raw,
                'EIG_EMA': eig_ema,
                'best_EIG_EMA': best_eig_ema,
                'best_EIG_EMA_step': best_eig_ema_step,
                'time': '',
                'seed': self.seed,
                'lambda': self.eig_lambda,
                'design_round': design_round,
                'median_distance': median_distance_value,
                **lc2st_metrics,
            }
            round_diagnostics.append(diagnostic_row.copy())
            writer.writerow(diagnostic_row)
            logf.flush()
            print(
                f"Design round {design_round} median distance: "
                f"{median_distance_value}"
            )
            print(
                f"Design round {design_round} L-C2ST statistic: "
                f"{lc2st_metrics['lc2st_statistic']}; "
                f"p-value: {lc2st_metrics['lc2st_p_value']}; "
                f"reject: {lc2st_metrics['lc2st_reject']}"
            )
            if self.cfg.wandb.use_wandb:
                wandb.log({
                    "median_distance": median_distance_value,
                    "LC2ST_statistic": lc2st_metrics['lc2st_statistic'],
                    "LC2ST_p_value": lc2st_metrics['lc2st_p_value'],
                    "LC2ST_reject": lc2st_metrics['lc2st_reject'],
                    "LC2ST_alpha": lc2st_metrics['lc2st_alpha'],
                    "final_EIG_EMA": eig_ema,
                    "final_best_EIG_EMA": best_eig_ema,
                    "final_best_EIG_EMA_step": best_eig_ema_step,
                    "final_best_EIG_EMA_raw_EIG": best_eig_ema_raw,
                })

            # Cleaning up data, saving data, and resetting params
            if self.refine_rounds != 0:
                del data_history
                del theta_history
            
            # create an additional set of params to train for designs
            prior_params = self.log_prob.init(
                next(prng_seq),
                np.zeros((1, *self.EVENT_SHAPE)),
                np.zeros((1, *self.theta_shape)),
                np.zeros((1, *self.xi_shape)),
            )

            # Reset which params are being trained and sampled from
            flow_params, prior_params = prior_params, flow_params

            # TODO: Add xi_params to self.d and randomly reinitalize xi_params
            self.d = jnp.concatenate([self.d, xi_params['xi']], axis=0)
            self.xi = jrandom.uniform(next(prng_seq), shape=(self.cfg.designs.num_xi,), minval=-10, maxval=10)
            self.d_sim = jnp.concatenate((self.d, xi_params['xi']), axis=0)
            xi_params['xi'] = self.xi
            
            if norm_type == "inf":
                xi_params_scaled['xi'] = jnp.divide(xi_params['xi'], scale_factor)
            else:
                raise ValueError(f"Norm type {norm_type} not recognized.")

        if self.cfg.experiment.save_params:
            objects = {
                'flow_params': jax.device_get(prior_params),
                'xi_params': jax.device_get(xi_params),
                'x_obs': jax.device_get(x_obs),
                'x_obs_scale': jax.device_get(x_obs_scale),
                'd_sim': jax.device_get(self.d_sim),
                'median_distance': median_distance_value,
                'lc2st_metrics': lc2st_metrics,
                'final_EIG_EMA': eig_ema,
                'final_best_EIG_EMA': best_eig_ema,
                'final_best_EIG_EMA_step': best_eig_ema_step,
                'final_best_EIG_EMA_raw_EIG': best_eig_ema_raw,
                'final_best_EIG_EMA_xi': best_eig_ema_xi,
                'round_diagnostics': round_diagnostics,
            }
            with open(f"{self.subdir}/{self.cfg.experiment.save_name}.pkl", "wb") as f:
                pkl.dump(objects, f)

    
    def _init_logging(self):
        path = os.path.join(self.subdir, 'log.csv')
        logf = open(path, 'a') 
        fieldnames = [
            'STEP',
            'Xi',
            'Loss',
            'EIG',
            'EIG_EMA',
            'best_EIG_EMA',
            'best_EIG_EMA_step',
            'time',
            'seed',
            'lambda',
            'design_round',
            'median_distance',
            'lc2st_statistic',
            'lc2st_p_value',
            'lc2st_reject',
            'lc2st_alpha',
        ]
        writer = csv.DictWriter(logf, fieldnames=fieldnames)
        if os.stat(path).st_size == 0:
            writer.writeheader()
            logf.flush()
        return logf, writer


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg):
    fname = os.getcwd() + '/latest.pt'
    if os.path.exists(fname):
        #TODO: Test this portion of the code
        print(f'Resuming fom {fname}')
        with open(fname, 'rb') as f:
            workspace = pkl.load(f)
        print(f"STEP: {workspace.step:5d}; Xi: {workspace.xi};\
             Xi Grads: {workspace.xi_grads}; Loss: {workspace.loss}")
    else:
        workspace = Workspace(cfg)

    workspace.run()


if __name__ == "__main__":
    main()

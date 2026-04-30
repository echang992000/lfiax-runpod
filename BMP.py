import os
import omegaconf
import hydra
from hydra.core.hydra_config import HydraConfig
import wandb
import sys
import csv, time
import pickle as pkl
import math
import random
from tqdm import tqdm
from collections import deque
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

import torch
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
import tensorflow_probability.substrates.jax as tfp

from sbi.diagnostics.lc2st import LC2ST

from lfiax.flows.nsf import make_nsf
from lfiax.utils.oed_losses import lf_pce_eig_scan, lf_pce_design_dist_bmp
from lfiax.utils.simulators import make_bmp_prior, make_bmp_prior_uniform, sim_linear_data_vmap, sim_linear_data_vmap_theta
from lfiax.utils.utils import run_mcmc_bmp, inverse_probit_logdetjac, run_mcmc_smc_bmp, vi_fkl_sbc_post_loss, vi_post_iwelbo, vi_fkl_post_loss, generate_vi_post_samples
from lfiax.utils.sbi_utils import run_sbc
from lfiax.utils.sbi_losses import kl_sbc_loss_fn_general
from lfiax.utils.sir_utils import reduce_on_plateau
from lfiax.utils.update_funs import update_pce
from lfiax.utils.plotting_utils import plot_posteriors, plot_calibration
from lfiax.utils.diagnostics import lc2st_metrics
from lfiax.utils.design_baselines import select_baseline_design, validate_baseline_config

import psutil

def memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)  # in MB


from bmp_simulator.simulate_bmp import bmp_simulator, simulate_bmp_experiment_conditions


from typing import (
    Any,
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

def check_for_nans(param_dict):
    def is_nan(x):
        return jnp.any(jnp.isnan(x))
    nan_map = jax.tree_util.tree_map(is_nan, param_dict)
    return jax.tree_util.tree_reduce(lambda x, y: x or y, nan_map, initializer=False)

@jax.jit
def probit_transform(x):
    """Probit transform: inverse CDF of standard normal distribution."""
    return jax.scipy.stats.norm.ppf(x)

@jax.jit
def inverse_probit_transform(x):
    """Inverse of probit transform: CDF of standard normal distribution."""
    return jax.scipy.stats.norm.cdf(x)

@jax.jit
def probit_trans_logdetjac(x):
    """Compute the log determinant of the Jacobian for the probit transformation."""
    # Clip to avoid numerical issues
    x = jnp.clip(x, 1e-12, 1.0 - 1e-12)
    jac_diag = jax.vmap(jax.grad(probit_transform))(x)
    return jnp.sum(jnp.log(jnp.abs(jac_diag)))

@jax.jit
def shuffle_samples(key, x, theta, xi):
    num_samples = x.shape[0]
    shuffled_indices = jax.random.permutation(key, num_samples)
    return x[shuffled_indices], theta[shuffled_indices], xi[shuffled_indices]

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
        self.sbi_rounds = self.cfg.experiment.sbi_rounds
        self.true_theta = self.cfg.experiment.true_theta
        self.device = self.cfg.experiment.device
        self.hpc = self.cfg.experiment.hpc
        baseline_cfg = self.cfg.get("baseline", {})
        self.design_policy = baseline_cfg.get("design_policy", "optimized")
        self.likelihood_objective = baseline_cfg.get("likelihood_objective", "infonce_lambda")
        early_stopping_cfg = baseline_cfg.get("early_stopping", {})
        self.baseline_early_stopping = early_stopping_cfg.get("enabled", True)
        self.baseline_early_stopping_patience = int(early_stopping_cfg.get("patience", 10))
        self.baseline_early_stopping_scale = float(early_stopping_cfg.get("scale", 1e-3))
        validate_baseline_config(self.design_policy, self.likelihood_objective)

        if self.hpc:
            self.work_dir = "/pub/vzaballa/lfiax_data/bmp"
            print(f'workspace: {self.work_dir}')

            current_time = time.localtime()
            current_time_str = f"{current_time.tm_year}.{current_time.tm_mon:02d}.{current_time.tm_mday:02d}.{current_time.tm_hour:02d}.{current_time.tm_min:02d}"

            path_parts = [self.work_dir, "icml_2025"]
            if self.design_policy != "optimized":
                path_parts.extend([self.design_policy, self.likelihood_objective])
            path_parts.extend([str(cfg.seed), current_time_str])
            self.subdir = os.path.join(*path_parts)
            os.makedirs(self.subdir, exist_ok=True)
        else:
            self.work_dir = os.getcwd()
            print(f'workspace: {self.work_dir}')

            current_time = time.localtime()
            current_time_str = f"{current_time.tm_year}.{current_time.tm_mon:02d}.{current_time.tm_mday:02d}.{current_time.tm_hour:02d}.{current_time.tm_min:02d}"
            
            eig_lambda_str = str(cfg.optimization_params.eig_lambda).replace(".", "-")
            file_name = f"eig_lambda_{eig_lambda_str}"
            path_parts = [os.getcwd(), 'BMP', cfg.designs.norm_type, file_name]
            if self.design_policy != "optimized":
                path_parts.extend([self.design_policy, self.likelihood_objective])
            path_parts.extend([str(cfg.designs.num_xi), str(cfg.seed), current_time_str])
            self.subdir = os.path.join(*path_parts)
            os.makedirs(self.subdir, exist_ok=True)

        self.seed = self.cfg.seed
        rng = jrandom.PRNGKey(self.seed)
        keys = jrandom.split(rng)

        self.d = jnp.array([])
        low = jnp.log(1e-6)
        high = jnp.log(1e3)

        # Loading designs
        # uniform = distrax.Uniform(low=jnp.array([0.0]), high=jnp.array([1.0]))
        # log_uniform = distrax.Transformed(
        #     uniform, bijector=distrax.Lambda(lambda x: jnp.exp(x * (high - low) + low)))
        self.xi = distrax.Uniform(low=0., high=1e3).sample(seed=keys[0], sample_shape=(self.cfg.designs.num_xi,1))
        self.xi = self.xi.T
        self.d_sim = self.xi

        # design initialization for design distribution
        self.xi_mu = self.cfg.designs.xi_mu
        self.xi_stddev = self.cfg.designs.xi_stddev
        self.d = None
        self.static_outputs = None
        # self.design_dist = self.cfg.designs.design_dist
        
        # Event shapes needed for initializing flows
        len_xi = self.xi.shape[-1]
        self.xi_shape = (len_xi,)
        self.theta_shape = (2,)
        self.EVENT_SHAPE = (self.d_sim.shape[-1],)
        EVENT_DIM = self.cfg.param_shapes.event_dim

        # contrastive sampling parameters
        self.M = self.cfg.contrastive_sampling.M
        self.N = self.cfg.contrastive_sampling.N

        # likelihood flow's params
        flow_num_layers = self.cfg.flow_params.num_layers
        mlp_num_layers = self.cfg.flow_params.mlp_num_layers
        hidden_size = self.cfg.flow_params.mlp_hidden_size
        num_bins = self.cfg.flow_params.num_bins
        self.activation = self.cfg.flow_params.activation

        # Optimization parameters
        self.learning_rate = self.cfg.optimization_params.learning_rate
        self.xi_lr_init = self.cfg.optimization_params.xi_learning_rate
        self.training_steps = self.cfg.optimization_params.training_steps
        self.refine_likelihood_rounds = self.cfg.optimization_params.refine_likelihood_rounds
        self.xi_optimizer = self.cfg.optimization_params.xi_optimizer
        self.xi_scheduler = self.cfg.optimization_params.xi_scheduler
        self.flow_beta2 = self.cfg.optimization_params.flow_beta2
        self.xi_beta2 = self.cfg.optimization_params.xi_beta2
        self.xi_lr_end = self.cfg.optimization_params.xi_lr_end
        self.decay_rate = self.cfg.optimization_params.decay_rate
        self.eig_lambda = self.cfg.optimization_params.eig_lambda
        self.ewma_smoothing = self.cfg.optimization_params.ewma_smoothing
        self.coefficient = self.cfg.optimization_params.coefficient
        self.tau = self.cfg.optimization_params.tau
        self.flow_grad_clip = self.cfg.optimization_params.flow_grad_clip
        self.xi_grad_clip = self.cfg.optimization_params.xi_grad_clip
        self.end_sigma = self.cfg.optimization_params.end_sigma
        self.importance_sampling = self.cfg.optimization_params.importance_sampling
        self.dropout_rate = self.cfg.optimization_params.dropout_rate

        # Posterior optimization parameters
        self.post_vi_steps = self.cfg.post_optimization.steps
        self.post_k = self.cfg.post_optimization.K
        self.post_N = self.cfg.post_optimization.post_N
        self.post_resnet = self.cfg.post_optimization.resnet
        self.post_lr = self.cfg.post_optimization.lr
        self.vi_type = self.cfg.post_optimization.vi_type
        self.sbc_lam = self.cfg.post_optimization.sbc_lam
        self.perform_sbc = self.cfg.post_optimization.perform_sbc
        
        # old mcmc params
        self.num_adapt_steps = self.cfg.post_optimization.num_adapt_steps

        # mcmc params
        self.num_warmup_steps = self.cfg.mcmc.num_warmup_steps
        self.target_ess = self.cfg.mcmc.target_ess
        self.sbc_mcmc_samples = self.cfg.mcmc.sbc_mcmc_samples
        
        # Scheduler params
        self.patience = self.cfg.xi_scheduler.patience
        self.reduce_factor = self.cfg.xi_scheduler.reduce_factor
        self.min_improvement = self.cfg.xi_scheduler.min_improvement
        self.cooldown = self.cfg.xi_scheduler.cooldown
        
        # Scheduler to use
        if self.xi_scheduler == "None":
            self.schedule = self.xi_lr_init
        elif self.xi_scheduler == "Custom":
            # LR scheduling items
            self.schedule = self.xi_lr_init
            self.previous_loss = float("inf")
            self.learning_rate = self.xi_lr_init
            self.momentum_term = 1.0
        else:
            raise AssertionError("Specified unsupported scheduler.")

        @jax.jit
        def trans_logdetjac(x):
            """Compute the log determinant of the Jacobian for the probit transformation."""
            x = x + 1e-12  # Avoid log(0)
            y = jax.scipy.stats.norm.ppf(x)
            logdetjac = -jax.scipy.stats.norm.logpdf(y)
            # Sum over the appropriate axes
            logdetjac = jnp.sum(logdetjac, axis=tuple(range(1, logdetjac.ndim)))
            return logdetjac.reshape(-1, 1)
        
        @hk.transform
        def log_prob(x: Array, theta: Array, xi: Array) -> Array:
            '''Pass in all args unnormalized.'''
            model = make_nsf(
                event_shape=self.EVENT_SHAPE,
                num_layers=flow_num_layers,
                hidden_sizes=[hidden_size] * mlp_num_layers,
                num_bins=num_bins,
                standardize_theta=False,
                use_resnet=True,
                conditional=True,
                activation=self.activation,
                dropout_rate=self.dropout_rate,
            )
            gauss_theta = probit_transform(theta)
            norm_x = jax.scipy.stats.norm.ppf(x + 1e-8)
            norm_xi = probit_transform(xi/1000)
            lps = model.log_prob(norm_x, gauss_theta, norm_xi)
            logdetjac = trans_logdetjac(x)
            return lps - logdetjac.squeeze()

        self.log_prob = log_prob

        @hk.without_apply_rng
        @hk.transform
        def log_prob_nodrop(x: Array, theta: Array, xi: Array) -> Array:
            """
            Likelihood without dropout (for non-stochastic MCMC use).
            """
            model = make_nsf(
                event_shape=self.EVENT_SHAPE,
                num_layers=flow_num_layers,
                hidden_sizes=[hidden_size] * mlp_num_layers,
                num_bins=num_bins,
                standardize_theta=False,
                use_resnet=True,
                conditional=True,
                activation=self.activation,
                dropout_rate=0.0,  # no dropout during inference
            )
            # Since base is gaussian, transform from lognormal to normal
            gauss_theta = probit_transform(theta)
            norm_x = jax.scipy.stats.norm.ppf(x + 1e-8)
            norm_xi = probit_transform(xi/1000)
            lps = model.log_prob(norm_x, gauss_theta, norm_xi)
            logdetjac = trans_logdetjac(x)
            return lps - logdetjac.squeeze()
        
        self.log_prob_nodrop = log_prob_nodrop

        @hk.without_apply_rng
        @hk.transform
        def post_log_prob(theta: Array) -> Array:
            """Only need to pass in theta."""
            model = make_nsf(
                event_shape=self.theta_shape,
                num_layers=flow_num_layers,
                hidden_sizes=[hidden_size] * mlp_num_layers,
                num_bins=num_bins,
                standardize_theta=False,
                use_resnet=self.post_resnet,
                conditional=False,
                activation=self.activation,
                base_dist = "uniform",
            )
            return model.log_prob(theta)
        
        self.post_log_prob = post_log_prob

        @hk.without_apply_rng
        @hk.transform
        def post_sample(prng_seq: hk.PRNGSequence, 
                        num_samples: int,
                        ) -> Array:
            """vi sampling the posterior distribution."""
            model = make_nsf(
                event_shape=self.theta_shape,
                num_layers=flow_num_layers,
                hidden_sizes=[hidden_size] * mlp_num_layers,
                num_bins=num_bins,
                standardize_theta=False,
                use_resnet=self.post_resnet,
                conditional=False,
                activation=self.activation,
                base_dist = "uniform",
            )
            samples, log_probs = model._sample_n_and_log_prob(key=prng_seq,                                                       
                                    n=num_samples,
                                    )
            return samples, log_probs
        
        self.post_sample = post_sample

        # Simulator (BMP onestep model) to use
        model_size = (1,1,1)
        fixed_receptor = True
        
        self.simulator = simulate_bmp_experiment_conditions

    def run(self) -> Callable:
        logf, writer = self._init_logging()

        @partial(jax.jit, static_argnums=[6,7,10])
        def compute_grads_and_loss(
            flow_params: hk.Params,
            xi_params: hk.Params,
            static_designs: Array,
            static_outputs: Array,
            prng_key: PRNGKey,
            design_prng_key: PRNGKey,
            N: int,
            M: int,
            theta_0: Array, 
            scaled_x: Array,
            lam: float
        ):
            '''Basic compute grads of InfoNCE objective.'''
            log_prob_fun = lambda params, x, theta, xi: self.log_prob_nodrop.apply(
                params, x, theta, xi)
            (loss, (conditional_lp, EIG, EIGs, shuff_d_sim)), grads = jax.value_and_grad(
                lf_pce_design_dist_bmp, argnums=[0,1], has_aux=True)(
                flow_params, xi_params, static_designs, static_outputs, prng_key, design_prng_key,
                theta_0, scaled_x, log_prob_fun, N=N, M=M, lam=lam, train=True
                )
            return grads, loss, conditional_lp, EIG, EIGs, shuff_d_sim

        def update_pce_design_dists(
            flow_params: hk.Params,
            xi_params: hk.Params, # Note: these are scaled
            static_designs: Array,
            static_outputs: Array,
            prng_key: PRNGKey,
            design_prng_key: PRNGKey,
            opt_state: OptState,
            ema: optax.GradientTransformation,
            ema_opt_state: OptState,
            N: int,
            M: int,
            theta_0: Array,
            lam: float,
            lr: float,
            opt_round: int,
            scaled_x: Array,
        ) -> Tuple[hk.Params, OptState]:
            """Single SGD update step for design optimization."""
            grads, loss, conditional_lp, EIG, EIGs, shuff_d_sim = compute_grads_and_loss(
                flow_params, xi_params, static_designs, static_outputs, prng_key, design_prng_key, N, M, theta_0, scaled_x, lam)
            
            combo_params = (flow_params, xi_params)
            updates, new_opt_state = optimizer.update(grads, opt_state, combo_params)
            new_params, new_xi_params = optax.apply_updates(combo_params, updates)
            ema_params, new_ema_opt_state = ema.update(new_params, ema_opt_state)

            # New smooth decay calculation
            start_sigma = self.xi_stddev
            decay_rate = -jnp.log(self.end_sigma / start_sigma) / self.training_steps
            current_step = jnp.minimum(opt_round, self.training_steps)  # Cap at total_steps
            new_xi_params['xi_stddev'] = jnp.log(start_sigma * jnp.exp(-decay_rate * current_step))
            
            # Importance sampling step for designs
            # d_sim: shape [N,1]
            # EIGs: shape [N,]
            if self.importance_sampling:
                # Exponential schedule for standard deviation
                d_sim = shuff_d_sim
                pdf_values = jax.scipy.stats.norm.pdf(
                    d_sim, loc=jnp.exp(xi_params['xi_mu']), scale=jnp.exp(xi_params['xi_stddev']))
                truncation_correction = jax.scipy.stats.norm.cdf(
                    scale_factor, loc=jnp.exp(xi_params['xi_mu']), scale=jnp.exp(xi_params['xi_stddev'])) - \
                    jax.scipy.stats.norm.cdf(0., loc=jnp.exp(xi_params['xi_mu']), scale=jnp.exp(xi_params['xi_stddev']))
                adjusted_pdf_values = pdf_values / truncation_correction
                design_log_probs = jnp.log(adjusted_pdf_values)
                
                normalized_EIGs = (EIGs - jnp.min(EIGs)) / (jnp.max(EIGs) - jnp.min(EIGs) + 1e-8)
                w_i = design_log_probs.squeeze() + jnp.log(normalized_EIGs + 1e-8)
                p_tilde = jnp.exp(w_i - jnp.max(w_i)) / jnp.sum(jnp.exp(w_i - jnp.max(w_i)))
                E_p_tilde_xi = jnp.sum(d_sim.squeeze() * p_tilde)
                new_xi_params['xi_mu'] = jnp.log((jnp.exp(new_xi_params['xi_mu']) + E_p_tilde_xi) / 2)
            
            return new_params, new_xi_params, new_opt_state, loss, grads, conditional_lp, EIG, ema_params, new_ema_opt_state

        def _bmp_total_log_prob(params: hk.Params, x: Array, theta: Array, xi: Array) -> Array:
            log_prob_fun = lambda params, x, theta, xi: self.log_prob_nodrop.apply(
                params, x, theta, xi)
            conditional_lps = jax.vmap(log_prob_fun, in_axes=(None, -1, None, -1))(
                params, x[:, jnp.newaxis], theta, xi[:, jnp.newaxis])
            return jnp.sum(conditional_lps, axis=0)

        def _fixed_design_training_data(
            current_x: Array,
            current_theta: Array,
            current_designs: Array,
        ) -> Tuple[Array, Array, Array]:
            if self.d is None:
                return current_x, current_theta, current_designs
            return (
                jnp.concatenate([self.static_outputs[:self.N], current_x], axis=1),
                current_theta,
                jnp.concatenate([self.d[:self.N], current_designs], axis=1),
            )

        def _fixed_design_eig_bmp(
            params: hk.Params,
            static_designs: Array,
            static_outputs: Array,
            prng_key: PRNGKey,
            theta: Array,
            x: Array,
            xi_value: Array,
        ) -> Tuple[Array, Array]:
            log_prob_fun = lambda params, x, theta, xi: self.log_prob_nodrop.apply(
                params, x, theta, xi)
            fixed_xi_params = {"xi_mu": xi_value}
            _, (conditional_lps, eig, _, _) = lf_pce_design_dist_bmp(
                params,
                fixed_xi_params,
                static_designs,
                static_outputs,
                prng_key,
                prng_key,
                theta,
                x,
                log_prob_fun,
                N=self.N,
                M=self.M,
                lam=self.eig_lambda,
                train=False,
            )
            return conditional_lps, eig

        def _nle_loss_bmp(
            params: hk.Params,
            prng_key: PRNGKey,
            x: Array,
            theta: Array,
            xi: Array,
        ) -> Tuple[Array, Array]:
            x, theta, xi = shuffle_samples(prng_key, x, theta, xi)
            conditional_lps = _bmp_total_log_prob(params, x, theta, xi)
            return -jnp.mean(conditional_lps), conditional_lps

        def update_fixed_nle_bmp(
            params: hk.Params,
            prng_key: PRNGKey,
            opt_state: OptState,
            ema: optax.GradientTransformation,
            ema_opt_state: OptState,
            x: Array,
            theta: Array,
            xi: Array,
        ) -> Tuple[hk.Params, OptState]:
            (loss, conditional_lps), grads = jax.value_and_grad(
                _nle_loss_bmp, has_aux=True)(params, prng_key, x, theta, xi)
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            ema_params, new_ema_opt_state = ema.update(new_params, ema_opt_state)
            return new_params, new_opt_state, loss, grads, conditional_lps, ema_params, new_ema_opt_state

        def update_fixed_infonce_bmp(
            params: hk.Params,
            static_designs: Array,
            static_outputs: Array,
            prng_key: PRNGKey,
            opt_state: OptState,
            ema: optax.GradientTransformation,
            ema_opt_state: OptState,
            theta: Array,
            x: Array,
            xi_value: Array,
        ) -> Tuple[hk.Params, OptState]:
            log_prob_fun = lambda params, x, theta, xi: self.log_prob_nodrop.apply(
                params, x, theta, xi)
            fixed_xi_params = {"xi_mu": xi_value}
            (loss, (conditional_lps, eig, _, _)), grads = jax.value_and_grad(
                lf_pce_design_dist_bmp, argnums=0, has_aux=True)(
                params,
                fixed_xi_params,
                static_designs,
                static_outputs,
                prng_key,
                prng_key,
                theta,
                x,
                log_prob_fun,
                N=self.N,
                M=self.M,
                lam=self.eig_lambda,
                train=False,
            )
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            ema_params, new_ema_opt_state = ema.update(new_params, ema_opt_state)
            return new_params, new_opt_state, loss, grads, conditional_lps, eig, ema_params, new_ema_opt_state
        
        ############### Setting up params to optimize and hyperparams ###############
        # Initialize the net's params
        prng_seq = hk.PRNGSequence(self.seed)
        design_round = 0
        
        flow_params = self.log_prob.init(
            next(prng_seq),
            np.zeros((1, *self.EVENT_SHAPE)),
            np.zeros((1, *self.theta_shape)),
            np.zeros((1, *self.xi_shape)),
        )

        optimizer = optax.chain(
                optax.clip_by_global_norm(self.flow_grad_clip),
                optax.adamw(self.learning_rate, b2=self.flow_beta2))

        ema = optax.ema(decay=0.9999, debias=False)
        ema_opt_state = ema.init(flow_params)
        ema_params = flow_params
        
        # Initialize design xi
        flow_params['xi_mu'] = jnp.array(self.xi_mu)
        flow_params['xi_stddev'] = jnp.array(self.xi_stddev)
        xi_params = {key: value for key, value in flow_params.items() if key == 'xi_mu' or key == 'xi_stddev'}

        # Normalize xi values for optimizer
        design_min = 1e-6
        design_max = 1e3
        norm_type = self.cfg.designs.norm_type
        scale_factor = design_max

        if norm_type == "inf":
            scale_factor = float(jnp.max(jnp.array([jnp.abs(design_min), jnp.abs(design_max)])))
            xi_params_scaled = {k: jnp.divide(v, scale_factor) for k, v in xi_params.items() if k in ['xi_mu', 'xi_stddev']}
        elif norm_type == "log":
            xi_params_scaled = {k: jnp.log(v) for k, v in xi_params.items()}
        else:
            raise ValueError(f"Norm type {norm_type} not recognized.")
        
        flow_params = {key: value for key, value in flow_params.items() if key != 'xi_mu' and key != 'xi_stddev'}
        priors = distrax.Uniform(low=0., high=1)
        @jax.jit
        def prior_lp_fun(theta):
            if len(theta.shape) == 1:
                return jnp.array(0.)
            return jnp.zeros(theta.shape[0])

        # Collecting optimal design & EIG history
        d_hist = []
        best_eig_hist = []
        obs_hist = []
        x_means = []
        median_distances = []

        # LR scheduling stuff
        sch_init_fn, sch_update_fn = reduce_on_plateau(
            reduce_factor=self.reduce_factor,
            patience=self.patience,
            min_improvement=self.min_improvement,
            cooldown=self.cooldown,
            lr=self.xi_lr_init,
        )
        learning_rate = self.xi_lr_init
        is_baseline = self.design_policy != "optimized"
        
        ################# Start Design Optimization #################
        for design_round in range(self.design_rounds):
            ################# Start BOED #################
            # Initialize the optimizers for the next round of design optimization
            selected_baseline_design = None
            if is_baseline:
                selected_baseline_design = jnp.asarray(select_baseline_design(
                    self.design_policy,
                    design_round,
                    self.seed,
                    design_min,
                    design_max,
                    shape=(1,),
                ))
                xi_params['xi_mu'] = selected_baseline_design
                if norm_type == "inf":
                    xi_params_scaled['xi_mu'] = jnp.divide(selected_baseline_design, scale_factor)
                elif norm_type == "log":
                    xi_params_scaled['xi_mu'] = jnp.log(selected_baseline_design)
                else:
                    raise ValueError(f"Norm type {norm_type} not recognized.")
                opt_state = optimizer.init(flow_params)
            else:
                opt_state = optimizer.init((flow_params, xi_params_scaled))
            ema_opt_state = ema.init(flow_params)
            ema_params = flow_params
            early_best_val_loss = float("inf")
            early_bad_steps = 0
            early_best_ema_params = ema_params

            for step in range(self.training_steps):
                # (Re)set data structs to keep track of best-seen xi_params (checkpointing)
                eig_history = deque(maxlen=10)
                xi_mu_history = deque(maxlen=10)
                best_avg_eig = float('-inf')
                best_xi_mu_eig = None
                if is_baseline:
                    best_xi_mu_eig = selected_baseline_design

                # Drawing prior samples
                # TODO: double-check that these are the right spots
                if design_round == 0:
                    theta_0, _ = priors.sample_and_log_prob(seed=next(prng_seq), sample_shape=(self.N,2))
                    theta_0 = theta_0.squeeze()
                else:
                    best_xi_mu_eig = xi_params['xi_mu']
                    # resused mcmc_posterior from median distance calculation in previous round
                    theta_0 = post_samples[:self.N].squeeze()
                tic = time.time()
                
                design_prng_key = next(prng_seq)
                if is_baseline:
                    d_sim = jnp.broadcast_to(selected_baseline_design, (self.N, 1))
                    x = self.simulator(d_sim.T, theta_0)
                    scaled_x = x.T
                    x_mean, x_std = jnp.mean(x), jnp.std(x) + 1e-10
                    x_train, theta_train, xi_train = _fixed_design_training_data(
                        scaled_x, theta_0, d_sim)

                    if self.likelihood_objective == "nle":
                        flow_params, opt_state, loss, grads, conditional_lps, ema_params, ema_opt_state = update_fixed_nle_bmp(
                            flow_params,
                            next(prng_seq),
                            opt_state,
                            ema,
                            ema_opt_state,
                            x_train,
                            theta_train,
                            xi_train,
                        )
                        _, EIG = _fixed_design_eig_bmp(
                            flow_params,
                            self.d[:self.N] if self.d is not None else None,
                            self.static_outputs[:self.N] if self.static_outputs is not None else None,
                            next(prng_seq),
                            theta_0,
                            scaled_x,
                            selected_baseline_design,
                        )
                    else:
                        flow_params, opt_state, loss, grads, conditional_lps, EIG, ema_params, ema_opt_state = update_fixed_infonce_bmp(
                            flow_params,
                            self.d[:self.N] if self.d is not None else None,
                            self.static_outputs[:self.N] if self.static_outputs is not None else None,
                            next(prng_seq),
                            opt_state,
                            ema,
                            ema_opt_state,
                            theta_0,
                            scaled_x,
                            selected_baseline_design,
                        )
                    xi_grads = {"xi_mu": jnp.array(0.0), "xi_stddev": jnp.array(0.0)}
                    xi_updates = {"xi_mu": jnp.array(0.0), "xi_stddev": jnp.array(0.0)}
                else:
                    # Unnormalize designs to get proper design dist values
                    a, b = (design_min - xi_params['xi_mu']) / xi_params['xi_stddev'], (design_max - xi_params['xi_mu']) / xi_params['xi_stddev']
                    d_sim = xi_params['xi_mu'] + xi_params['xi_stddev'] * jrandom.truncated_normal(design_prng_key, a, b, shape=(self.N,1))
                    x = self.simulator(d_sim.T, theta_0)
                    # ^ returns shape (1, N) for now
                    scaled_x = x.T  # no normalization outside log_prob function anymore
                    x_mean, x_std = jnp.mean(x), jnp.std(x) + 1e-10

                    if self.d is not None:
                        flow_params, xi_params_scaled, opt_state, loss, grads, conditional_lps, EIG, ema_params, ema_opt_state = update_pce_design_dists(
                            flow_params, xi_params_scaled, self.d[:self.N], self.static_outputs[:self.N], next(prng_seq), design_prng_key, \
                                opt_state, ema, ema_opt_state, N=self.N, M=self.M, theta_0=theta_0, lam=self.eig_lambda, lr=learning_rate, \
                                    opt_round=step, scaled_x=scaled_x)
                    else:
                        flow_params, xi_params_scaled, opt_state, loss, grads, conditional_lps, EIG, ema_params, ema_opt_state = update_pce_design_dists(
                            flow_params, xi_params_scaled, self.d, self.static_outputs, next(prng_seq), design_prng_key, \
                                opt_state, ema, ema_opt_state, N=self.N, M=self.M, theta_0=theta_0, lam=self.eig_lambda, lr=learning_rate, \
                                    opt_round=step, scaled_x=scaled_x)
                
                val_loss = -jnp.mean(conditional_lps)
                should_stop_early = False

                if (not is_baseline) and self.xi_scheduler == "Custom":
                    # Using custom ReduceLROnPlateau scheduler
                    if step == 0:
                        rlrop_state = sch_init_fn(xi_params_scaled)
                    xi_updates, rlrop_state = sch_update_fn(
                        grads[1],
                        rlrop_state,
                        min_lr=self.xi_lr_end,
                        extra_args={'loss': loss}
                    )
                    self.schedule = rlrop_state.lr
                else:
                    learning_rate = self.schedule
                
                flow_grads = grads[0] if not is_baseline else grads
                if check_for_nans(flow_grads):
                    print("Flow gradients contain NaNs. Resetting to EMA params.")
                    flow_params = ema_params
                    opt_state = optimizer.init(flow_params if is_baseline else (ema_params, xi_params_scaled))
                    ema_opt_state = ema.init(flow_params)
                if (not is_baseline) and check_for_nans(grads[1]):
                    print("Xi gradients contain NaNs. Resetting to EMA params.")
                    xi_params['xi_mu'] = jnp.array(best_xi_mu_eig)
                    xi_params_scaled['xi_mu'] = jnp.log(xi_params['xi_mu'])
                    flow_params = ema_params
                    opt_state = optimizer.init((ema_params, xi_params_scaled))
                    ema_opt_state = ema.init(flow_params)

                if is_baseline:
                    xi_params['xi_mu'] = selected_baseline_design
                elif norm_type == "inf":
                    # Setting bounds on the designs
                    max_bound = jnp.divide(design_max, scale_factor)-0.001
                    xi_params_scaled['xi_mu'] = jnp.clip(
                        xi_params_scaled['xi_mu'], 
                        a_min=jnp.divide(design_min, scale_factor), 
                        a_max=max_bound
                        )
                    xi_params['xi_mu'] = jnp.multiply(xi_params_scaled['xi_mu'], scale_factor)
                    xi_params['xi_stddev'] = jnp.multiply(xi_params_scaled['xi_stddev'], scale_factor)
                elif norm_type == "log":
                    max_bound = jnp.log(design_max)
                    min_bound = jnp.log(design_min)
                    xi_params_scaled['xi_mu'] = jnp.clip(
                        xi_params_scaled['xi_mu'], 
                        a_min=min_bound,
                        a_max=max_bound
                        )
                    xi_params['xi_mu'] = jnp.exp(
                        xi_params_scaled['xi_mu'])
                    xi_params['xi_stddev'] = jnp.exp(
                        xi_params_scaled['xi_stddev'])
                else:
                    raise ValueError(f"Norm type {norm_type} not recognized.")
                
                eig_history.append(EIG)
                xi_mu_history.append(xi_params['xi_mu'])

                # Check if we have 100 measurements to calculate the rolling average
                rolling_average_eig = jnp.mean(np.array(eig_history))
                if rolling_average_eig > best_avg_eig:
                    best_avg_eig = rolling_average_eig
                    best_xi_mu_eig = jnp.mean(np.array(xi_mu_history))

                if is_baseline and self.baseline_early_stopping:
                    val_loss_float = float(jax.device_get(val_loss))
                    if val_loss_float < early_best_val_loss - self.baseline_early_stopping_scale:
                        early_best_val_loss = val_loss_float
                        early_bad_steps = 0
                        early_best_ema_params = ema_params
                    else:
                        early_bad_steps += 1

                    should_stop_early = early_bad_steps >= self.baseline_early_stopping_patience
                
                inference_time = time.time()-tic
                xi_mu_value = float(jnp.squeeze(xi_params['xi_mu']))
                xi_update_value = float(jnp.squeeze(xi_updates['xi_mu']))
                early_bad_steps_log = early_bad_steps if is_baseline else ""
                early_best_val_loss_log = (
                    early_best_val_loss
                    if is_baseline and np.isfinite(early_best_val_loss)
                    else ""
                )

                print(f"STEP: {step}; Xi: {xi_mu_value:.3f}; Xi Updates: {xi_update_value:.3f}; Loss: {float(loss):.3f}; EIG: {float(EIG):.3f}; Val Loss: {float(val_loss):.3f} Inference Time: {inference_time:.5f}")

                # Saving contents to file
                writer.writerow({
                    'STEP': step, 
                    'Xi': xi_params['xi_mu'],
                    'Loss': loss,
                    'Val Loss': val_loss,
                    'EIG': EIG,
                    'inference_time':float(inference_time),
                    'seed': self.seed,
                    'design_round': design_round,
                    'design_policy': self.design_policy,
                    'likelihood_objective': self.likelihood_objective,
                    'early_stopping_bad_steps': early_bad_steps_log,
                    'best_val_loss': early_best_val_loss_log,
                })
                logf.flush()
                
                if self.cfg.wandb.use_wandb:
                    step_metric_name = f"boed_{design_round}/step"
                    wandb.define_metric(step_metric_name)
                    wandb.define_metric(f"boed_{design_round}/*", step_metric=step_metric_name)
                    wandb_payload = {
                            f"boed_{design_round}/loss": loss,
                            f"boed_{design_round}/val_loss": val_loss,
                            f"boed_{design_round}/design_mu": xi_params['xi_mu'],
                            f"boed_{design_round}/design_stddev": xi_params['xi_stddev'],
                            f"boed_{design_round}/best_xi_mu_eig": best_xi_mu_eig,
                            f"boed_{design_round}/best_avg_eig": best_avg_eig,
                            f"boed_{design_round}/EIG": EIG,
                            f"boed_{design_round}/mean_scaled_x": x_mean,
                            f"boed_{design_round}/std_scaled_x": x_std,
                            f"boed_{design_round}/learning_rate": learning_rate,
                            f"boed_{design_round}/design_policy": self.design_policy,
                            f"boed_{design_round}/likelihood_objective": self.likelihood_objective,
                            step_metric_name: step,
                            }
                    if is_baseline:
                        wandb_payload.update({
                            f"boed_{design_round}/early_stopping_bad_steps": early_bad_steps,
                            f"boed_{design_round}/best_val_loss": early_best_val_loss,
                        })
                    wandb.log(wandb_payload)

                if should_stop_early:
                    print(
                        f"Early stopping baseline likelihood training at step {step}; "
                        f"best val NLL {early_best_val_loss:.6f}"
                    )
                    ema_params = early_best_ema_params
                    flow_params = early_best_ema_params
                    break

            ############# Posterior sampling #############
            # TODO: Check that you're saving and using the righ tparameters for simulation
            xi_params['xi_mu'] = jnp.array(best_xi_mu_eig)
            if norm_type == "inf":
                xi_params_scaled['xi_mu'] = jnp.divide(xi_params['xi_mu'], scale_factor)
            elif norm_type == "log":
                xi_params_scaled['xi_mu'] = jnp.log(xi_params['xi_mu'])
            else:
                raise ValueError(f"Norm type {norm_type} not recognized.")
            flow_params = ema_params
            
            # Log experiment design
            if self.d is None:
                self.d = jnp.broadcast_to(xi_params['xi_mu'], (self.sbi_prior_samples, 1))
            else:
                self.d = jnp.concatenate((self.d, jnp.broadcast_to(xi_params['xi_mu'], (self.sbi_prior_samples, 1))), axis=1)
            
            ############# Log experiment #############
            # Use best xi_mu corresponding to best EIG for SBI
            best_eig_hist.append(best_avg_eig)
            self.d_sim = jnp.array([xi_params['xi_mu']])
            x_obs = self.simulator(self.d_sim[:,None], jnp.array([self.true_theta]))
            if self.static_outputs is None:
                self.static_outputs = jnp.broadcast_to(x_obs, (self.sbi_prior_samples, 1))
            else: 
                self.static_outputs = jnp.concatenate((self.static_outputs, 
                                                       jnp.broadcast_to(x_obs, (self.sbi_prior_samples, 1))), axis=1)

            ############ END BOED & draw posterior sample that becomes new theta_0 ###############
            # Log posterior parameters to wandb
            @jax.jit
            def inverse_probit_logdetjac(x):
                """Compute the log determinant of the Jacobian for the inverse probit transformation."""
                x = x + 1e-12
                logdetjac = jax.scipy.stats.norm.logpdf(x)
                logdetjac = jnp.sum(logdetjac, axis=tuple(range(1, logdetjac.ndim)))
                return logdetjac.reshape(-1, 1)
            
            if self.perform_sbc:
                # TODO: sample 1k prior samples that you use in run_sbc over multiple rounds
                print("simulating samples for SBC")
                if design_round == 0:
                    theta_sbc, _ = priors.sample_and_log_prob(seed=next(prng_seq), sample_shape=(self.sbc_mcmc_samples,2))
                    theta_sbc = theta_sbc.squeeze()
                    x_sbc = self.simulator(self.d.T, theta_sbc)
                else:
                    theta_sbc = post_samples[:self.sbc_mcmc_samples].squeeze()
                    x_sbc = jnp.stack([self.simulator(self.d.T[i:i+1, :], theta_sbc) for i in range(self.d.T.shape[0])])
                    x_sbc = x_sbc.squeeze()

            ############ Start multi-round SBI ############
            # Log posterior samples and other metrics to wandb
            for sbi_round in range(self.sbi_rounds):
                # NOTE: will the post_samples be larger? want to just save last sbi_prior_samples size
                if design_round != 0: og_post_samples = post_samples[:self.sbi_prior_samples].squeeze()
                if sbi_round != 0:
                    # NOTE: skipping first round bc BOED optimized that likelihood
                    ######### sbi training likelihood #########
                    if design_round == 0 and sbi_round == 1:
                        print("simulating samples for SBI")
                        # need to sample from the prior and make simulations then append prev post samples and sims
                        theta_prior, _ = priors.sample_and_log_prob(seed=next(prng_seq), sample_shape=(self.sbi_prior_samples,2))
                        theta_prior = theta_prior.squeeze()
                        x_prior = self.simulator(self.d.T, theta_prior)
                        x_refine = jnp.concatenate((x_prior.T, x_post.T), axis=0)
                        nll_theta = jnp.concatenate((theta_prior, post_samples), axis=0)
                    elif design_round != 0 and sbi_round == 1:
                        print("simulating samples for SBI")
                        theta_prior = og_post_samples
                        x_prior = self.simulator(self.d.T, theta_prior)
                        x_refine = jnp.concatenate((x_prior.T, x_post.T), axis=0)
                        nll_theta = jnp.concatenate((theta_prior, post_samples), axis=0)
                    else:
                        x_refine = jnp.concatenate((x_refine, x_post.T), axis=0)
                        nll_theta = jnp.concatenate((nll_theta, post_samples), axis=0)
                    
                    # SBI update function - loss that wraps global variables theta_history & xi_sim
                    def nll_loss_fn(params: hk.Params, prng_key: Array, x: Array, theta: Array, xi: Array) -> Array:
                        # TODO: think about adding dropout back in...
                        log_prob_fun = lambda params, x, theta, xi: self.log_prob_nodrop.apply(
                            params, x, theta, xi)
                        x, theta, xi = shuffle_samples(prng_key, x, theta, xi)
                        log_likelihoods = jax.vmap(log_prob_fun, in_axes=(None, -1, None, -1))(
                            params,
                            x[:, jnp.newaxis],
                            theta,
                            xi[:, jnp.newaxis])
                        log_likelihoods = jnp.sum(log_likelihoods, axis=0)
                        nll = -jnp.mean(log_likelihoods)
                        return nll
                    
                    @jax.jit
                    def update(params: hk.Params,
                            prng_key: Array,
                            opt_state: OptState,
                            x: Array,
                            theta: Array,
                            xi: Array) -> Tuple[hk.Params, OptState]:
                        """Single SGD update step."""
                        loss, grads = jax.value_and_grad(nll_loss_fn)(params, prng_key, x, theta, xi)
                        updates, new_opt_state = optimizer.update(grads, opt_state, params)
                        new_params = optax.apply_updates(params, updates)
                        return new_params, new_opt_state, loss
                    
                    # TODO: maybe add ema best params during training
                    # reset optimizer for only likelihood refinement
                    rng = next(prng_seq)
                    # NOTE: just using the old optimizer here
                    opt_state = optimizer.init(flow_params)
                    progress_bar = tqdm(range(self.sbi_train_steps), desc="Training SBI")
                    for _ in progress_bar:
                        rng, step_rng = jrandom.split(rng)
                        flow_params, opt_state, loss = update(
                            flow_params, step_rng, opt_state, x_refine, nll_theta, self.d)
                        progress_bar.set_postfix(loss=loss.item(), refresh=False)
                        if self.cfg.wandb.use_wandb:
                            wandb.log({f"boed_{design_round}/sbi_{sbi_round}/refine_loss": loss,})

                # select the type of likleihood/posterior to use given BOED round (data size)
                if design_round == 0:
                    # don't need to use prior bc it's uniform
                    mcmc_posterior = lambda theta, x: self.log_prob_nodrop.apply(
                            flow_params,
                            x, # x_obs,
                            inverse_probit_transform(theta),
                            jnp.array([[xi_params['xi_mu']]])
                        ).squeeze() + inverse_probit_logdetjac(
                            theta
                        ).squeeze()
                    loglikelihood = lambda theta, x: self.log_prob_nodrop.apply(
                            flow_params,
                            x, # x_obs,
                            inverse_probit_transform(theta)[None,:],
                            jnp.array([[xi_params['xi_mu']]])
                        ).squeeze() + inverse_probit_logdetjac(
                            theta[None,:]
                        ).squeeze()
                else:
                    mcmc_posterior = lambda theta, x: jnp.sum(jax.vmap(
                        self.log_prob_nodrop.apply, in_axes=(None, -1, None, -1))(
                            flow_params, 
                            x, # self.static_outputs[0,:][None,None,:],
                            inverse_probit_transform(theta), 
                            self.d[0,:][None,None,:])
                        ).squeeze() + inverse_probit_logdetjac(
                            theta
                        ).squeeze()
                    loglikelihood = lambda theta, x: jnp.sum(jax.vmap(
                        self.log_prob_nodrop.apply, in_axes=(None, -1, None, -1))(
                            flow_params,
                            x, # self.static_outputs[0,:][None,None,:],
                            inverse_probit_transform(theta)[None,:],
                            self.d[0,:][None,None,:])
                        ).squeeze() + inverse_probit_logdetjac(
                            theta[None,:]
                        ).squeeze()
                smc_log_likelihood = partial(loglikelihood, x=self.static_outputs[0,:][None,None,:])
                smc_mcmc_posterior = partial(mcmc_posterior, x=self.static_outputs[0,:][None,None,:])
                print('starting smc mcmc posterior sampling')
                post_samples, _ = run_mcmc_smc_bmp(next(prng_seq), prior_lp_fun, smc_log_likelihood, smc_mcmc_posterior,
                                                   theta_0, self.sbi_prior_samples, self.num_warmup_steps,
                                                   self.target_ess)
                post_samples = jnp.clip(post_samples, 1e-6, 1-1e-6)
                
                # Simulate data for median distance calculation & subsequent design round
                # BUG: new simulator isn't simulating the right number of samples
                if design_round == 0:
                    x_post = self.simulator(self.d.T, post_samples)
                else:
                    x_post = jnp.stack([self.simulator(self.d.T[i:i+1, :], post_samples) for i in range(self.d.T.shape[0])])
                    x_post = x_post.squeeze()
                print("simulating samples for SBI")
                scaled_x_post = x_post[:,0][:, jnp.newaxis]

                # TODO: gotta do this stuff if you don't do SBI
                median_distance = jnp.median(jnp.linalg.norm(self.static_outputs.T - scaled_x_post, axis=-1))
                print(f"Design round {design_round}/sbi_{sbi_round}/median distance: {median_distance}")
                
                ############# Record LC2ST Metrics #############
                print("LC2ST Logging")
                if self.device == "cuda":
                    x_o = torch.from_numpy(np.asarray(self.static_outputs[0,:][None,:])).cuda()
                    post_samples_torch = torch.from_numpy(np.array(post_samples)).cuda()
                    xs = torch.from_numpy(np.array(x_post)).cuda()
                    thetas = torch.from_numpy(np.array(post_samples)).cuda()
                else:
                    x_o = torch.from_numpy(np.asarray(self.static_outputs[0,:][None,:])).float()
                    post_samples_torch = torch.from_numpy(np.array(post_samples)).float()
                    xs = torch.from_numpy(np.array(x_post)).float()
                    # BUG: should this be the same as the post_samples?
                    thetas = torch.from_numpy(np.array(post_samples)).float()
                
                # NOTE: xs shape (1, N) here
                # breakpoint()
                lc2st = LC2ST(
                    thetas=thetas,
                    xs=xs.T,
                    posterior_samples=post_samples_torch, #[:xs.shape[0]],
                    seed=self.seed,
                    num_folds=1,
                    num_ensemble=1,
                    classifier="mlp",
                    z_score=False,
                    num_trials_null=100,
                    permutation=True,
                )

                lc2st.train_under_null_hypothesis()
                lc2st.train_on_observed_data()
                
                theta_o = post_samples_torch
                statistic = lc2st.get_statistic_on_observed_data(theta_o=theta_o, x_o=x_o)
                print("L-C2ST statistic on observed data:", statistic)
                p_value = lc2st.p_value(theta_o=theta_o, x_o=x_o)
                print("P-value for L-C2ST:", p_value)

                # Decide whether to reject the null hypothesis at a significance level alpha
                alpha = 0.05  # 95% confidence level
                reject = lc2st.reject_test(theta_o=theta_o, x_o=x_o, alpha=alpha)
                print(f"Reject null hypothesis at alpha = {alpha}:", reject)
                del lc2st

                ###### plot the calibration curve ########
                if self.perform_sbc:
                    print(f"Running SBC for design round {design_round}/sbi_{sbi_round}")
                    _, empirical_coverage, levels, lower_bounds, upper_bounds = run_sbc(
                        theta_sbc, # TODO: which prior samples to use?
                        x_sbc.T,  # This needs to be _simulations_ of the data
                        mcmc_fn=run_mcmc_smc_bmp,
                        num_warmup_steps=self.num_warmup_steps,
                        num_mcmc_samples=self.sbc_mcmc_samples,
                        prng_seq=next(prng_seq),
                        prior_lp=prior_lp_fun,
                        log_likelihood_fn=loglikelihood,
                        mcmc_posterior=mcmc_posterior,
                    )

                ############# save likelihood and log metrics #############
                if self.cfg.experiment.save_params:
                    flow_save_key = f"design_round_{design_round}_flow_params_sbi_{sbi_round}"
                    objects = {flow_save_key: jax.device_get(flow_params),
                            "theta_0": jax.device_get(theta_0),
                            "x_obs": jax.device_get(x_obs),
                            "xi": jax.device_get(xi_params['xi_mu']),
                            "design_policy": self.design_policy,
                            "likelihood_objective": self.likelihood_objective,
                            'LC2ST_statistic': statistic,
                            'LC2ST_p_value': p_value,
                            'LC2ST_reject': reject,}
                    with open(f"{self.subdir}/{flow_save_key}.pkl", "wb") as f:
                        pkl.dump(objects, f)
                
                if self.cfg.wandb.use_wandb:
                    ####### Plotting for the posterior distribution #######
                    plot_path0 = plot_posteriors(post_samples, (0.85, 0.85), self.subdir)

                    wandb_payload = {
                        f'boed_{design_round}/sbi_{sbi_round}/posteriors': wandb.Image(str(plot_path0)),
                        f"boed_{design_round}/sbi_{sbi_round}/variance": jnp.var(post_samples),
                        f"boed_{design_round}/sbi_{sbi_round}/median_distance": median_distance,
                        f"boed_{design_round}/sbi_{sbi_round}/LC2ST_statistic": statistic,
                        f"boed_{design_round}/sbi_{sbi_round}/LC2ST_p_value": p_value,
                        f"boed_{design_round}/sbi_{sbi_round}/LC2ST_reject": reject,
                    }
                    if self.perform_sbc:
                        ####### Plotting for the calibration curve #######
                        plot_path1 = plot_calibration(levels, empirical_coverage, lower_bounds, upper_bounds, self.subdir)
                        wandb_payload[f'boed_{design_round}/sbi_{sbi_round}/calibration'] = wandb.Image(str(plot_path1))
                    wandb.log(wandb_payload)

            ############ Reset for BOED ############
            # Reset to be in the middle of the domain again
            self.xi = jnp.array(self.xi_mu)
            self.d_sim = self.xi
            
            x_means.append(x_mean)
            obs_hist.append(x_obs)
            d_hist.append(xi_params['xi_mu'])
            median_distances.append(median_distance)
            if self.cfg.wandb.use_wandb:
                wandb.log({
                    f"boed_{design_round}/round_EIG": best_avg_eig,
                    f"boed_{design_round}/round_median_distance": median_distance,
                    f"boed_{design_round}/round_LC2ST_statistic": statistic,
                    f"boed_{design_round}/round_LC2ST_p_value": p_value,
                    f"boed_{design_round}/round_LC2ST_reject": reject,
                    f"boed_{design_round}/round_design": xi_params['xi_mu'],
                    "round/design_round": design_round,
                    "round/EIG": best_avg_eig,
                    "round/median_distance": median_distance,
                    "round/LC2ST_statistic": statistic,
                    "round/LC2ST_p_value": p_value,
                    "round/LC2ST_reject": reject,
                })
            xi_params['xi_mu'] = self.xi
            xi_params['xi_stddev'] = self.xi_stddev
            
            # Reset the xi_params_scaled for optimization
            if norm_type == "inf":
                xi_params_scaled['xi_mu'] = jnp.divide(self.xi, scale_factor)
                xi_params_scaled['xi_stddev'] = jnp.divide(self.xi_stddev, scale_factor)
            elif norm_type == "log":
                xi_params_scaled['xi_mu'] = jnp.log(self.xi)
                xi_params_scaled['xi_stddev'] = jnp.log(self.xi_stddev)
            else:
                raise ValueError(f"Norm type {norm_type} not recognized.")
            
             # Save the params every round
            if self.cfg.experiment.save_params:
                flow_save_key = f"design_round_{design_round}_flow_params"
                objects = {flow_save_key: jax.device_get(flow_params),
                           "x_obs": jax.device_get(x_obs),
                           "best_xi": jax.device_get(xi_params['xi_mu']),
                           "design_policy": self.design_policy,
                           "likelihood_objective": self.likelihood_objective,
                           }
                with open(f"{self.subdir}/{flow_save_key}.pkl", "wb") as f:
                    pkl.dump(objects, f)
            
            # optionally reset the params
            if self.cfg.flow_params.reset_flow:
                del flow_params
                flow_params = self.log_prob.init(
                    next(prng_seq),
                    np.zeros((1, *self.EVENT_SHAPE)),
                    np.zeros((1, *self.theta_shape)),
                    np.zeros((1, *self.xi_shape))
                )

        if self.cfg.wandb.use_wandb:
            wandb.log({f"boed_{design_round}/final_design_EIG": jnp.sum(np.array(best_eig_hist))})
        print(f"final design EIG: {jnp.sum(np.array(best_eig_hist))}")
        
        if self.cfg.experiment.save_params:
            objects = {
                'xi_params': jax.device_get(xi_params),
                'x_obs': jax.device_get(obs_hist),
                'd_hist': jax.device_get(d_hist),
                'best_eig_hist': jax.device_get(best_eig_hist),
                'post_samples': jax.device_get(post_samples),
                'median_distances': jax.device_get(median_distances),
                'design_policy': self.design_policy,
                'likelihood_objective': self.likelihood_objective,
            }
            with open(f"{self.subdir}/{self.cfg.experiment.save_name}.pkl", "wb") as f:
                pkl.dump(objects, f)

    def _init_logging(self):
        path = os.path.join(self.subdir, 'log.csv')
        logf = open(path, 'a') 
        fieldnames = [
            'STEP',
            'd_sim',
            'Xi',
            'Loss',
            'Val Loss',
            'EIG',
            'inference_time',
            'seed',
            'design_round',
            'design_policy',
            'likelihood_objective',
            'early_stopping_bad_steps',
            'best_val_loss',
        ]
        writer = csv.DictWriter(logf, fieldnames=fieldnames)
        if os.stat(path).st_size == 0:
            writer.writeheader()
            logf.flush()
        return logf, writer


from BMP import Workspace as W

@hydra.main(version_base=None, config_path=".", config_name="config_bmp")
def main(cfg):
    fname = os.getcwd() + '/latest.pt'
    if os.path.exists(fname):
        print(f'Resuming fom {fname}')
        with open(fname, 'rb') as f:
            workspace = pkl.load(f)
        print(f"STEP: {workspace.step:5d}; Xi: {workspace.xi};\
             Xi Grads: {workspace.xi_grads}; Loss: {workspace.loss}")
    else:
        workspace = W(cfg)

    workspace.run()


if __name__ == "__main__":
    main()

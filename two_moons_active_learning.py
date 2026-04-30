import os
import time
import pickle
import csv
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy as jsp
import optax
import haiku as hk
import numpy as np
import sbibm
import torch
from torch.utils.data import Dataset, DataLoader
import hydra
from hydra.core.hydra_config import HydraConfig
import omegaconf
from omegaconf import DictConfig
import wandb
from functools import partial

from lfiax.flows.nsf import make_nsf
from lfiax.utils.utils import run_mcmc_two_moons, unconstrained_to_constrained, unconstrained_to_constrained_logdetjac
from lfiax.utils.sbi_losses import kl_loss_fn, kl_sbc_loss_fn, lf_pce_eig_scan
from lfiax.utils.oed_losses import lf_epig_scan
from lfiax.utils.sbi_utils import run_sbc_two_moons
from lfiax.utils.dropout_diagnostics import (
    summarize_dropout_diagnostics,
    summarize_parameter_dropout_diagnostics,
)

from sbi.diagnostics.lc2st import LC2ST
try:
    from sbi.diagnostics.c2st import c2st
except ImportError:  # pragma: no cover - handles older sbi APIs
    try:
        from sbi.utils.metrics import c2st
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Unable to import c2st from sbi. Please check your sbi version and API."
        ) from exc

from typing import Mapping, Any, Tuple, Callable

# Define types
Array = jnp.ndarray
PRNGKey = Array
Batch = Mapping[str, np.ndarray]
OptState = Any

# Create custom dataset and dataloader
class CustomDataset(Dataset):
    def __init__(self, x, theta):
        self.x = x
        self.theta = theta

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {
            'x': self.x[idx],
            'theta': self.theta[idx],
        }

def custom_collate(batch):
    batch_x = torch.stack([item['x'] for item in batch])
    batch_theta = torch.stack([item['theta'] for item in batch])
    return {
        'x': batch_x,
        'theta': batch_theta,
    }

def compute_validation_loss(params, val_loader, eval_fn):
    val_losses = []
    for val_batch in val_loader:
        val_batch_jax = {'x': jnp.array(val_batch['x'].numpy()), 'theta': jnp.array(val_batch['theta'].numpy())}
        val_loss = eval_fn(params, val_batch_jax)
        val_losses.append(val_loss)
    return jnp.mean(jnp.array(val_losses))


def compute_dropout_diagnostics(
    params,
    x_val,
    theta_val,
    prng_key,
    log_prob_apply,
    eval_log_prob_apply,
    num_particles,
    num_bins,
):
    x_val = np.asarray(x_val)
    theta_val = np.asarray(theta_val)
    has_repeated_simulations = x_val.ndim == 3
    if has_repeated_simulations:
        num_thetas, simulations_per_theta = x_val.shape[:2]
        x_flat = x_val.reshape((num_thetas * simulations_per_theta,) + x_val.shape[2:])
        theta_flat = np.repeat(theta_val, simulations_per_theta, axis=0)
    else:
        num_thetas = None
        simulations_per_theta = None
        x_flat = x_val
        theta_flat = theta_val

    x_flat = jnp.array(x_flat)
    theta_flat = jnp.array(theta_flat)
    xi = jnp.zeros((x_flat.shape[0], 0))
    dropout_keys = jrandom.split(prng_key, int(num_particles))

    def particle_log_prob(dropout_key):
        return log_prob_apply(params, dropout_key, x_flat, theta_flat, xi)

    log_probs = jax.vmap(particle_log_prob)(dropout_keys)
    diagnostics = {
        "point": summarize_dropout_diagnostics(np.array(log_probs), num_bins),
    }
    if has_repeated_simulations:
        deterministic_log_probs = eval_log_prob_apply(params, x_flat, theta_flat, xi)
        diagnostics["parameter"] = summarize_parameter_dropout_diagnostics(
            np.array(log_probs).reshape((int(num_particles), num_thetas, simulations_per_theta)),
            np.array(deterministic_log_probs).reshape((num_thetas, simulations_per_theta)),
            num_bins,
        )
    return diagnostics

@partial(jax.jit, static_argnums=(4, 5))
def kl_update(params, prng_key, opt_state, batch, loss_fn, optimizer):
    grads = jax.grad(loss_fn)(params, prng_key, batch)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state

def kl_sbc_update(params, prng_key, opt_state, batch, loss_fn, optimizer):
    grads = jax.grad(loss_fn)(params, prng_key, batch)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state

@partial(jax.jit, static_argnums=(4, 5))
def mi_update(params, prng_key, opt_state, batch, loss_fn, optimizer):
    (loss, (conditional_lp, EIG)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, prng_key, batch)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, EIG

@partial(jax.jit, static_argnums=(4, 5))
def mi_update_topk(params, prng_key, opt_state, batch, loss_fn, optimizer):
    (loss, (conditional_lp, EIG, top_thetas, top_scores)), grads = jax.value_and_grad(
        loss_fn, has_aux=True
    )(params, prng_key, batch)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, EIG, top_thetas, top_scores


def update_topk_list(prev_thetas, prev_scores, new_thetas, new_scores, top_k):
    if prev_scores is None:
        return new_thetas, new_scores
    all_scores = np.concatenate([prev_scores, new_scores], axis=0)
    all_thetas = np.concatenate([prev_thetas, new_thetas], axis=0)
    top_idx = np.argsort(all_scores)[-top_k:]
    return all_thetas[top_idx], all_scores[top_idx]


def _prepare_theta_xi_for_epig(theta, y=None):
    if theta.ndim == 1:
        theta_b = theta[None, :]
        xi_b = jnp.zeros((1, 0))
        if y is not None:
            y_b = y[None, :]
            return theta_b, xi_b, y_b, True
        return theta_b, xi_b, True
    xi_b = jnp.zeros((theta.shape[0], 0))
    if y is not None:
        return theta, xi_b, y, False
    return theta, xi_b, False




@hydra.main(version_base=None, config_path=".", config_name="config_two_moons_active_learning")
def main(cfg: DictConfig):
    if cfg.wandb.use_wandb:
        wandb.config = omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
            )
        wandb.config.update(wandb.config)
        wandb.init(
            entity=cfg.wandb.entity, 
            project=cfg.wandb.project, 
            config=wandb.config
            )
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Simulate data using sbibm (PyTorch-based)
    task = sbibm.get_task("two_moons")
    prior = task.get_prior()
    simulator = task.get_simulator()

    # Generate initial data
    num_samples = cfg.simulator.num_samples
    thetas = prior(num_samples)  # PyTorch tensor
    xs = simulator(thetas)       # PyTorch tensor

    dropout_diag_thetas = None
    dropout_diag_xs = None
    if cfg.dropout_diagnostics.enabled:
        torch_rng_state = torch.get_rng_state()
        dropout_diag_thetas = prior(cfg.dropout_diagnostics.num_validation_samples)
        simulations_per_theta = cfg.dropout_diagnostics.simulations_per_theta
        dropout_diag_theta_repeats = dropout_diag_thetas.repeat_interleave(
            simulations_per_theta, dim=0
        )
        dropout_diag_xs_flat = simulator(dropout_diag_theta_repeats)
        dropout_diag_xs = dropout_diag_xs_flat.reshape(
            cfg.dropout_diagnostics.num_validation_samples,
            simulations_per_theta,
            dropout_diag_xs_flat.shape[-1],
        )
        torch.set_rng_state(torch_rng_state)

    # Define shapes
    EVENT_SHAPE = (xs.shape[1],)
    theta_shape = (thetas.shape[1],)

    # Hyperparameters from config
    flow_num_layers = cfg.flow_num_layers
    mlp_num_layers = cfg.mlp_num_layers
    hidden_size = cfg.hidden_size
    num_bins = cfg.num_bins
    batch_size = cfg.batch_size
    learning_rate = cfg.learning_rate
    sbc_n_samples = cfg.sbc_n_samples
    sbc_lambda = cfg.sbc_lambda

    # Use raw values (no normalization)
    x_train_mean = jnp.zeros(EVENT_SHAPE)
    x_train_std = jnp.ones(EVENT_SHAPE)
    theta_train_mean = jnp.zeros(theta_shape)
    theta_train_std = jnp.ones(theta_shape)

    # contrastive learning parameters
    M = cfg.mi_params.M
    eig_lambda = cfg.mi_params.eig_lambda
    top_k = cfg.mi_params.top_k
    return_top_k = top_k > 0
    
    # @hk.without_apply_rng
    @hk.transform
    def log_prob(x: Array, theta: Array, xi: Array) -> Array:
        model = make_nsf(
            event_shape=EVENT_SHAPE,
            num_layers=flow_num_layers,
            hidden_sizes=[hidden_size] * mlp_num_layers,
            num_bins=num_bins,
            standardize_theta=False,
            use_resnet=True,
            activation="gelu",
            dropout_rate=0.2,
        )
        return model.log_prob(x, theta, xi)

    @hk.transform
    def sample_fn(prng_key: PRNGKey, num_samples: int, theta: Array, xi: Array) -> Array:
        model = make_nsf(
            event_shape=EVENT_SHAPE,
            num_layers=flow_num_layers,
            hidden_sizes=[hidden_size] * mlp_num_layers,
            num_bins=num_bins,
            standardize_theta=False,
            use_resnet=True,
            activation="gelu",
            dropout_rate=0.2,
        )
        return model._sample_n(key=prng_key, n=num_samples, theta=theta, xi=xi)
    
    @hk.without_apply_rng
    @hk.transform
    def eval_log_prob(x: Array, theta: Array, xi: Array) -> Array:
        model = make_nsf(
            event_shape=EVENT_SHAPE,
            num_layers=flow_num_layers,
            hidden_sizes=[hidden_size] * mlp_num_layers,
            num_bins=num_bins,
            standardize_theta=False,
            use_resnet=True,
            activation="gelu",
            dropout_rate=0.0,
        )
        return model.log_prob(x, theta, xi)

    @jax.jit
    def eval_fn(params: hk.Params, batch: Batch) -> Array:
        x = batch['x']
        theta = batch['theta']
        dummy_xi = jnp.zeros((x.shape[0], 0))
        loss = -jnp.mean(eval_log_prob.apply(params, x, theta, dummy_xi))
        return loss
    
    # Initialize model parameters and optimizer
    prng_seq = hk.PRNGSequence(cfg.seed)

    def prior_sampler(n_samples):
        prng_key = next(prng_seq)
        lower = jnp.array([-1.0, -1.0])
        upper = jnp.array([1.0, 1.0])
        prior_samples = jrandom.uniform(prng_key, (n_samples, 2), minval=lower, maxval=upper)
        prior_lps = jnp.full((n_samples,), -jnp.sum(jnp.log(upper - lower)))
        return prior_samples, prior_lps

    def prior_sampler_mi(prng_key, n_samples):
        lower = jnp.array([-1.0, -1.0])
        upper = jnp.array([1.0, 1.0])
        prior_samples = jrandom.uniform(prng_key, (n_samples, 2), minval=lower, maxval=upper)
        return prior_samples

    def run_active_round(active_round, thetas, xs):
        num_samples = thetas.shape[0]
        if num_samples == 1:
            thetas_train = thetas
            xs_train = xs
            thetas_val = thetas
            xs_val = xs
        else:
            train_size = int(num_samples * 0.8)
            indices = torch.randperm(num_samples)
            thetas_shuffled = thetas[indices]
            xs_shuffled = xs[indices]
            thetas_train = thetas_shuffled[:train_size]
            xs_train = xs_shuffled[:train_size]
            thetas_val = thetas_shuffled[train_size:]
            xs_val = xs_shuffled[train_size:]

        train_dataset = CustomDataset(xs_train, thetas_train)
        val_dataset = CustomDataset(xs_val, thetas_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)

        N = num_samples

        round_prefix = f"active_round_{active_round + 1}/"
        def log_round(data):
            if cfg.wandb.use_wandb:
                wandb.log({f"{round_prefix}{k}": v for k, v in data.items()})

        if cfg.hpc:
            work_dir = "/pub/vzaballa/lfiax_data"
            print(f'workspace: {work_dir}')

            current_time = time.localtime()
            current_time_str = f"{current_time.tm_year}.{current_time.tm_mon:02d}.{current_time.tm_mday:02d}.{current_time.tm_hour:02d}.{current_time.tm_min:02d}"

            subdir = os.path.join(
                work_dir,
                "two_moons",
                f"pretrain_{cfg.pre_training.enabled}",
                cfg.pre_training.loss_type,
                cfg.main_training.loss_type,
                str(cfg.seed),
                f"round_{active_round + 1}",
                current_time_str
            )
            os.makedirs(subdir, exist_ok=True)
        else:
            print("not saving params when not in HPC")
            subdir = None

        params = log_prob.init(
            next(prng_seq),
            jnp.zeros((1, *EVENT_SHAPE)),
            jnp.zeros((1, *theta_shape)),
            jnp.zeros((1, 0)),
        )
        optimizer = optax.adamw(learning_rate)
        opt_state = optimizer.init(params)

        # Function to switch between loss functions
        def get_loss_and_update_fn(loss_type):
            if loss_type == 'kl_loss':
                loss_fn = lambda params, prng_key, batch: kl_loss_fn(
                    params, prng_key, batch, x_train_mean, x_train_std, theta_train_mean, theta_train_std, log_prob)
                update_fn = lambda params, prng_key, opt_state, batch: kl_update(
                        params, prng_key, opt_state, batch, loss_fn, optimizer)
            elif loss_type == 'kl_sbc_loss':
                loss_fn = lambda params, prng_key, batch: kl_sbc_loss_fn(
                    params, prng_key, batch, prior_sampler, x_train_mean, x_train_std, theta_train_mean, theta_train_std, 
                    log_prob, sbc_n_samples, sbc_lambda)
                update_fn = lambda params, prng_key, opt_state, batch: kl_sbc_update(
                    params, prng_key, opt_state, batch, loss_fn, optimizer)
            elif loss_type == 'mi_loss':
                loss_fn = lambda params, prng_key, batch: lf_pce_eig_scan(
                    params, prng_key, batch, log_prob.apply, N, M, eig_lambda,
                    x_train_mean, x_train_std, theta_train_mean, theta_train_std, prior_sampler_mi,
                    return_top_k=return_top_k, top_k=top_k)
                if return_top_k:
                    update_fn = lambda params, prng_key, opt_state, batch: mi_update_topk(
                            params, prng_key, opt_state, batch, loss_fn, optimizer)
                else:
                    update_fn = lambda params, prng_key, opt_state, batch: mi_update(
                            params, prng_key, opt_state, batch, loss_fn, optimizer)
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")
            return loss_fn, update_fn

        # Pre-training phase
        topk_thetas = None
        topk_scores = None

        if cfg.pre_training.enabled:
            _, pre_train_update_fn = get_loss_and_update_fn(cfg.pre_training.loss_type)
            for epoch in range(cfg.pre_training.num_epochs):
                total_eig = 0.0
                batch_count = 0
                for batch in train_loader:
                    batch_jax = {'x': jnp.array(batch['x'].numpy()), 'theta': jnp.array(batch['theta'].numpy())}
                    if cfg.pre_training.loss_type == 'mi_loss':
                        if return_top_k:
                            params, opt_state, EIG, top_thetas, top_scores = pre_train_update_fn(
                                params, next(prng_seq), opt_state, batch_jax
                            )
                            topk_thetas, topk_scores = update_topk_list(
                                topk_thetas, topk_scores,
                                np.array(top_thetas), np.array(top_scores), top_k
                            )
                        else:
                            params, opt_state, EIG = pre_train_update_fn(params, next(prng_seq), opt_state, batch_jax)
                        total_eig += EIG
                        batch_count += 1
                    else:
                        params, opt_state = pre_train_update_fn(params, next(prng_seq), opt_state, batch_jax)
                
                # Compute average EIG and validation loss
                avg_eig = total_eig / batch_count if batch_count > 0 else 0
                avg_val_loss = compute_validation_loss(params, val_loader, eval_fn)
                
                # Validation and logging for pre-training
                avg_val_loss = compute_validation_loss(params, val_loader, eval_fn)
                if cfg.pre_training.loss_type == 'mi_loss':
                    print(f"Epoch {epoch + 1}, Validation Loss: {avg_val_loss:.3f}, Average EIG: {avg_eig:.3f}")
                else:
                    print(f"Epoch {epoch + 1}, Validation Loss: {avg_val_loss:.3f}")
                if cfg.wandb.use_wandb:
                    if cfg.pre_training.loss_type == 'mi_loss' and batch_count > 0:
                        log_round({"pre_training_epoch": epoch + 1, "pre_training_validation_loss": avg_val_loss, "avg_eig": avg_eig})
                    else:
                        log_round({"pre_training_epoch": epoch + 1, "pre_training_validation_loss": avg_val_loss})

        # Main training phase
        main_training_metrics = []

        _, main_train_update_fn = get_loss_and_update_fn(cfg.main_training.loss_type)
        for epoch in range(cfg.main_training.num_epochs):
            total_eig = 0.0
            batch_count = 0
            
            for batch in train_loader:
                batch_jax = {'x': jnp.array(batch['x'].numpy()), 'theta': jnp.array(batch['theta'].numpy())}
                if cfg.main_training.loss_type == 'mi_loss':
                    if return_top_k:
                        params, opt_state, EIG, top_thetas, top_scores = main_train_update_fn(
                            params, next(prng_seq), opt_state, batch_jax
                        )
                        topk_thetas, topk_scores = update_topk_list(
                            topk_thetas, topk_scores,
                            np.array(top_thetas), np.array(top_scores), top_k
                        )
                    else:
                        params, opt_state, EIG = main_train_update_fn(params, next(prng_seq), opt_state, batch_jax)
                    total_eig += EIG
                    batch_count += 1
                else:
                    params, opt_state = main_train_update_fn(params, next(prng_seq), opt_state, batch_jax)

            # Compute average EIG and validation loss
            avg_eig = total_eig / batch_count if batch_count > 0 else 0
            avg_val_loss = compute_validation_loss(params, val_loader, eval_fn)

            if cfg.main_training.loss_type == 'mi_loss':
                print(f"Epoch {epoch + 1}, Validation Loss: {avg_val_loss:.3f}, Average EIG: {avg_eig:.3f}")
                main_training_metrics.append({
                    'epoch': epoch + 1,
                    'avg_eig': float(avg_eig),
                    'validation_loss': float(avg_val_loss)
                })
            else:
                print(f"Epoch {epoch + 1}, Validation Loss: {avg_val_loss:.3f}")
                main_training_metrics.append({
                    'epoch': epoch + 1,
                    'validation_loss': float(avg_val_loss)
                })

            # Log to WandB
            if cfg.wandb.use_wandb:
                if cfg.main_training.loss_type == 'mi_loss' and batch_count > 0:
                    log_round({"epoch": epoch + 1, "validation_loss": avg_val_loss, "avg_eig": avg_eig})
                else:
                    log_round({"epoch": epoch + 1, "validation_loss": avg_val_loss})

        if cfg.hpc:
            main_training_csv_path = os.path.join(subdir, 'main_training_metrics.csv')
            with open(main_training_csv_path, 'w', newline='') as csvfile:
                fieldnames = ['epoch', 'avg_eig', 'validation_loss']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for data in main_training_metrics:
                    writer.writerow(data)

        theta_next = None

        # ---------- Random sampling baseline ----------
        if cfg.active_learning.random_sampling:
            n_acquire = max(1, cfg.mi_params.N)
            rand_key = next(prng_seq)
            theta_next = prior_sampler_mi(rand_key, n_acquire)
            if cfg.wandb.use_wandb:
                theta_mean = jnp.mean(theta_next, axis=0)
                log_round(
                    {
                        "random/theta_next_0": float(theta_mean[0]),
                        "random/theta_next_1": float(theta_mean[1]),
                        "random/theta_next_count": int(theta_next.shape[0]),
                    }
                )
        # ---------- EPIG active learning ----------
        elif return_top_k and cfg.epig.enabled:
            if topk_thetas is None:
                raise RuntimeError("Top-k thetas are empty; run MI loss with top_k > 0 first.")

            theta_targets = jnp.array(topk_thetas)
            if theta_targets.ndim == 3:
                theta_targets = jnp.mean(theta_targets, axis=1)

            def epig_log_prob_fun(params, y, theta, xi, dropout_key):
                theta_b, xi_b, y_b, squeeze = _prepare_theta_xi_for_epig(theta, y)
                lp = log_prob.apply(params, dropout_key, y_b, theta_b, xi_b)
                return lp.squeeze(0) if squeeze else lp

            def epig_sample_fun(params, theta, xi, sample_key, dropout_key):
                theta_b, xi_b, squeeze = _prepare_theta_xi_for_epig(theta)
                y = sample_fn.apply(params, dropout_key, sample_key, 1, theta_b, xi_b)
                return y.squeeze(0) if squeeze else y

            n_acquire = max(1, cfg.mi_params.N)
            if cfg.epig.K < n_acquire:
                print(f"EPIG K={cfg.epig.K} < n_acquire={n_acquire}; using K={n_acquire}.")
            effective_K = max(cfg.epig.K, n_acquire)
            epig_key = next(prng_seq)
            cand_key, score_key = jrandom.split(epig_key, 2)
            theta_candidates = prior_sampler_mi(cand_key, cfg.epig.num_candidates)
            xi = jnp.zeros((1, 0))
            _, scores = lf_epig_scan(
                params,
                xi,
                score_key,
                theta_candidates,
                theta_targets,
                epig_log_prob_fun,
                epig_sample_fun,
                K=effective_K,
                S=cfg.epig.S,
            )
            scores_np = np.array(scores)
            top_idx = np.argsort(scores_np)[-n_acquire:]
            theta_next = jnp.array(theta_candidates[top_idx])
            score_next = float(np.max(scores_np))
            print(f"EPIG best score: {float(score_next):.4f}")
            if cfg.wandb.use_wandb:
                theta_mean = jnp.mean(theta_next, axis=0)
                log_round(
                    {
                        "epig/best_score": float(score_next),
                        "epig/theta_next_0": float(theta_mean[0]),
                        "epig/theta_next_1": float(theta_mean[1]),
                        "epig/theta_next_count": int(theta_next.shape[0]),
                    }
                )

        dropout_diagnostic = None
        if cfg.dropout_diagnostics.enabled:
            print("Running MC-dropout held-out NLL diagnostic...")
            dropout_diag_key = jrandom.fold_in(jrandom.PRNGKey(cfg.seed), active_round + 10_000)
            dropout_diagnostic = compute_dropout_diagnostics(
                params,
                dropout_diag_xs.numpy(),
                dropout_diag_thetas.numpy(),
                dropout_diag_key,
                log_prob.apply,
                eval_log_prob.apply,
                cfg.dropout_diagnostics.num_particles,
                cfg.dropout_diagnostics.num_bins,
            )
            point_diagnostic = dropout_diagnostic["point"]
            parameter_diagnostic = dropout_diagnostic.get("parameter")
            diag_summary = point_diagnostic["summary"]
            diag_bins = point_diagnostic["bins"]
            print(
                "Point dropout diagnostic: "
                f"Spearman(std, NLL)={diag_summary['spearman_corr_std_nll']:.3f}, "
                f"Pearson(std, NLL)={diag_summary['pearson_corr_std_nll']:.3f}, "
                f"mean mixture NLL={diag_summary['mean_mixture_nll']:.3f}"
            )
            if parameter_diagnostic is not None:
                param_summary = parameter_diagnostic["summary"]
                print(
                    "Parameter dropout diagnostic: "
                    f"Spearman(std, deterministic NLL)="
                    f"{param_summary['spearman_corr_std_deterministic_nll']:.3f}, "
                    f"Pearson(std, deterministic NLL)="
                    f"{param_summary['pearson_corr_std_deterministic_nll']:.3f}, "
                    f"mean deterministic NLL={param_summary['mean_deterministic_nll']:.3f}"
                )

            if cfg.hpc:
                summary_path = os.path.join(subdir, "dropout_diagnostic_summary.csv")
                with open(summary_path, "w", newline="") as csvfile:
                    fieldnames = ["active_round"] + list(diag_summary.keys())
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerow({"active_round": active_round + 1, **diag_summary})

                bins_path = os.path.join(subdir, "dropout_diagnostic_bins.csv")
                with open(bins_path, "w", newline="") as csvfile:
                    fieldnames = [
                        "active_round",
                        "bin",
                        "dropout_std_lower",
                        "dropout_std_upper",
                        "dropout_std_mean",
                        "mean_nll",
                        "sem_nll",
                        "count",
                    ]
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    for row in diag_bins:
                        writer.writerow({"active_round": active_round + 1, **row})

                if parameter_diagnostic is not None:
                    param_summary = parameter_diagnostic["summary"]
                    param_summary_path = os.path.join(
                        subdir, "parameter_dropout_diagnostic_summary.csv"
                    )
                    with open(param_summary_path, "w", newline="") as csvfile:
                        fieldnames = ["active_round"] + list(param_summary.keys())
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerow({"active_round": active_round + 1, **param_summary})

                    param_bins_path = os.path.join(
                        subdir, "parameter_dropout_diagnostic_bins.csv"
                    )
                    with open(param_bins_path, "w", newline="") as csvfile:
                        fieldnames = [
                            "active_round",
                            "bin",
                            "dropout_std_lower",
                            "dropout_std_upper",
                            "dropout_std_mean",
                            "mean_deterministic_nll",
                            "sem_mean_deterministic_nll",
                            "count",
                        ]
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()
                        for row in parameter_diagnostic["deterministic_bins"]:
                            writer.writerow({"active_round": active_round + 1, **row})

                    param_mixture_bins_path = os.path.join(
                        subdir, "parameter_dropout_diagnostic_mixture_bins.csv"
                    )
                    with open(param_mixture_bins_path, "w", newline="") as csvfile:
                        fieldnames = [
                            "active_round",
                            "bin",
                            "dropout_std_lower",
                            "dropout_std_upper",
                            "dropout_std_mean",
                            "mean_mixture_nll",
                            "sem_mean_mixture_nll",
                            "count",
                        ]
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()
                        for row in parameter_diagnostic["mixture_bins"]:
                            writer.writerow({"active_round": active_round + 1, **row})

            if cfg.wandb.use_wandb:
                diag_log_data = {
                    f"dropout_diagnostic/point/{key}": value
                    for key, value in diag_summary.items()
                }
                diag_log_data.update({
                    f"dropout_diagnostic/{key}": value
                    for key, value in diag_summary.items()
                })
                original_cwd = hydra.utils.get_original_cwd()
                plot_directory = os.path.join(original_cwd, "temp_plot_data")
                os.makedirs(plot_directory, exist_ok=True)
                plot_path = os.path.join(
                    plot_directory,
                    f"point_dropout_diagnostic_round_{active_round + 1}.png",
                )
                plt.errorbar(
                    [row["dropout_std_mean"] for row in diag_bins],
                    [row["mean_nll"] for row in diag_bins],
                    yerr=[row["sem_nll"] for row in diag_bins],
                    marker="o",
                    capsize=3,
                )
                plt.xlabel("Dropout log-probability std")
                plt.ylabel("Held-out log-mixture NLL")
                plt.title("Held-out NLL vs MC-dropout uncertainty")
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(plot_path)
                plt.close()

                diag_log_data["dropout_diagnostic/point/binned_nll_plot"] = wandb.Image(plot_path)
                diag_log_data["dropout_diagnostic/binned_nll_plot"] = wandb.Image(plot_path)

                if parameter_diagnostic is not None:
                    param_summary = parameter_diagnostic["summary"]
                    diag_log_data.update({
                        f"dropout_diagnostic/parameter/{key}": value
                        for key, value in param_summary.items()
                    })
                    param_bins = parameter_diagnostic["deterministic_bins"]
                    param_plot_path = os.path.join(
                        plot_directory,
                        f"parameter_dropout_diagnostic_round_{active_round + 1}.png",
                    )
                    plt.errorbar(
                        [row["dropout_std_mean"] for row in param_bins],
                        [row["mean_deterministic_nll"] for row in param_bins],
                        yerr=[row["sem_mean_deterministic_nll"] for row in param_bins],
                        marker="o",
                        capsize=3,
                    )
                    plt.xlabel("Mean dropout log-probability std per theta")
                    plt.ylabel("Mean held-out deterministic NLL per theta")
                    plt.title("Parameter-level held-out NLL vs MC-dropout uncertainty")
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(param_plot_path)
                    plt.close()

                    diag_log_data["dropout_diagnostic/parameter/binned_nll_plot"] = wandb.Image(
                        param_plot_path
                    )

                    param_mixture_bins = parameter_diagnostic["mixture_bins"]
                    param_mixture_plot_path = os.path.join(
                        plot_directory,
                        f"parameter_dropout_mixture_diagnostic_round_{active_round + 1}.png",
                    )
                    plt.errorbar(
                        [row["dropout_std_mean"] for row in param_mixture_bins],
                        [row["mean_mixture_nll"] for row in param_mixture_bins],
                        yerr=[row["sem_mean_mixture_nll"] for row in param_mixture_bins],
                        marker="o",
                        capsize=3,
                    )
                    plt.xlabel("Mean dropout log-probability std per theta")
                    plt.ylabel("Mean held-out log-mixture NLL per theta")
                    plt.title("Parameter-level log-mixture NLL vs MC-dropout uncertainty")
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(param_mixture_plot_path)
                    plt.close()

                    diag_log_data["dropout_diagnostic/parameter/binned_mixture_nll_plot"] = (
                        wandb.Image(param_mixture_plot_path)
                    )
                log_round(diag_log_data)

        # compute calibration error and LC2ST
        # sbc first
        prior_samples = jnp.array(thetas.numpy())[:cfg.final_sbc_samples]
        observed_data = jnp.array(xs.numpy())[:cfg.final_sbc_samples]
        
        @jax.jit
        def log_likelihood_fn(params, x, theta):
            xi = jnp.zeros((1, 0))
            return eval_log_prob.apply(params, x, theta, xi)
        
        @jax.jit
        def log_prior_fn(theta):
            lower = jnp.array([-1.0, -1.0])
            upper = jnp.array([1.0, 1.0])
            prior_lps = -jnp.sum(jnp.log(upper - lower))
            n_samples = theta.shape[0]
            prior_lps = jnp.full((n_samples,), prior_lps)
            return prior_lps
        
        # Define the posterior log-probability function for this x_i
        def mcmc_posterior(theta_unconstrained, x_obs):
            theta = unconstrained_to_constrained(theta_unconstrained)

            # Create dummy xi array
            xi = jnp.zeros((1, 0))
            
            # Compute log-likelihood
            log_likelihood = eval_log_prob.apply(params, x_obs[None, :], theta[None, :], xi).squeeze()

            # Compute log-prior (uniform between -1 and 1)
            log_prior = jnp.where(
                (theta >= -1.0) & (theta <= 1.0),
                -jnp.log(2.0),
                -jnp.inf
            ).sum()

            # Add the log determinant of the Jacobian from the logit transformation
            logdetjac = unconstrained_to_constrained_logdetjac(theta_unconstrained)

            # Return log-posterior adjusted with logdet Jacobian
            log_post = log_likelihood + log_prior + logdetjac

            return log_post.squeeze()

        print("Running SBC...")
        ranks, empirical_coverage, levels, lower_bounds, upper_bounds = run_sbc_two_moons(
            prior_samples,
            observed_data,
            log_likelihood_fn=log_likelihood_fn,
            log_prior_fn=log_prior_fn,
            mcmc_fn=run_mcmc_two_moons,
            num_adapt_steps=500,
            num_mcmc_samples=1000,
            theta_train_mean=theta_train_mean,
            theta_train_std=theta_train_std,
            params=params,
            prng_seq=jrandom.PRNGKey(cfg.seed),
            mcmc_posterior=mcmc_posterior,
            show_progress=True,
        )

        # ---------- LC2ST ----------
        # observe a data point
        print("Running LC2ST...")
        observation = task.get_observation(num_observation=1)
        x_o = observation

        prng_key, mcmc_key = jrandom.split(jrandom.PRNGKey(cfg.seed), 2)
        mcmc_obs = partial(mcmc_posterior, x_obs=jnp.array(x_o.numpy()).squeeze())
        theta_init = jnp.array(thetas.mean(0).numpy())
        post_samples, _ = run_mcmc_two_moons(
            mcmc_key, mcmc_obs, theta_init=theta_init, num_adapt_steps=500, num_mcmc_samples=1000
        )

        post_samples_np = np.array(post_samples)

        # Convert NumPy arrays to PyTorch tensors
        post_samples_torch = torch.from_numpy(post_samples_np).float()

        lc2st_thetas = thetas
        lc2st_xs = xs
        lc2st_post = post_samples_torch
        if lc2st_thetas.shape[0] != lc2st_post.shape[0]:
            lc2st_n = lc2st_post.shape[0]
            lc2st_thetas = prior(lc2st_n)
            lc2st_xs = simulator(lc2st_thetas)

        if lc2st_thetas.shape[0] < 2:
            print("Skipping LC2ST: need at least 2 joint samples.")
            statistic = np.nan
            p_value = np.nan
            reject = False
        else:
            # Initialize the LC2ST object
            lc2st = LC2ST(
                thetas=lc2st_thetas,
                xs=lc2st_xs,
                posterior_samples=lc2st_post,
                seed=cfg.seed,
                num_folds=1,
                num_ensemble=1,
                classifier="mlp",
                z_score=False,
                num_trials_null=100,
                permutation=True,
            )

            # Train the classifiers under the null hypothesis
            lc2st.train_under_null_hypothesis()
            # Train the classifier on observed data
            lc2st.train_on_observed_data()

            # Define the observed data point x_o and the corresponding posterior samples theta_o
            # Assuming x_o is a single observation (e.g., the first one)
            theta_o = lc2st_post  # Samples from the posterior conditioned on x_o

            # Compute the L-C2ST statistic on observed data
            statistic = lc2st.get_statistic_on_observed_data(theta_o=theta_o, x_o=x_o)
            print("L-C2ST statistic on observed data:", statistic)

            # Compute the p-value for the test
            p_value = lc2st.p_value(theta_o=theta_o, x_o=x_o)
            print("P-value for L-C2ST:", p_value)

            # Decide whether to reject the null hypothesis at a significance level alpha
            alpha = 0.05  # 95% confidence level
            reject = lc2st.reject_test(theta_o=theta_o, x_o=x_o, alpha=alpha)
            print(f"Reject null hypothesis at alpha = {alpha}:", reject)

        # ---------- C2ST ----------
        print("Running C2ST...")
        reference_posterior = task.get_reference_posterior_samples(num_observation=1)
        n_compare = min(post_samples_torch.shape[0], reference_posterior.shape[0])
        post_samples_c2st = post_samples_torch[:n_compare]
        reference_samples_c2st = reference_posterior[:n_compare]
        c2st_result = c2st(
            post_samples_c2st,
            reference_samples_c2st,
            seed=cfg.seed,
            n_folds=5,
            classifier="mlp",
            z_score=False,
        )
        c2st_accuracy = float(c2st_result[0]) if isinstance(c2st_result, (tuple, list)) else float(c2st_result)
        print("C2ST accuracy:", c2st_accuracy)

        # Save trained parameters
        if cfg.hpc:
            objects = {
                    'params': jax.device_get(params),
                    'post_samples': jax.device_get(post_samples),
                    'ranks': jax.device_get(ranks),
                    'empirical_coverage': jax.device_get(empirical_coverage),
                    'levels': jax.device_get(levels),
                    'lower_bounds': jax.device_get(lower_bounds),
                    'upper_bounds': jax.device_get(upper_bounds),
                    'observation': observation,
                    'LC2ST_statistic': statistic,
                    'LC2ST_p_value': p_value,
                    'LC2ST_reject': reject,
                    'C2ST_accuracy': c2st_accuracy,
                    'dropout_diagnostic': dropout_diagnostic,
                    'mi_params_N': N,
                    'eig_lambda': eig_lambda
                }
            with open(f"{subdir}/{cfg.experiment.save_name}.pkl", "wb") as f:
                pickle.dump(objects, f)

        # TODO: make a helper function to plot these
        if cfg.wandb.use_wandb:
            plt.plot(levels, empirical_coverage, label='Empirical Coverage', color='blue')
            plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
            plt.fill_between(levels, lower_bounds, upper_bounds, color='grey', alpha=0.2, label='95% Confidence Region')
            plt.xlabel('Credibility Level (1 - α)')
            plt.ylabel('Expected Coverage')
            plt.grid(True)
            original_cwd = hydra.utils.get_original_cwd()
            plot_directory = os.path.join(original_cwd, 'temp_plot_data')
            os.makedirs(plot_directory, exist_ok=True)
            plot_path1 = os.path.join(plot_directory, 'sbc_plot.png')
            plt.savefig(plot_path1)
            plt.close()
            # TODO: plot the two moons to see if you're getting mode collapse or not
            plt.hist2d(post_samples_np[:,0], post_samples_np[:,1], range=[[-1, 1], [-1, 1]], bins=100)[-1]
            original_cwd = hydra.utils.get_original_cwd()
            plot_directory = os.path.join(original_cwd, 'temp_plot_data')
            os.makedirs(plot_directory, exist_ok=True)
            plot_path2 = os.path.join(plot_directory, '2moons_post_hist.png')
            plt.savefig(plot_path2)
            plt.close()
            log_round({
                "SBC": wandb.Image(plot_path1),
                "Posterior": wandb.Image(plot_path2),
                "LC2ST stat": statistic,
                "LC2ST p value": p_value,
            })
            log_round({"c2st": c2st_accuracy})
            wandb.log({"c2st": c2st_accuracy, "active_round": active_round + 1}, step=active_round + 1)

        return theta_next, c2st_accuracy, subdir

    c2st_history = []
    for active_round in range(cfg.active_learn_rounds):
        theta_next, c2st_accuracy, subdir = run_active_round(active_round, thetas, xs)
        c2st_history.append(c2st_accuracy)
        save_dir = subdir if subdir is not None else os.getcwd()
        c2st_path = os.path.join(save_dir, "c2st_over_rounds.npy")
        np.save(c2st_path, np.array(c2st_history))
        print(f"C2ST history saved to: {c2st_path}")
        if active_round < cfg.active_learn_rounds - 1:
            if theta_next is None:
                raise RuntimeError("EPIG did not produce a theta_next for active learning.")
            theta_next_np = np.array(theta_next)
            if theta_next_np.ndim == 1:
                theta_next_np = theta_next_np[None, :]
            theta_next_torch = torch.tensor(theta_next_np, dtype=thetas.dtype)
            x_next = simulator(theta_next_torch)
            thetas = torch.cat([thetas, theta_next_torch], dim=0)
            xs = torch.cat([xs, x_next], dim=0)
    if cfg.wandb.use_wandb:
        c2st_table = wandb.Table(columns=["active_round", "c2st"])
        for idx, value in enumerate(c2st_history, start=1):
            c2st_table.add_data(idx, float(value))
        wandb.log({
            "c2st_history": c2st_table,
            "c2st_history_plot": wandb.plot.line(
                c2st_table,
                "active_round",
                "c2st",
                title="C2ST over rounds",
            ),
        })

if __name__ == "__main__":
    main()

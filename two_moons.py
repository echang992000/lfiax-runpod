import os
import time
import pickle
import csv
import json
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import jax.random as jrandom
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
from lfiax.utils.sbi_utils import run_sbc_two_moons

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

from typing import Mapping, Any, Tuple

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


@hydra.main(version_base=None, config_path=".", config_name="config_two_moons")
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

    # Generate data
    num_samples = cfg.simulator.num_samples
    if num_samples == 1:
        num_samples = cfg.mi_params.N
    thetas = prior(num_samples)  # PyTorch tensor
    xs = simulator(thetas)       # PyTorch tensor

    # Split data into training and validation sets
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

    train_dataset = CustomDataset(xs_train, thetas_train)
    val_dataset = CustomDataset(xs_val, thetas_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)

    # Use raw values (no normalization)
    x_train_mean = jnp.zeros(EVENT_SHAPE)
    x_train_std = jnp.ones(EVENT_SHAPE)
    theta_train_mean = jnp.zeros(theta_shape)
    theta_train_std = jnp.ones(theta_shape)

    # contrastive learning parameters
    N = cfg.mi_params.N
    M = cfg.mi_params.M
    eig_lambda = cfg.mi_params.eig_lambda

    # set save directory if on the hpc
    current_time = time.localtime()
    current_time_str = (
        f"{current_time.tm_year}.{current_time.tm_mon:02d}.{current_time.tm_mday:02d}."
        f"{current_time.tm_hour:02d}.{current_time.tm_min:02d}"
    )
    if cfg.hpc:
        work_dir = "/pub/vzaballa/lfiax_data"
        print(f'workspace: {work_dir}')
        subdir = os.path.join(
            work_dir,
            "two_moons",
            f"pretrain_{cfg.pre_training.enabled}",
            cfg.pre_training.loss_type,
            cfg.main_training.loss_type,
            str(cfg.seed),
            current_time_str
        )
    else:
        original_cwd = hydra.utils.get_original_cwd()
        subdir = os.path.join(
            original_cwd,
            "outputs",
            "two_moons",
            f"pretrain_{cfg.pre_training.enabled}",
            cfg.pre_training.loss_type,
            cfg.main_training.loss_type,
            str(cfg.seed),
            current_time_str
        )
        print(f"local output dir: {subdir}")
    os.makedirs(subdir, exist_ok=True)
    
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
    params = log_prob.init(
        next(prng_seq),
        jnp.zeros((1, *EVENT_SHAPE)),
        jnp.zeros((1, *theta_shape)),
        jnp.zeros((1, 0)),
    )
    optimizer = optax.adamw(learning_rate)
    opt_state = optimizer.init(params)

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
                x_train_mean, x_train_std, theta_train_mean, theta_train_std, prior_sampler_mi)
            update_fn = lambda params, prng_key, opt_state, batch: mi_update(
                    params, prng_key, opt_state, batch, loss_fn, optimizer)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        return loss_fn, update_fn

    # Pre-training phase
    if cfg.pre_training.enabled:
        _, pre_train_update_fn = get_loss_and_update_fn(cfg.pre_training.loss_type)
        for epoch in range(cfg.pre_training.num_epochs):
            total_eig = 0.0
            batch_count = 0
            for batch in train_loader:
                batch_jax = {'x': jnp.array(batch['x'].numpy()), 'theta': jnp.array(batch['theta'].numpy())}
                if cfg.pre_training.loss_type == 'mi_loss':
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
                    wandb_log_data = {"pre_training_epoch": epoch + 1, "pre_training_validation_loss": avg_val_loss, "avg_eig": avg_eig}
                else:
                    wandb_log_data = {"pre_training_epoch": epoch + 1, "pre_training_validation_loss": avg_val_loss}
                wandb.log(wandb_log_data)

    # Main training phase
    main_training_metrics = []
    main_training_val_logprobs = []
    main_training_avg_eigs = []

    _, main_train_update_fn = get_loss_and_update_fn(cfg.main_training.loss_type)
    for epoch in range(cfg.main_training.num_epochs):
        total_eig = 0.0
        batch_count = 0
        
        for batch in train_loader:
            batch_jax = {'x': jnp.array(batch['x'].numpy()), 'theta': jnp.array(batch['theta'].numpy())}
            if cfg.main_training.loss_type == 'mi_loss':
                params, opt_state, EIG = main_train_update_fn(params, next(prng_seq), opt_state, batch_jax)
                total_eig += EIG
                batch_count += 1
            else:
                params, opt_state = main_train_update_fn(params, next(prng_seq), opt_state, batch_jax)

        # Compute average EIG and validation loss
        avg_eig = total_eig / batch_count if batch_count > 0 else 0
        avg_val_loss = compute_validation_loss(params, val_loader, eval_fn)

        val_logprob = -avg_val_loss
        main_training_val_logprobs.append(float(val_logprob))
        main_training_avg_eigs.append(float(avg_eig))

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
            # Log EIG if loss_type is mi_loss
            if cfg.main_training.loss_type == 'mi_loss' and batch_count > 0:
                wandb_log_data = {"epoch": epoch + 1, "validation_loss": avg_val_loss, "avg_eig": avg_eig}
            else:
                wandb_log_data = {"epoch": epoch + 1, "validation_loss": avg_val_loss}
            
            wandb.log(wandb_log_data)

    main_training_csv_path = os.path.join(subdir, 'main_training_metrics.csv')
    with open(main_training_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['epoch', 'avg_eig', 'validation_loss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for data in main_training_metrics:
            writer.writerow(data)
    
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
    post_samples, _ = run_mcmc_two_moons(mcmc_key, mcmc_obs, theta_init=theta_init, num_adapt_steps=500, num_mcmc_samples=1000)

    post_samples_np = np.array(post_samples)

    # Convert NumPy arrays to PyTorch tensors
    post_samples_torch = torch.from_numpy(post_samples_np).float()

    # Initialize the LC2ST object
    lc2st = LC2ST(
        thetas=thetas,
        xs=xs,
        posterior_samples=post_samples_torch,
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
    theta_o = post_samples_torch  # Samples from the posterior conditioned on x_o

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
    if cfg.wandb.use_wandb:
        wandb.log({"C2ST_accuracy": c2st_accuracy})

    arrays_path = os.path.join(subdir, 'main_training_arrays.npz')
    np.savez(
        arrays_path,
        validation_logprob=np.array(main_training_val_logprobs),
        avg_eig=np.array(main_training_avg_eigs),
    )

    final_metrics = {
        "C2ST_accuracy": c2st_accuracy,
        "LC2ST_statistic": statistic,
        "LC2ST_p_value": p_value,
        "LC2ST_reject": reject,
    }
    final_metrics_path = os.path.join(subdir, "final_metrics.json")
    with open(final_metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=2, sort_keys=True)

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
                'mi_params_N': cfg.mi_params.N,
                'eig_lambda': cfg.mi_params.eig_lambda,
                'main_training_validation_logprob': np.array(main_training_val_logprobs),
                'main_training_avg_eig': np.array(main_training_avg_eigs),
                'final_metrics_path': final_metrics_path,
                'main_training_arrays_path': arrays_path,
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
        wandb.log({f'SBC': wandb.Image(plot_path1),
                   f'Posterior': wandb.Image(plot_path2),
                    f"LC2ST stat": statistic,
                    f"LC2ST p value": p_value,
                    f"C2ST accuracy": c2st_accuracy,
                    })

if __name__ == "__main__":
    main()

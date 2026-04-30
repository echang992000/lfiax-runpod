from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.lax as lax
import jax.random as jrandom
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import distrax
from lfiax.flows.nsf import make_nsf

import matplotlib.pyplot as plt

from typing import List, Optional, Tuple, Union

import haiku as hk
import numpy as np
from scipy.stats import binom
import tensorflow_datasets as tfds
from scipy.stats import t

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


def plot_contour_prior_posterior(prior_samples, posterior_samples, true_theta, filename):
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Estimate the density for the prior samples
    x_prior = prior_samples[:, 0]
    y_prior = prior_samples[:, 1]
    kde_prior = gaussian_kde(np.vstack([x_prior, y_prior]))
    density_prior = kde_prior(np.vstack([x_prior, y_prior]))

    # Create a grid of x and y values
    x_grid_prior, y_grid_prior = np.meshgrid(np.linspace(x_prior.min(), x_prior.max(), num=100),
                                             np.linspace(y_prior.min(), y_prior.max(), num=100))
    z_grid_prior = kde_prior(np.vstack([x_grid_prior.ravel(), y_grid_prior.ravel()]))
    density_prior = z_grid_prior.reshape(x_grid_prior.shape)

    # Generate the contour plot for the prior samples in the first subplot
    levels_prior = np.linspace(density_prior.min(), density_prior.max(), num=10)
    ax1.contour(x_grid_prior, y_grid_prior, density_prior, levels=levels_prior, cmap='viridis')
    ax1.set_title("Prior Contour Density")
    ax1.set_xlabel("\u03B8\u2081")
    ax1.set_ylabel("\u03B8\u2080")

    # Estimate the density for the posterior samples
    x_posterior = posterior_samples[:, 0]
    y_posterior = posterior_samples[:, 1]
    kde_posterior = gaussian_kde(np.vstack([x_posterior, y_posterior]))
    density_posterior = kde_posterior(np.vstack([x_posterior, y_posterior]))

    # Create a grid of x and y values
    x_grid_posterior, y_grid_posterior = np.meshgrid(np.linspace(x_posterior.min(), x_posterior.max(), num=100),
                                                     np.linspace(y_posterior.min(), y_posterior.max(), num=100))
    z_grid_posterior = kde_posterior(np.vstack([x_grid_posterior.ravel(), y_grid_posterior.ravel()]))
    density_posterior = z_grid_posterior.reshape(x_grid_posterior.shape)

    # Generate the contour plot for the posterior samples in the second subplot
    levels_posterior = np.linspace(density_posterior.min(), density_posterior.max(), num=10)
    ax2.contour(x_grid_posterior, y_grid_posterior, density_posterior, levels=levels_posterior, cmap='viridis')
    ax2.set_title("Posterior Contour Density")
    ax2.set_xlabel("\u03B8\u2081")
    ax2.set_ylabel("\u03B8\u2080")


def plot_prior_posteriors(prior_samples, posterior_samples, posterior_samples1, posterior_samples2, true_theta, filename,
                         bandwidth=0.5):
    # Create a figure with four subplots in a row
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Estimate the density for the prior samples
    x_prior = prior_samples[:, 0]
    y_prior = prior_samples[:, 1]
    kde_prior = gaussian_kde(np.vstack([x_prior, y_prior]), bw_method=bandwidth)
    density_prior = kde_prior(np.vstack([x_prior, y_prior]))

    # Create a grid of x and y values for the prior samples
    x_grid_prior, y_grid_prior = np.meshgrid(np.linspace(x_prior.min(), x_prior.max(), num=100),
                                             np.linspace(y_prior.min(), y_prior.max(), num=100))
    z_grid_prior = kde_prior(np.vstack([x_grid_prior.ravel(), y_grid_prior.ravel()]))
    density_prior = z_grid_prior.reshape(x_grid_prior.shape)

    # Generate the contour plot for the prior samples
    levels_prior = np.linspace(density_prior.min(), density_prior.max(), num=10)
    axes[0].contour(x_grid_prior, y_grid_prior, density_prior, levels=levels_prior, cmap='viridis')
    axes[0].set_title("Prior Density")
    axes[0].set_xlabel("\u03B8\u2081")
    axes[0].set_ylabel("\u03B8\u2080")
    axes[0].set_xlim(-10, 10)
    axes[0].set_ylim(-10, 10)
    axes[0].plot(5, 2, marker='x', color='red', markersize=10)

    # Define the posterior samples and their labels
    posteriors = [
        {"samples": posterior_samples, "label": "D=1"},
        {"samples": posterior_samples1, "label": "D=10"},
        {"samples": posterior_samples2, "label": "D=100"}
    ]

    # Iterate over the posteriors and generate the contour plots
    for i, posterior in enumerate(posteriors, start=1):
        samples = posterior["samples"]
        x_posterior = samples[:, 0]
        y_posterior = samples[:, 1]
        kde_posterior = gaussian_kde(np.vstack([x_posterior, y_posterior]), bw_method=bandwidth)
        density_posterior = kde_posterior(np.vstack([x_posterior, y_posterior]))

        # Create a grid of x and y values for the posterior samples
        x_grid_posterior, y_grid_posterior = np.meshgrid(np.linspace(x_posterior.min(), x_posterior.max(), num=100),
                                                         np.linspace(y_posterior.min(), y_posterior.max(), num=100))
        z_grid_posterior = kde_posterior(np.vstack([x_grid_posterior.ravel(), y_grid_posterior.ravel()]))
        density_posterior = z_grid_posterior.reshape(x_grid_posterior.shape)

        # Generate the contour plot for the posterior samples
        levels_posterior = np.linspace(density_posterior.min(), density_posterior.max(), num=10)
        ax = axes[i]
        ax.contour(x_grid_posterior, y_grid_posterior, density_posterior, levels=levels_posterior, cmap='viridis')
        ax.set_title(f"{posterior['label']} Density")
        ax.set_xlabel("\u03B8\u2081")
        # ax.set_ylabel("\u03B8\u2080")
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.plot(5, 2, marker='x', color='red', markersize=10)

    # Adjust the spacing between subplots
    # plt.subplots_adjust(wspace=0.4)

    # Save the plot as a PNG file with the provided filename
    plt.savefig(filename, dpi=900, bbox_inches='tight')
    

def plot_prior_posterior(prior_samples, posterior_samples, true_theta, filename):
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Plot the first histogram in the first subplot
    ax1.hist(prior_samples[:, 0], bins=30)
    ax1.hist(posterior_samples[:, 0], bins=30, alpha=0.5, color='orange', label='Posterior')
    ax1.axvline(posterior_samples[:, 0].mean(), color='g', linestyle='--')
    ax1.axvline(true_theta[0][0], color='r', linestyle='--')
    ax1.set_title("\u03B8\u2081")
    ax1.set_xlabel("Value")
    ax1.set_ylabel("Frequency")

    # Plot the second histogram in the second subplot
    ax2.hist(prior_samples[:, 1], bins=30)
    ax2.hist(posterior_samples[:, 1], bins=30, alpha=0.5, color='orange', label='Posterior')
    ax2.axvline(posterior_samples[:, 1].mean(), color='g', linestyle='--')
    ax2.axvline(true_theta[0][1], color='r', linestyle='--')
    ax2.set_title("\u03B8\u2080")
    ax2.set_xlabel("Value")
    ax2.set_ylabel("Frequency")

    # Save the plot as a PNG file with the provided filename
    plt.savefig(filename, dpi=300, bbox_inches='tight')

    # Display the figure
    # plt.show()


def save_posterior_marginal(posterior_samples_marginal, filename):
    # Create a figure
    fig = plt.figure(figsize=(8, 6))
    
    # Plot the histogram
    plt.hist(posterior_samples_marginal, bins=50)
    
    # Set labels and title
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Posterior Marginal")
    
    # Save the plot as a PNG file with the provided filename
    plt.savefig(filename, dpi=300, bbox_inches='tight')


def plot_posteriors(samples: Array, true_theta: Tuple[float, float], outdir: Path) -> Path:
    """2‑D density contour plot (old logic but isolated)."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "posterior_kde.png"
    kde = gaussian_kde(samples.T)
    grid = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(grid, grid)
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
    plt.contourf(X, Y, Z, cmap="gray_r")
    plt.plot(*true_theta, "rx", ms=12)
    plt.xlabel("K")
    plt.ylabel("ε")
    plt.savefig(path)
    plt.close()
    return path

def plot_calibration(levels: Array,
                     empirical: Array,
                     lb: Array,
                     ub: Array,
                     outdir: Path) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "calibration.png"
    plt.figure(figsize=(6, 6))
    plt.plot(levels, empirical, label="Empirical")
    plt.plot([0, 1], [0, 1], "k--")
    plt.fill_between(levels, lb, ub, alpha=0.2)
    plt.xlabel("Credibility level (1-α)")
    plt.ylabel("Coverage")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path

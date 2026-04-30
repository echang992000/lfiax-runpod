import math

import numpy as np


def _rankdata_average_ties(values):
    values = np.asarray(values)
    sorter = np.argsort(values, kind="mergesort")
    sorted_values = values[sorter]
    ranks = np.empty(values.shape[0], dtype=np.float64)
    start = 0
    while start < values.shape[0]:
        end = start + 1
        while end < values.shape[0] and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[sorter[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _safe_correlation(x, y, method="pearson"):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.shape[0] < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return np.nan
    if method == "spearman":
        x = _rankdata_average_ties(x)
        y = _rankdata_average_ties(y)
        if np.std(x) == 0.0 or np.std(y) == 0.0:
            return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def summarize_dropout_diagnostics(log_probs, num_bins):
    """Summarize MC-dropout log likelihood particles for held-out pairs."""
    log_probs = np.asarray(log_probs, dtype=np.float64)
    if log_probs.ndim != 2:
        raise ValueError(f"log_probs must have shape [num_particles, num_points], got {log_probs.shape}")
    num_particles, num_points = log_probs.shape
    if num_particles < 1 or num_points < 1:
        raise ValueError("log_probs must contain at least one particle and one validation point.")

    dropout_std = np.std(log_probs, axis=0)
    dropout_var = np.var(log_probs, axis=0)
    mean_nll = -np.mean(log_probs, axis=0)
    mixture_nll = -(np.logaddexp.reduce(log_probs, axis=0) - math.log(num_particles))

    effective_bins = max(1, min(int(num_bins), num_points))
    sorted_indices = np.argsort(dropout_std, kind="mergesort")
    bin_indices = np.array_split(sorted_indices, effective_bins)
    bins = []
    for bin_id, indices in enumerate(bin_indices):
        bin_std = dropout_std[indices]
        bin_nll = mixture_nll[indices]
        count = int(indices.shape[0])
        sem_nll = float(np.std(bin_nll, ddof=1) / math.sqrt(count)) if count > 1 else 0.0
        bins.append({
            "bin": bin_id,
            "dropout_std_lower": float(np.min(bin_std)),
            "dropout_std_upper": float(np.max(bin_std)),
            "dropout_std_mean": float(np.mean(bin_std)),
            "mean_nll": float(np.mean(bin_nll)),
            "sem_nll": sem_nll,
            "count": count,
        })

    summary = {
        "spearman_corr_std_nll": _safe_correlation(dropout_std, mixture_nll, method="spearman"),
        "pearson_corr_std_nll": _safe_correlation(dropout_std, mixture_nll, method="pearson"),
        "mean_mixture_nll": float(np.mean(mixture_nll)),
        "mean_mean_nll": float(np.mean(mean_nll)),
        "mean_dropout_std": float(np.mean(dropout_std)),
        "mean_dropout_var": float(np.mean(dropout_var)),
        "num_particles": int(num_particles),
        "num_validation_points": int(num_points),
    }
    return {
        "summary": summary,
        "bins": bins,
        "dropout_std": dropout_std,
        "dropout_var": dropout_var,
        "mean_nll": mean_nll,
        "mixture_nll": mixture_nll,
    }


def _make_uncertainty_bins(uncertainty, loss, num_bins, loss_key="mean_nll"):
    effective_bins = max(1, min(int(num_bins), uncertainty.shape[0]))
    sorted_indices = np.argsort(uncertainty, kind="mergesort")
    bin_indices = np.array_split(sorted_indices, effective_bins)
    bins = []
    for bin_id, indices in enumerate(bin_indices):
        bin_uncertainty = uncertainty[indices]
        bin_loss = loss[indices]
        count = int(indices.shape[0])
        sem_loss = float(np.std(bin_loss, ddof=1) / math.sqrt(count)) if count > 1 else 0.0
        bins.append({
            "bin": bin_id,
            "dropout_std_lower": float(np.min(bin_uncertainty)),
            "dropout_std_upper": float(np.max(bin_uncertainty)),
            "dropout_std_mean": float(np.mean(bin_uncertainty)),
            loss_key: float(np.mean(bin_loss)),
            f"sem_{loss_key}": sem_loss,
            "count": count,
        })
    return bins


def summarize_parameter_dropout_diagnostics(log_probs, deterministic_log_probs, num_bins):
    """Summarize MC-dropout diagnostics after averaging repeated simulations per theta.

    Args:
        log_probs: MC-dropout log probabilities with shape
            [num_particles, num_theta, simulations_per_theta].
        deterministic_log_probs: No-dropout log probabilities with shape
            [num_theta, simulations_per_theta].
    """
    log_probs = np.asarray(log_probs, dtype=np.float64)
    deterministic_log_probs = np.asarray(deterministic_log_probs, dtype=np.float64)
    if log_probs.ndim != 3:
        raise ValueError(
            "log_probs must have shape [num_particles, num_theta, simulations_per_theta], "
            f"got {log_probs.shape}"
        )
    num_particles, num_theta, simulations_per_theta = log_probs.shape
    if deterministic_log_probs.shape != (num_theta, simulations_per_theta):
        raise ValueError(
            "deterministic_log_probs must have shape [num_theta, simulations_per_theta], "
            f"got {deterministic_log_probs.shape}"
        )
    if num_particles < 1 or num_theta < 1 or simulations_per_theta < 1:
        raise ValueError("log_probs must contain at least one particle, theta, and simulation.")

    pair_dropout_std = np.std(log_probs, axis=0)
    pair_dropout_var = np.var(log_probs, axis=0)
    pair_mixture_nll = -(np.logaddexp.reduce(log_probs, axis=0) - math.log(num_particles))
    pair_mean_nll = -np.mean(log_probs, axis=0)
    pair_deterministic_nll = -deterministic_log_probs

    theta_dropout_std = np.mean(pair_dropout_std, axis=1)
    theta_dropout_var = np.mean(pair_dropout_var, axis=1)
    theta_deterministic_nll = np.mean(pair_deterministic_nll, axis=1)
    theta_mixture_nll = np.mean(pair_mixture_nll, axis=1)
    theta_mean_nll = np.mean(pair_mean_nll, axis=1)

    det_bins = _make_uncertainty_bins(
        theta_dropout_std,
        theta_deterministic_nll,
        num_bins,
        loss_key="mean_deterministic_nll",
    )
    mixture_bins = _make_uncertainty_bins(
        theta_dropout_std,
        theta_mixture_nll,
        num_bins,
        loss_key="mean_mixture_nll",
    )

    summary = {
        "spearman_corr_std_deterministic_nll": _safe_correlation(
            theta_dropout_std, theta_deterministic_nll, method="spearman"
        ),
        "pearson_corr_std_deterministic_nll": _safe_correlation(
            theta_dropout_std, theta_deterministic_nll, method="pearson"
        ),
        "spearman_corr_std_mixture_nll": _safe_correlation(
            theta_dropout_std, theta_mixture_nll, method="spearman"
        ),
        "pearson_corr_std_mixture_nll": _safe_correlation(
            theta_dropout_std, theta_mixture_nll, method="pearson"
        ),
        "mean_deterministic_nll": float(np.mean(theta_deterministic_nll)),
        "mean_mixture_nll": float(np.mean(theta_mixture_nll)),
        "mean_mean_nll": float(np.mean(theta_mean_nll)),
        "mean_dropout_std": float(np.mean(theta_dropout_std)),
        "mean_dropout_var": float(np.mean(theta_dropout_var)),
        "num_particles": int(num_particles),
        "num_validation_thetas": int(num_theta),
        "simulations_per_theta": int(simulations_per_theta),
    }
    return {
        "summary": summary,
        "deterministic_bins": det_bins,
        "mixture_bins": mixture_bins,
        "theta_dropout_std": theta_dropout_std,
        "theta_dropout_var": theta_dropout_var,
        "theta_deterministic_nll": theta_deterministic_nll,
        "theta_mixture_nll": theta_mixture_nll,
        "theta_mean_nll": theta_mean_nll,
    }

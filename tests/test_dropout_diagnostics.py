import math

import numpy as np


from lfiax.utils.dropout_diagnostics import (
    summarize_dropout_diagnostics,
    summarize_parameter_dropout_diagnostics,
)


def test_summarize_dropout_diagnostics_shapes_and_mixture_nll():
    log_probs = np.array([
        [0.0, -2.0, -4.0],
        [-1.0, -1.0, -5.0],
        [-2.0, -3.0, -6.0],
    ])

    result = summarize_dropout_diagnostics(log_probs, num_bins=2)
    expected_mixture_nll = -(np.logaddexp.reduce(log_probs, axis=0) - math.log(log_probs.shape[0]))

    assert result["dropout_std"].shape == (3,)
    assert result["dropout_var"].shape == (3,)
    assert result["mean_nll"].shape == (3,)
    assert result["mixture_nll"].shape == (3,)
    assert np.allclose(result["mixture_nll"], expected_mixture_nll)
    assert len(result["bins"]) == 2
    assert sum(row["count"] for row in result["bins"]) == 3


def test_summarize_dropout_diagnostics_constant_uncertainty_returns_nan_corr():
    log_probs = np.ones((4, 5))

    result = summarize_dropout_diagnostics(log_probs, num_bins=3)

    assert math.isnan(result["summary"]["spearman_corr_std_nll"])
    assert math.isnan(result["summary"]["pearson_corr_std_nll"])


def test_summarize_parameter_dropout_diagnostics_averages_over_simulations():
    log_probs = np.array([
        [[0.0, -1.0], [-2.0, -3.0], [-4.0, -5.0]],
        [[-1.0, -2.0], [-1.0, -2.0], [-5.0, -6.0]],
        [[-2.0, -3.0], [-3.0, -4.0], [-6.0, -7.0]],
    ])
    deterministic_log_probs = np.array([
        [-0.5, -1.5],
        [-2.5, -3.5],
        [-4.5, -5.5],
    ])

    result = summarize_parameter_dropout_diagnostics(log_probs, deterministic_log_probs, num_bins=2)
    expected_pair_std = np.std(log_probs, axis=0)
    expected_theta_std = expected_pair_std.mean(axis=1)
    expected_theta_det_nll = (-deterministic_log_probs).mean(axis=1)

    assert result["theta_dropout_std"].shape == (3,)
    assert result["theta_deterministic_nll"].shape == (3,)
    assert np.allclose(result["theta_dropout_std"], expected_theta_std)
    assert np.allclose(result["theta_deterministic_nll"], expected_theta_det_nll)
    assert len(result["deterministic_bins"]) == 2
    assert len(result["mixture_bins"]) == 2
    assert result["summary"]["num_validation_thetas"] == 3
    assert result["summary"]["simulations_per_theta"] == 2

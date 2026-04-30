import pytest

np = pytest.importorskip("numpy", exc_type=ImportError)

from lfiax.utils.design_baselines import (
    select_baseline_design,
    validate_baseline_config,
)


@pytest.mark.parametrize(
    ("policy", "low", "high"),
    [
        ("random", 1e-6, 1e3),
        ("random", 0.01, 100.0),
        ("sobol", 1e-6, 1e3),
        ("sobol", 0.01, 100.0),
    ],
)
def test_baseline_designs_are_within_bounds(policy, low, high):
    design = select_baseline_design(policy, design_round=3, seed=123, low=low, high=high)

    assert design.shape == (1,)
    assert np.all(design >= low)
    assert np.all(design <= high)


def test_sobol_design_is_deterministic_for_seed_and_round():
    first = select_baseline_design("sobol", design_round=2, seed=17, low=0.01, high=100.0)
    second = select_baseline_design("sobol", design_round=2, seed=17, low=0.01, high=100.0)

    assert np.allclose(first, second)


def test_random_design_is_deterministic_for_seed_and_round():
    first = select_baseline_design("random", design_round=2, seed=17, low=1e-6, high=1e3)
    second = select_baseline_design("random", design_round=2, seed=17, low=1e-6, high=1e3)

    assert np.allclose(first, second)


def test_validate_baseline_config_rejects_unknown_values():
    with pytest.raises(ValueError, match="design_policy"):
        validate_baseline_config("bad", "nle")

    with pytest.raises(ValueError, match="likelihood_objective"):
        validate_baseline_config("random", "bad")

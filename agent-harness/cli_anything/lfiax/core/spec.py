"""ExperimentSpec loader for generic BOED runs.

Mirrors the shape used by boed_agent's ExperimentSpec so the same JSON file can
drive either the boed_agent backend adapter or this CLI directly.

See: boed_agent/src/boed_agent/models.py (ExperimentSpec)
     boed_agent/examples/specs/lfiax_stub.json
"""
from __future__ import annotations

import json
import os
from typing import Any


REQUIRED_FIELDS = [
    "backend",
    "simulator_ref",
    "design_variables",
    "objective.estimator",
]

# Estimators this CLI knows how to run in-process against lfiax.
SUPPORTED_ESTIMATORS = {"lf_pce_eig", "lf_pce_eig_scan"}

# How the design variable xi is parameterized.
#   "point"        -> single xi vector, clipped to bounds (Adam on xi directly).
#   "distribution" -> trainable (xi_mu, xi_stddev) of a truncated normal over
#                     the design bounds. Each training batch draws N designs
#                     (one per prior sample) via the reparameterization trick
#                     so gradients flow into (xi_mu, xi_stddev). This mirrors
#                     sir.py's use_design_dist=True.
VALID_DESIGN_MODES = ("point", "distribution")


def load_spec(path: str) -> dict[str, Any]:
    """Load an ExperimentSpec from a JSON file and normalize it."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Spec not found: {path}")
    with open(path) as f:
        raw = json.load(f)
    return normalize_spec(raw)


def normalize_spec(raw: dict[str, Any]) -> dict[str, Any]:
    """Fill in defaults and coerce known fields without losing unknown ones."""
    spec = dict(raw)
    spec.setdefault("backend", "lfiax")
    spec.setdefault("design_variables", [])
    spec.setdefault("observation_labels", [])
    spec.setdefault("target_latent_labels", [])
    spec.setdefault("compute_budget", {})
    spec.setdefault("objective", {})
    spec.setdefault("artifacts", {})
    spec.setdefault("backend_options", {})
    spec.setdefault("surrogate", {})
    spec.setdefault("initial_design", [])
    spec.setdefault("differentiable", None)

    # Normalize design_variables entries
    normed_vars = []
    for item in spec["design_variables"]:
        normed_vars.append(
            {
                "name": str(item["name"]),
                "lower": float(item["lower"]),
                "upper": float(item["upper"]),
                "initial": None if item.get("initial") is None else float(item["initial"]),
                "shape": int(item.get("shape", 1)),
                "description": item.get("description"),
            }
        )
    spec["design_variables"] = normed_vars

    obj = dict(spec["objective"])
    obj.setdefault("name", "expected_information_gain")
    obj.setdefault("estimator", None)
    obj.setdefault("estimator_kwargs", {})
    spec["objective"] = obj

    cb = dict(spec["compute_budget"])
    cb.setdefault("num_outer_samples", 256)  # N in lf_pce_eig_scan
    cb.setdefault("num_inner_samples", 10)   # M
    cb.setdefault("num_optimization_steps", 500)
    cb.setdefault("design_learning_rate", 0.05)
    cb.setdefault("flow_learning_rate", 1e-3)
    spec["compute_budget"] = cb

    # Design mode: default to point-estimate for backward compatibility.
    bo = dict(spec["backend_options"])
    bo.setdefault("design_mode", "point")
    # Initial values for distribution mode; fall back to reasonable defaults
    # derived from the design_variables bounds if not provided.
    if bo["design_mode"] == "distribution":
        bo.setdefault("xi_mu_init", None)       # list[float] or None -> midpoint
        bo.setdefault("xi_stddev_init", None)   # list[float] or None -> (upper-lower)/4
        bo.setdefault("xi_stddev_min", 1e-3)    # clamp to keep grad stable
        bo.setdefault("end_sigma", 0.01)        # annealed effective stddev floor
        bo.setdefault("decay_rate", 10.0)       # exponential annealing rate
    spec["backend_options"] = bo

    art = dict(spec["artifacts"])
    art.setdefault("output_dir", "artifacts")
    spec["artifacts"] = art

    return spec


def _get(spec: dict, dotted: str):
    cur: Any = spec
    for part in dotted.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
    return cur


def _missing(spec: dict, dotted: str) -> bool:
    v = _get(spec, dotted)
    if v is None:
        return True
    if isinstance(v, str):
        return v.strip() == ""
    if isinstance(v, (list, dict, tuple, set)):
        return len(v) == 0
    return False


def validate_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Validate an ExperimentSpec for the lfiax backend.

    Returns a dict shaped like boed_agent's ValidationReport:
    {valid, errors: [{path, message}], warnings: [...], missing_fields: [...], backend}
    """
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    missing: list[str] = []

    for path in REQUIRED_FIELDS:
        if _missing(spec, path):
            missing.append(path)

    # Need either prior_sampler_ref or latent_sampler_ref
    if not spec.get("prior_sampler_ref") and not spec.get("latent_sampler_ref"):
        missing.append("prior_sampler_ref_or_latent_sampler_ref")
        errors.append(
            {
                "path": "prior_sampler_ref_or_latent_sampler_ref",
                "message": "Provide either `prior_sampler_ref` or `latent_sampler_ref`.",
            }
        )

    for path in missing:
        if path == "prior_sampler_ref_or_latent_sampler_ref":
            continue
        errors.append({"path": path, "message": f"Missing required field `{path}`."})

    # Design variable bounds check
    for i, var in enumerate(spec.get("design_variables", [])):
        if var["lower"] >= var["upper"]:
            errors.append(
                {
                    "path": f"design_variables[{i}]",
                    "message": f"Design variable '{var['name']}' must have lower < upper.",
                }
            )

    # Estimator support check
    estimator = _get(spec, "objective.estimator")
    if estimator and estimator not in SUPPORTED_ESTIMATORS:
        warnings.append(
            {
                "path": "objective.estimator",
                "message": (
                    f"Estimator '{estimator}' is not in the v1 supported set "
                    f"{sorted(SUPPORTED_ESTIMATORS)}. It will be treated as 'lf_pce_eig_scan'."
                ),
            }
        )

    # Design mode check
    mode = (spec.get("backend_options") or {}).get("design_mode", "point")
    if mode not in VALID_DESIGN_MODES:
        errors.append(
            {
                "path": "backend_options.design_mode",
                "message": (
                    f"Unknown design_mode '{mode}'. "
                    f"Must be one of {list(VALID_DESIGN_MODES)}."
                ),
            }
        )

    if mode == "distribution":
        try:
            initial_design_distribution(spec)
        except ValueError as exc:
            errors.append(
                {
                    "path": "backend_options",
                    "message": str(exc),
                }
            )

    candidate_designs = (spec.get("backend_options") or {}).get("candidate_designs")
    if candidate_designs is not None:
        if not isinstance(candidate_designs, list) or not candidate_designs:
            errors.append(
                {
                    "path": "backend_options.candidate_designs",
                    "message": "`candidate_designs` must be a non-empty list of design vectors.",
                }
            )
        else:
            expected_dim = len(spec.get("design_variables", []))
            for index, candidate in enumerate(candidate_designs):
                if not isinstance(candidate, (list, tuple)):
                    errors.append(
                        {
                            "path": f"backend_options.candidate_designs[{index}]",
                            "message": "Each candidate design must be a list or tuple of numeric values.",
                        }
                    )
                    continue
                if expected_dim and len(candidate) != expected_dim:
                    errors.append(
                        {
                            "path": f"backend_options.candidate_designs[{index}]",
                            "message": (
                                f"Candidate design {index} has dimension {len(candidate)} but expected {expected_dim} values."
                            ),
                        }
                    )

    # Initial design dimensionality
    init = spec.get("initial_design") or []
    n_vars = len(spec.get("design_variables", []))
    if init and n_vars and len(init) != n_vars:
        errors.append(
            {
                "path": "initial_design",
                "message": (
                    f"initial_design has length {len(init)} but there are "
                    f"{n_vars} design_variables."
                ),
            }
        )

    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "missing_fields": sorted(set(missing)),
        "backend": spec.get("backend"),
    }


def effective_initial_design(spec: dict[str, Any]) -> list[float]:
    """Return the design vector to start optimization from."""
    if spec.get("initial_design"):
        return [float(v) for v in spec["initial_design"]]
    values: list[float] = []
    for var in spec.get("design_variables", []):
        if var.get("initial") is not None:
            values.append(float(var["initial"]))
        else:
            values.append((float(var["lower"]) + float(var["upper"])) / 2.0)
    return values


def design_bounds(spec: dict[str, Any]) -> tuple[list[float], list[float]]:
    lower = [float(v["lower"]) for v in spec.get("design_variables", [])]
    upper = [float(v["upper"]) for v in spec.get("design_variables", [])]
    return lower, upper


def design_mode(spec: dict[str, Any]) -> str:
    return (spec.get("backend_options") or {}).get("design_mode", "point")


def execution_path(spec: dict[str, Any]) -> str:
    return "differentiable" if spec.get("differentiable") is True else "black_box"


def initial_design_distribution(spec: dict[str, Any]) -> tuple[list[float], list[float]]:
    """Return (xi_mu_init, xi_stddev_init) for distribution-mode runs.

    Falls back to midpoint-of-bounds for mu and (upper-lower)/4 for stddev
    when not provided in backend_options.
    """
    bo = spec.get("backend_options") or {}
    lower, upper = design_bounds(spec)
    n = len(lower)

    mu = bo.get("xi_mu_init")
    if mu is None:
        mu_list = [(lo + hi) / 2.0 for lo, hi in zip(lower, upper)]
    else:
        mu_list = [float(v) for v in mu]
        if len(mu_list) != n:
            raise ValueError(
                f"xi_mu_init has length {len(mu_list)} but there are {n} design variables."
            )

    sd = bo.get("xi_stddev_init")
    if sd is None:
        sd_list = [max((hi - lo) / 4.0, 1e-3) for lo, hi in zip(lower, upper)]
    else:
        sd_list = [float(v) for v in sd]
        if len(sd_list) != n:
            raise ValueError(
                f"xi_stddev_init has length {len(sd_list)} but there are {n} design variables."
            )

    return mu_list, sd_list

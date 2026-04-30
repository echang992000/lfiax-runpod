"""Generic in-process BOED optimizer backed by lfiax.

This module supports two execution paths selected from the shared
``ExperimentSpec``:

- ``differentiable``: keep the existing gradient-through-simulator behavior.
- ``black_box``: simulate data outside autodiff and update the design only
  through the learned surrogate objective.

Both execution paths share the same training scaffold:

1. sample ``theta`` from the user prior
2. sample or broadcast ``xi`` from the current design parameterization
3. simulate ``y`` from the user simulator
4. jointly update the flow likelihood surrogate and the design parameters

Distribution-mode designs preserve LFIAX's annealed narrowing behavior by
shrinking the effective design stddev toward ``end_sigma`` over time.
"""

from __future__ import annotations

import json
import os
import pickle
import time
from typing import Any, Callable

from cli_anything.lfiax.core.problem import load_problem
from cli_anything.lfiax.core.spec import (
    design_bounds,
    design_mode as _design_mode,
    effective_initial_design,
    execution_path as _execution_path,
    initial_design_distribution,
    validate_spec,
)


def describe_backend() -> dict[str, Any]:
    """Return a BackendDescriptor-shaped dict."""
    return {
        "name": "lfiax",
        "description": (
            "In-process LF-PCE BOED optimizer using lfiax's NSF flow + "
            "lf_pce_eig_scan objective. Supports differentiable and black-box "
            "simulators, point designs, and annealed design distributions."
        ),
        "capabilities": {
            "supported_estimators": ["lf_pce_eig_scan", "lf_pce_eig"],
            "optimization_strategies": ["joint_gradient"],
            "design_modes": ["point", "distribution"],
            "execution_paths": ["differentiable", "black_box"],
            "supports_user_simulator_ref": True,
            "supports_user_prior_ref": True,
            "supports_black_box_simulators": True,
            "supports_differentiable_simulators": True,
            "exports_likelihood_checkpoint": True,
        },
        "required_fields": [
            "backend",
            "simulator_ref",
            "prior_sampler_ref_or_latent_sampler_ref",
            "design_variables",
            "objective.estimator",
        ],
        "optional_fields": [
            "differentiable",
            "backend_options.design_mode",
            "backend_options.xi_mu_init",
            "backend_options.xi_stddev_init",
            "backend_options.xi_stddev_min",
            "backend_options.end_sigma",
            "backend_options.decay_rate",
            "backend_options.flow.num_layers",
            "backend_options.flow.hidden_sizes",
            "backend_options.flow.num_bins",
        ],
        "status": "available",
    }


def plan_run(spec: dict[str, Any], seed: int = 0) -> dict[str, Any]:
    """Return the normalized plan without touching JAX/lfiax."""
    report = validate_spec(spec)
    lower, upper = design_bounds(spec)
    mode = _design_mode(spec)
    execution_path = _execution_path(spec)
    plan: dict[str, Any] = {
        "status": "dry_run",
        "backend": "lfiax",
        "estimator": _estimator_name(spec),
        "seed": seed,
        "design_mode": mode,
        "execution_path": execution_path,
        "design_bounds": {"lower": lower, "upper": upper},
        "compute_budget": spec.get("compute_budget", {}),
        "simulator_ref": spec.get("simulator_ref"),
        "prior_ref": spec.get("prior_sampler_ref") or spec.get("latent_sampler_ref"),
        "validation": report,
    }
    if mode == "distribution":
        try:
            mu, sd = initial_design_distribution(spec)
            plan["xi_mu_init"] = mu
            plan["xi_stddev_init"] = sd
            plan["annealing"] = {
                "end_sigma": _vector_setting(
                    spec.get("backend_options", {}).get("end_sigma", 0.01),
                    len(mu),
                    "end_sigma",
                ),
                "decay_rate": float(spec.get("backend_options", {}).get("decay_rate", 10.0)),
            }
        except ValueError as exc:
            plan["validation"]["errors"].append({"path": "backend_options", "message": str(exc)})
            plan["validation"]["valid"] = False
    else:
        plan["initial_design"] = effective_initial_design(spec)
    return plan


def optimize_design(
    spec: dict[str, Any],
    seed: int = 0,
    dry_run: bool = False,
    write_artifacts: bool = True,
) -> dict[str, Any]:
    """Run in-process BOED optimization and return an OptimizationResult dict."""
    report = validate_spec(spec)
    if not report["valid"]:
        return {
            "status": "invalid_spec",
            "backend": "lfiax",
            "validation": report,
        }

    if dry_run:
        return plan_run(spec, seed=seed)

    execution_path = _execution_path(spec)
    if execution_path == "differentiable":
        return optimize_design_differentiable(spec, seed=seed, write_artifacts=write_artifacts)
    return optimize_design_black_box(spec, seed=seed, write_artifacts=write_artifacts)


def optimize_design_differentiable(
    spec: dict[str, Any],
    seed: int = 0,
    write_artifacts: bool = True,
) -> dict[str, Any]:
    """Run the gradient-through-simulator BOED path."""
    try:
        runtime = _import_runtime()
        problem = load_problem(spec)
        state = _initialize_state(spec, seed, runtime, problem)
        return _run_joint_optimization(
            spec=spec,
            runtime=runtime,
            problem=problem,
            state=state,
            execution_path="differentiable",
            write_artifacts=write_artifacts,
            differentiable_simulator=True,
        )
    except Exception as exc:
        return {
            "status": "error",
            "backend": "lfiax",
            "execution_path": "differentiable",
            "error": str(exc),
        }


def optimize_design_black_box(
    spec: dict[str, Any],
    seed: int = 0,
    write_artifacts: bool = True,
) -> dict[str, Any]:
    """Run the black-box BOED path with simulator outputs detached from autodiff."""
    try:
        runtime = _import_runtime()
        problem = load_problem(spec)
        state = _initialize_state(spec, seed, runtime, problem)
        return _run_joint_optimization(
            spec=spec,
            runtime=runtime,
            problem=problem,
            state=state,
            execution_path="black_box",
            write_artifacts=write_artifacts,
            differentiable_simulator=False,
        )
    except Exception as exc:
        return {
            "status": "error",
            "backend": "lfiax",
            "execution_path": "black_box",
            "error": str(exc),
        }


def _run_joint_optimization(
    *,
    spec: dict[str, Any],
    runtime: dict[str, Any],
    problem: dict[str, Any],
    state: dict[str, Any],
    execution_path: str,
    write_artifacts: bool,
    differentiable_simulator: bool,
) -> dict[str, Any]:
    jax = runtime["jax"]
    jnp = runtime["jnp"]
    jrandom = runtime["jrandom"]
    optax = runtime["optax"]
    lf_pce_eig_scan = runtime["lf_pce_eig_scan"]

    simulator = problem["simulator"]
    prior = problem["prior"]
    mode = state["mode"]
    N = state["N"]
    M = state["M"]
    lam = state["lam"]
    xi_dim = state["xi_dim"]
    num_steps = state["num_steps"]
    lower_a = state["lower_a"]
    upper_a = state["upper_a"]
    log_prob_fn = state["log_prob_fn"]
    flow_params = state["flow_params"]
    flow_opt = optax.adam(state["flow_lr"])
    flow_opt_state = flow_opt.init(flow_params)
    design_opt = optax.adam(state["design_lr"])
    xi_state = state["xi_state"]
    design_opt_state = design_opt.init(xi_state)
    sigma_history: list[list[float]] = []
    history: list[dict[str, Any]] = []

    t0 = time.time()
    key = state["key"]

    def loss_from_fixed_batch(
        flow_tree: Any,
        xi_tree: dict[str, Any],
        *,
        theta: Any,
        y: Any,
        k_xi: Any,
        k_loss: Any,
        sigma_cap: Any,
    ) -> Any:
        xi_batch = _design_batch_from_state(
            xi_tree=xi_tree,
            mode=mode,
            key=k_xi,
            n=theta.shape[0],
            lower_a=lower_a,
            upper_a=upper_a,
            sigma_cap=sigma_cap,
            sd_min=state["sd_min"],
            runtime=runtime,
        )
        loss, _ = lf_pce_eig_scan(
            flow_tree, xi_batch, k_loss, y, theta, log_prob_fn.apply, N=N, M=M, lam=lam
        )
        return loss

    for step in range(num_steps):
        sigma_cap = _scheduled_sigma(
            step=step,
            num_steps=num_steps,
            start_sigma=state["start_sigma"],
            end_sigma=state["end_sigma"],
            mode=mode,
            runtime=runtime,
            decay_rate=state["decay_rate"],
        )
        key, k_prior, k_xi, k_sim, k_loss = jrandom.split(key, 5)
        theta = _sample_prior(prior, k_prior, N, runtime)
        xi_for_sim = _design_batch_from_state(
            xi_tree=xi_state,
            mode=mode,
            key=k_xi,
            n=N,
            lower_a=lower_a,
            upper_a=upper_a,
            sigma_cap=sigma_cap,
            sd_min=state["sd_min"],
            runtime=runtime,
        )
        if differentiable_simulator:
            grad_fn = jax.value_and_grad(
                lambda flow_tree, xi_tree: _loss_with_live_simulator(
                    flow_tree=flow_tree,
                    xi_tree=xi_tree,
                    theta=theta,
                    simulator=simulator,
                    k_xi=k_xi,
                    k_sim=k_sim,
                    k_loss=k_loss,
                    sigma_cap=sigma_cap,
                    mode=mode,
                    lower_a=lower_a,
                    upper_a=upper_a,
                    sd_min=state["sd_min"],
                    log_prob_apply=log_prob_fn.apply,
                    lf_pce_eig_scan=lf_pce_eig_scan,
                    runtime=runtime,
                    N=N,
                    M=M,
                    lam=lam,
                ),
                argnums=(0, 1),
            )
            loss, (g_flow, g_xi) = grad_fn(flow_params, xi_state)
            y = _call_simulator(simulator, theta, xi_for_sim, k_sim, runtime)
        else:
            y = _call_simulator(simulator, theta, xi_for_sim, k_sim, runtime)
            grad_fn = jax.value_and_grad(
                lambda flow_tree, xi_tree: loss_from_fixed_batch(
                    flow_tree,
                    xi_tree,
                    theta=theta,
                    y=y,
                    k_xi=k_xi,
                    k_loss=k_loss,
                    sigma_cap=sigma_cap,
                ),
                argnums=(0, 1),
            )
            loss, (g_flow, g_xi) = grad_fn(flow_params, xi_state)

        flow_updates, flow_opt_state = flow_opt.update(g_flow, flow_opt_state, flow_params)
        flow_params = optax.apply_updates(flow_params, flow_updates)

        design_updates, design_opt_state = design_opt.update(g_xi, design_opt_state, xi_state)
        xi_state = optax.apply_updates(xi_state, design_updates)
        xi_state = _clip_design_state(
            xi_state=xi_state,
            mode=mode,
            lower_a=lower_a,
            upper_a=upper_a,
            sd_min=state["sd_min"],
            runtime=runtime,
        )

        current_mu = _to_float_list(xi_state["xi_mu"])
        current_sigma = _effective_sigma_vector(
            xi_state=xi_state,
            mode=mode,
            sigma_cap=sigma_cap,
            sd_min=state["sd_min"],
            runtime=runtime,
        )
        sigma_history.append(_to_float_list(current_sigma))
        history_entry = {
            "step": step,
            "design": current_mu,
            "eig": float(-loss),
        }
        if mode == "distribution":
            history_entry["xi_mu"] = current_mu
            history_entry["xi_stddev"] = _to_float_list(current_sigma)
        history.append(history_entry)

    final_mu = _to_float_list(xi_state["xi_mu"])
    final_sigma = _to_float_list(
        _effective_sigma_vector(
            xi_state=xi_state,
            mode=mode,
            sigma_cap=_scheduled_sigma(
                step=max(num_steps - 1, 0),
                num_steps=num_steps,
                start_sigma=state["start_sigma"],
                end_sigma=state["end_sigma"],
                mode=mode,
                runtime=runtime,
                decay_rate=state["decay_rate"],
            ),
            sd_min=state["sd_min"],
            runtime=runtime,
        )
    )

    elapsed = time.time() - t0
    result: dict[str, Any] = {
        "status": "completed",
        "backend": "lfiax",
        "estimator": _estimator_name(spec),
        "design_mode": mode,
        "execution_path": execution_path,
        "design": final_mu,
        "eig": history[-1]["eig"] if history else None,
        "history": history,
        "warnings": [],
        "artifacts": {
            "elapsed_seconds": elapsed,
            "theta_dim": state["theta_dim"],
            "y_dim": state["y_dim"],
            "num_optimization_steps": num_steps,
            "compute_budget": spec.get("compute_budget", {}),
            "execution_path": execution_path,
            "sigma_history": sigma_history,
            "likelihood_metadata": {
                "theta_dim": state["theta_dim"],
                "y_dim": state["y_dim"],
                "design_dim": xi_dim,
                "design_mode": mode,
                "execution_path": execution_path,
                "flow_config": state["flow_cfg"],
            },
        },
    }
    result["xi_mu"] = final_mu
    if mode == "distribution":
        result["xi_stddev"] = final_sigma

    if write_artifacts:
        run_dir = _write_artifacts(
            spec=spec,
            result=result,
            flow_params=flow_params,
            t0=t0,
        )
        result["artifacts"]["run_dir"] = run_dir
        result["artifacts"]["likelihood_checkpoint"] = os.path.join(
            run_dir,
            spec.get("surrogate", {}).get("checkpoint_filename", "likelihood_checkpoint.pkl"),
        )

    return result


def _initialize_state(
    spec: dict[str, Any],
    seed: int,
    runtime: dict[str, Any],
    problem: dict[str, Any],
) -> dict[str, Any]:
    jnp = runtime["jnp"]
    jrandom = runtime["jrandom"]
    hk = runtime["hk"]
    make_nsf = runtime["make_nsf"]

    cb = spec["compute_budget"]
    lower, upper = design_bounds(spec)
    xi_dim = len(lower)
    mode = _design_mode(spec)
    key = jrandom.PRNGKey(seed)
    key, k_probe_p, k_probe_s = jrandom.split(key, 3)
    theta_probe = _sample_prior(problem["prior"], k_probe_p, 2, runtime)
    xi_probe = _initial_probe_design(spec, mode, xi_dim, runtime)
    y_probe = _call_simulator(problem["simulator"], theta_probe, xi_probe, k_probe_s, runtime)

    theta_dim = int(theta_probe.shape[-1])
    y_dim = int(y_probe.shape[-1])
    flow_cfg = spec.get("backend_options", {}).get("flow", {})
    num_layers = int(flow_cfg.get("num_layers", 4))
    hidden_sizes = list(flow_cfg.get("hidden_sizes", [64, 64]))
    num_bins = int(flow_cfg.get("num_bins", 8))

    @hk.without_apply_rng
    @hk.transform
    def log_prob_fn(data, context_theta, context_xi):
        model = make_nsf(
            event_shape=(y_dim,),
            num_layers=num_layers,
            hidden_sizes=hidden_sizes,
            num_bins=num_bins,
            conditional=True,
        )
        return model.log_prob(data, context_theta, context_xi)

    key, k_init = jrandom.split(key)
    flow_params = log_prob_fn.init(k_init, y_probe, theta_probe, xi_probe)

    if mode == "distribution":
        mu_init, sd_init = initial_design_distribution(spec)
        start_sigma = jnp.array(sd_init, dtype=jnp.float32)
        xi_state = {
            "xi_mu": jnp.array(mu_init, dtype=jnp.float32),
            "xi_stddev": jnp.array(sd_init, dtype=jnp.float32),
        }
        end_sigma = jnp.array(
            _vector_setting(
                spec.get("backend_options", {}).get("end_sigma", 0.01),
                xi_dim,
                "end_sigma",
            ),
            dtype=jnp.float32,
        )
    else:
        xi_state = {"xi_mu": jnp.array(effective_initial_design(spec), dtype=jnp.float32)}
        start_sigma = None
        end_sigma = None

    return {
        "N": int(cb["num_outer_samples"]),
        "M": int(cb["num_inner_samples"]),
        "num_steps": int(cb["num_optimization_steps"]),
        "design_lr": float(cb["design_learning_rate"]),
        "flow_lr": float(cb["flow_learning_rate"]),
        "lam": float((spec.get("objective", {}).get("estimator_kwargs") or {}).get("lam", 0.5)),
        "mode": mode,
        "xi_dim": xi_dim,
        "lower_a": jnp.array(lower, dtype=jnp.float32),
        "upper_a": jnp.array(upper, dtype=jnp.float32),
        "theta_dim": theta_dim,
        "y_dim": y_dim,
        "flow_cfg": {
            "num_layers": num_layers,
            "hidden_sizes": hidden_sizes,
            "num_bins": num_bins,
        },
        "log_prob_fn": log_prob_fn,
        "flow_params": flow_params,
        "xi_state": xi_state,
        "start_sigma": start_sigma,
        "end_sigma": end_sigma,
        "decay_rate": float(spec.get("backend_options", {}).get("decay_rate", 10.0)),
        "sd_min": float(spec.get("backend_options", {}).get("xi_stddev_min", 1e-3)),
        "key": key,
    }


def _loss_with_live_simulator(
    *,
    flow_tree: Any,
    xi_tree: dict[str, Any],
    theta: Any,
    simulator: Callable[..., Any],
    k_xi: Any,
    k_sim: Any,
    k_loss: Any,
    sigma_cap: Any,
    mode: str,
    lower_a: Any,
    upper_a: Any,
    sd_min: float,
    log_prob_apply: Callable[..., Any],
    lf_pce_eig_scan: Callable[..., Any],
    runtime: dict[str, Any],
    N: int,
    M: int,
    lam: float,
) -> Any:
    xi_batch = _design_batch_from_state(
        xi_tree=xi_tree,
        mode=mode,
        key=k_xi,
        n=theta.shape[0],
        lower_a=lower_a,
        upper_a=upper_a,
        sigma_cap=sigma_cap,
        sd_min=sd_min,
        runtime=runtime,
    )
    y = _call_simulator(simulator, theta, xi_batch, k_sim, runtime)
    loss, _ = lf_pce_eig_scan(flow_tree, xi_batch, k_loss, y, theta, log_prob_apply, N=N, M=M, lam=lam)
    return loss


def _design_batch_from_state(
    *,
    xi_tree: dict[str, Any],
    mode: str,
    key: Any,
    n: int,
    lower_a: Any,
    upper_a: Any,
    sigma_cap: Any,
    sd_min: float,
    runtime: dict[str, Any],
) -> Any:
    jnp = runtime["jnp"]
    jrandom = runtime["jrandom"]
    mu = xi_tree["xi_mu"]
    if mode == "point":
        return jnp.broadcast_to(mu, (n, mu.shape[-1]))

    effective_sd = _effective_sigma_vector(
        xi_state=xi_tree,
        mode=mode,
        sigma_cap=sigma_cap,
        sd_min=sd_min,
        runtime=runtime,
    )
    a = (lower_a - mu) / effective_sd
    b = (upper_a - mu) / effective_sd
    z = jrandom.truncated_normal(key, a, b, shape=(n, mu.shape[-1]))
    return mu + effective_sd * z


def _effective_sigma_vector(
    *,
    xi_state: dict[str, Any],
    mode: str,
    sigma_cap: Any,
    sd_min: float,
    runtime: dict[str, Any],
) -> Any:
    jnp = runtime["jnp"]
    if mode != "distribution":
        return jnp.zeros_like(xi_state["xi_mu"])
    raw_sd = jnp.maximum(xi_state["xi_stddev"], sd_min)
    if sigma_cap is None:
        return raw_sd
    return jnp.maximum(sd_min, jnp.minimum(raw_sd, sigma_cap))


def _scheduled_sigma(
    *,
    step: int,
    num_steps: int,
    start_sigma: Any,
    end_sigma: Any,
    mode: str,
    runtime: dict[str, Any],
    decay_rate: float,
) -> Any:
    if mode != "distribution" or start_sigma is None or end_sigma is None:
        return None
    jnp = runtime["jnp"]
    decay_constant = max(float(num_steps) / max(decay_rate, 1e-6), 1.0)
    return end_sigma + (start_sigma - end_sigma) * jnp.exp(-float(step) / decay_constant)


def _clip_design_state(
    *,
    xi_state: dict[str, Any],
    mode: str,
    lower_a: Any,
    upper_a: Any,
    sd_min: float,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    jnp = runtime["jnp"]
    clipped = {"xi_mu": jnp.clip(xi_state["xi_mu"], lower_a, upper_a)}
    if mode == "distribution":
        clipped["xi_stddev"] = jnp.maximum(xi_state["xi_stddev"], sd_min)
    return clipped


def _write_artifacts(
    *,
    spec: dict[str, Any],
    result: dict[str, Any],
    flow_params: Any,
    t0: float,
) -> str:
    out_dir = spec.get("artifacts", {}).get("output_dir", "artifacts")
    run_dir = os.path.join(out_dir, f"lfiax_oed_{int(t0)}")
    os.makedirs(run_dir, exist_ok=True)
    checkpoint_name = spec.get("surrogate", {}).get("checkpoint_filename", "likelihood_checkpoint.pkl")
    checkpoint_path = os.path.join(run_dir, checkpoint_name)
    with open(checkpoint_path, "wb") as handle:
        pickle.dump(
            {
                "flow_params": flow_params,
                "metadata": result["artifacts"]["likelihood_metadata"],
            },
            handle,
        )
    result["artifacts"]["likelihood_checkpoint"] = checkpoint_path
    metadata_path = os.path.join(run_dir, "likelihood_metadata.json")
    with open(metadata_path, "w") as handle:
        json.dump(_json_safe(result["artifacts"]["likelihood_metadata"]), handle, indent=2)
    result["artifacts"]["likelihood_metadata_json"] = metadata_path
    history_npz_path = os.path.join(run_dir, "optimization_history.npz")
    _write_history_npz(result, history_npz_path)
    result["artifacts"]["optimization_history_npz"] = history_npz_path
    history_summary = _history_summary(result.get("history", []))
    if history_summary is not None:
        result["history_summary"] = history_summary
        result["artifacts"]["history_summary"] = history_summary
    compact_result = _compact_result_payload(result)
    with open(os.path.join(run_dir, "result.json"), "w") as handle:
        json.dump(_json_safe(compact_result), handle, indent=2)
    with open(os.path.join(run_dir, "spec.json"), "w") as handle:
        json.dump(spec, handle, indent=2)
    return run_dir


def _import_runtime() -> dict[str, Any]:
    try:
        import haiku as hk
        import jax
        import jax.numpy as jnp
        import jax.random as jrandom
        import optax
    except ImportError as exc:
        raise RuntimeError(
            f"JAX/haiku/optax not available: {exc}. Install lfiax extras."
        ) from exc

    try:
        from lfiax.flows.nsf import make_nsf
        from lfiax.utils.oed_losses import lf_pce_eig_scan
    except ImportError as exc:
        raise RuntimeError(
            f"lfiax not importable: {exc}. Install lfiax (pip install -e .)."
        ) from exc

    return {
        "hk": hk,
        "jax": jax,
        "jnp": jnp,
        "jrandom": jrandom,
        "optax": optax,
        "make_nsf": make_nsf,
        "lf_pce_eig_scan": lf_pce_eig_scan,
    }


def _sample_prior(prior_fn: Callable[..., Any], key: Any, n: int, runtime: dict[str, Any]) -> Any:
    theta = prior_fn(key, n)
    if isinstance(theta, (tuple, list)):
        theta = theta[0]
    theta = _as_array(theta, runtime)
    if theta.ndim == 1:
        theta = theta[:, None]
    return theta


def _call_simulator(
    simulator: Callable[..., Any],
    theta: Any,
    xi: Any,
    key: Any,
    runtime: dict[str, Any],
) -> Any:
    y = simulator(theta, xi, key)
    if isinstance(y, (tuple, list)):
        y = y[0]
    y = _as_array(y, runtime)
    if y.ndim == 1:
        y = y[:, None]
    return y


def _initial_probe_design(spec: dict[str, Any], mode: str, xi_dim: int, runtime: dict[str, Any]) -> Any:
    jnp = runtime["jnp"]
    values = effective_initial_design(spec) if mode == "point" else initial_design_distribution(spec)[0]
    return jnp.broadcast_to(jnp.array(values, dtype=jnp.float32), (2, xi_dim))


def _estimator_name(spec: dict[str, Any]) -> str | None:
    estimator = (spec.get("objective") or {}).get("estimator")
    if estimator in {None, "lf_pce_eig", "lf_pce_eig_scan"}:
        return "lf_pce_eig_scan"
    return estimator


def _vector_setting(value: Any, expected_len: int, name: str) -> list[float]:
    if isinstance(value, (int, float)):
        return [float(value)] * expected_len
    if isinstance(value, (list, tuple)):
        values = [float(item) for item in value]
        if len(values) != expected_len:
            raise ValueError(
                f"{name} has length {len(values)} but there are {expected_len} design variables."
            )
        return values
    raise ValueError(f"{name} must be a number or list of numbers.")


def _to_float_list(value: Any) -> list[float]:
    try:
        return [float(item) for item in value]
    except TypeError:
        return [float(value)]


def _history_summary(history: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not history:
        return None
    eig_values = [entry["eig"] for entry in history if entry.get("eig") is not None]
    best_entry = max(
        (entry for entry in history if entry.get("eig") is not None),
        key=lambda entry: entry["eig"],
        default=history[-1],
    )
    first = history[0]
    last = history[-1]
    return {
        "num_steps": len(history),
        "start_design": first.get("design", []),
        "end_design": last.get("design", []),
        "best_design": best_entry.get("design", []),
        "best_step": best_entry.get("step"),
        "best_eig": best_entry.get("eig"),
        "eig_min": min(eig_values) if eig_values else None,
        "eig_max": max(eig_values) if eig_values else None,
        "checkpoints": _history_checkpoints(history),
    }


def _history_checkpoints(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not history:
        return []
    num_steps = len(history)
    indices = {0, num_steps - 1, num_steps // 2}
    if num_steps >= 4:
        indices.add(num_steps // 3)
        indices.add((2 * num_steps) // 3)
    return [history[index] for index in sorted(indices)]


def _compact_result_payload(result: dict[str, Any]) -> dict[str, Any]:
    compact = dict(result)
    compact.pop("history", None)
    if isinstance(compact.get("artifacts"), dict):
        compact["artifacts"] = dict(compact["artifacts"])
        compact["artifacts"].pop("sigma_history", None)
        compact["artifacts"].pop("history_summary", None)
    history = list(result.get("history", []))
    summary = result.get("history_summary") or _history_summary(history)
    if summary is not None:
        compact["history_summary"] = _compact_history_summary(summary)
    return compact


def _write_history_npz(result: dict[str, Any], output_path: str) -> None:
    try:
        import numpy as np
    except ImportError:
        return

    history = list(result.get("history", []))
    steps = np.asarray([int(entry.get("step", 0)) for entry in history], dtype=np.int64)
    designs = np.asarray([entry.get("design", []) for entry in history], dtype=np.float32)
    eig = np.asarray(
        [np.nan if entry.get("eig") is None else float(entry["eig"]) for entry in history],
        dtype=np.float32,
    )
    sigma_history = np.asarray(
        result.get("artifacts", {}).get("sigma_history", []),
        dtype=np.float32,
    )
    np.savez_compressed(
        output_path,
        steps=steps,
        designs=designs,
        eig=eig,
        sigma_history=sigma_history,
    )


def _compact_history_summary(summary: dict[str, Any]) -> dict[str, Any]:
    compact = dict(summary)
    compact.pop("checkpoints", None)
    return compact


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def _as_array(value: Any, runtime: dict[str, Any]) -> Any:
    return runtime["jnp"].asarray(value)

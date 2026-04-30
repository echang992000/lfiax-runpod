"""Resolve user-provided simulator and prior references.

Supports two forms:
    "package.module:attribute"  - standard import path
    "/abs/path/to/file.py:attr" - load module from a file path
"""
from __future__ import annotations

import importlib
import importlib.util
import os
from typing import Any, Callable


def resolve_reference(ref: str) -> Callable:
    """Resolve a 'module:attr' or 'file.py:attr' reference to a callable/object."""
    if not ref or ":" not in ref:
        raise ValueError(
            f"Invalid reference '{ref}'. Expected 'module.path:attr' or '/path/to/file.py:attr'."
        )
    module_part, _, attr = ref.rpartition(":")
    if not attr:
        raise ValueError(f"Reference '{ref}' is missing an attribute after ':'.")

    # File path form
    if module_part.endswith(".py") or os.path.sep in module_part or module_part.startswith("."):
        mod = _load_module_from_file(module_part)
    else:
        mod = importlib.import_module(module_part)

    if not hasattr(mod, attr):
        raise AttributeError(f"Module '{module_part}' has no attribute '{attr}'.")
    return getattr(mod, attr)


def _load_module_from_file(path: str) -> Any:
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Problem file not found: {abs_path}")
    module_name = f"_cli_anything_lfiax_problem_{abs(hash(abs_path))}"
    spec = importlib.util.spec_from_file_location(module_name, abs_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for {abs_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_problem(spec: dict) -> dict[str, Any]:
    """Resolve simulator + prior callables from an ExperimentSpec dict.

    Returns {"simulator": callable, "prior": callable, "metadata": {...}}.
    """
    sim_ref = spec.get("simulator_ref")
    if not sim_ref:
        raise ValueError("spec is missing `simulator_ref`.")
    simulator = resolve_reference(sim_ref)

    prior_ref = spec.get("prior_sampler_ref") or spec.get("latent_sampler_ref")
    if not prior_ref:
        raise ValueError("spec is missing `prior_sampler_ref` or `latent_sampler_ref`.")
    prior = resolve_reference(prior_ref)

    return {
        "simulator": simulator,
        "prior": prior,
        "simulator_ref": sim_ref,
        "prior_ref": prior_ref,
    }

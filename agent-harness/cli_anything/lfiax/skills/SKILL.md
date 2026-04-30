---
name: "cli-anything-lfiax"
description: "CLI harness for LFIAX — Likelihood Free Inference in JAX. Manages BOED experiments, configs, results, and simulators."
---

# cli-anything-lfiax

CLI harness for **LFIAX** (Likelihood Free Inference in JAX) — a JAX-based framework for Bayesian Optimal Experimental Design (BOED) and simulation-based inference.

## Prerequisites

- **lfiax** package installed (`pip install -e .` from lfiax repo root)
- **JAX** with appropriate backend (CPU/GPU/TPU)
- **Python** >= 3.9

## Installation

```bash
cd <lfiax-repo>/agent-harness
pip install -e .
```

## Command Syntax

```
cli-anything-lfiax [--json] [--project <path>] <command> [options]
```

### Global Flags

| Flag | Description |
|------|-------------|
| `--json` | Output machine-readable JSON (pipe-safe) |
| `--project <path>` | Path to project JSON file |

## Command Groups

### experiment — Run and manage BOED experiments

```bash
cli-anything-lfiax experiment run <type> [--config <yaml>] [-o key=val] [--seed N] [--device cpu|gpu] [--dry-run]
cli-anything-lfiax experiment list [--output-dir <dir>]
cli-anything-lfiax experiment status <workdir>
```

Experiment types: `bmp`, `sir`, `two_moons`, `two_moons_active_learning`

### config — Configuration management

```bash
cli-anything-lfiax config show <path>
cli-anything-lfiax config validate <path> [--type <experiment_type>]
cli-anything-lfiax config diff <config_a> <config_b>
```

### results — Inspect experiment results

```bash
cli-anything-lfiax results list [--output-dir <dir>]
cli-anything-lfiax results show <run_dir>
cli-anything-lfiax results checkpoint <checkpoint.pkl>
```

### simulator — Simulator information

```bash
cli-anything-lfiax simulator list
cli-anything-lfiax simulator info <name>
```

Simulators: `bmp` (Bone Morphogenetic Protein), `sir` (SIR epidemic), `linear_regression`, `two_moons`

### oed — Generic BOED optimizer (LF-PCE)

Runs in-process LF-PCE BOED against a **user-defined problem** (arbitrary
simulator + prior). Consumes an `ExperimentSpec` JSON compatible with the
`boed_agent` schema, so the same spec file can drive either the boed_agent
backend or this CLI directly.

```bash
cli-anything-lfiax oed describe
cli-anything-lfiax oed validate <spec.json>
cli-anything-lfiax oed optimize <spec.json> [--seed N] [--dry-run] [--no-artifacts]
cli-anything-lfiax oed init <out_dir>     # scaffold example problem.py + spec.json
```

**Spec shape** (minimal):
```json
{
  "backend": "lfiax",
  "simulator_ref": "my_pkg.problem:simulator",
  "prior_sampler_ref": "my_pkg.problem:prior",
  "design_variables": [{"name": "xi", "lower": -2.0, "upper": 2.0, "initial": 0.25}],
  "objective": {"estimator": "lf_pce_eig_scan", "estimator_kwargs": {"lam": 0.5}},
  "compute_budget": {
    "num_outer_samples": 256,
    "num_inner_samples": 10,
    "num_optimization_steps": 500,
    "design_learning_rate": 0.05,
    "flow_learning_rate": 0.001
  },
  "artifacts": {"output_dir": "artifacts"}
}
```

User module contract:
```python
def prior(key, n) -> theta          # shape (n, theta_dim), JAX arrays
def simulator(theta, xi, key) -> y  # shape (n, y_dim), JAX arrays
```

The optimizer builds an NSF flow sized from the probe outputs, then jointly
optimizes `(flow_params, xi)` under `lf_pce_eig_scan` via Adam, clipping xi to
the declared bounds. Result JSON is written to
`<output_dir>/lfiax_oed_<timestamp>/result.json`.

### env — Environment diagnostics

```bash
cli-anything-lfiax env
```

## Agent Usage Examples

### Check environment readiness
```bash
cli-anything-lfiax --json env
```
Returns: `{python: {version, executable}, lfiax: {installed, version}, jax: {available, version, devices}}`

### Discover available simulators
```bash
cli-anything-lfiax --json simulator list
```

### Run an experiment (dry run first)
```bash
cli-anything-lfiax --json experiment run bmp --dry-run -o experiment.design_rounds=3
cli-anything-lfiax experiment run bmp --config config_bmp.yaml --seed 42 --device gpu
```

### Inspect results
```bash
cli-anything-lfiax --json results list --output-dir ./outputs
cli-anything-lfiax --json results checkpoint ./outputs/run/design_round_0_flow_params_sbi_0.pkl
```

### Compare configurations
```bash
cli-anything-lfiax --json config diff config_bmp.yaml config_sir.yaml
```

## Error Handling

All errors in `--json` mode return: `{"error": "description"}`. Check the `error` key to detect failures.

## Interactive REPL

Run without arguments to enter interactive mode:
```bash
cli-anything-lfiax
```

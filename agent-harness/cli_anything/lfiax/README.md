# cli-anything-lfiax

CLI harness for **LFIAX** (Likelihood Free Inference in JAX) — a JAX-based framework for Bayesian Optimal Experimental Design (BOED) and simulation-based inference.

## Prerequisites

### 1. Install lfiax

```bash
cd /path/to/lfiax
pip install -e .
```

### 2. Install the CLI harness

```bash
cd /path/to/lfiax/agent-harness
pip install -e .
```

### 3. Verify installation

```bash
cli-anything-lfiax env
```

## Dependencies

- **lfiax** — the core library (JAX, Haiku, Distrax, Optax)
- **JAX** — differentiable computing backend
- **Hydra** — configuration management
- **wandb** — experiment tracking (optional but recommended)
- **click** — CLI framework
- **PyYAML** — config parsing

## Usage

### Interactive REPL

```bash
cli-anything-lfiax
```

Launches the interactive REPL with command completion and history.

### One-shot commands

```bash
# Run a BMP experiment
cli-anything-lfiax experiment run bmp --config config_bmp.yaml

# Run with overrides
cli-anything-lfiax experiment run sir -o experiment.design_rounds=3 -o optimization_params.learning_rate=0.001

# Dry run (print command without executing)
cli-anything-lfiax experiment run two_moons --dry-run

# List runs
cli-anything-lfiax experiment list --output-dir ./outputs

# Check run status
cli-anything-lfiax experiment status ./outputs/my_run

# Show a config
cli-anything-lfiax config show config_bmp.yaml

# Validate a config
cli-anything-lfiax config validate config_bmp.yaml --type bmp

# Compare configs
cli-anything-lfiax config diff config_bmp.yaml config_sir.yaml

# List results
cli-anything-lfiax results list

# Inspect a checkpoint
cli-anything-lfiax results checkpoint ./outputs/run/design_round_0_flow_params_sbi_0.pkl

# List simulators
cli-anything-lfiax simulator list

# Simulator details
cli-anything-lfiax simulator info bmp

# Environment info
cli-anything-lfiax env
```

### JSON output

All commands support `--json` for machine-readable output:

```bash
cli-anything-lfiax --json experiment list
cli-anything-lfiax --json simulator list
cli-anything-lfiax --json env
```

## Running tests

```bash
cd /path/to/lfiax/agent-harness
pip install -e .
python -m pytest cli_anything/lfiax/tests/ -v -s

# Force installed command testing
CLI_ANYTHING_FORCE_INSTALLED=1 python -m pytest cli_anything/lfiax/tests/ -v -s
```

## Command groups

| Group | Description |
|-------|-------------|
| `experiment` | Run, resume, list, and check status of BOED experiments |
| `config` | Show, validate, diff, and create experiment configurations |
| `results` | List runs, show metrics, inspect checkpoints |
| `simulator` | List and inspect available simulators |
| `env` | Show environment and dependency information |

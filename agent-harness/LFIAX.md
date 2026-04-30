# LFIAX CLI Harness — Software-Specific SOP

**Software:** LFIAX (Likelihood Free Inference in JAX)
**Harness:** cli-anything-lfiax
**Backend:** Python package (`lfiax`) using JAX, Haiku, Distrax
**Date:** 2026-04-03

---

## Phase 1 — Backend Analysis

### What the Software Is

LFIAX is a JAX-based library for Bayesian Optimal Experimental Design (BOED) and simulation-based inference (SBI). It trains normalizing flows (neural spline flows) to approximate posterior distributions and optimize experimental designs. There is no external GUI — the "real software" is the Python package and its experiment scripts.

### Backend Interface

| Module | Purpose |
|---|---|
| `src/lfiax/flows/nsf.py` | Neural Spline Flow architecture |
| `src/lfiax/bijectors/` | Bijector definitions for normalizing flows |
| `src/lfiax/utils/oed_losses.py` | BOED loss functions (NMC mutual information) |
| `src/lfiax/utils/sbi_losses.py` | SBI loss functions (forward/reverse KL) |
| `src/lfiax/utils/simulators.py` | Simulator definitions (BMP, SIR, linear regression) |
| `src/lfiax/utils/utils.py` | MCMC, transformations, calibration |

### Experiment Scripts

| Script | Config | Model |
|---|---|---|
| `BMP.py` | `config_bmp.yaml` | Bone Morphogenetic Protein ODE |
| `sir.py` | `config_sir.yaml` | SIR epidemic model |
| `two_moons.py` | `config_two_moons.yaml` | Two-moons benchmark |
| `two_moons_active_learning.py` | `config_two_moons_active_learning.yaml` | Active learning variant |

### Data Model

- **Configuration**: Hydra YAML files with sections: experiment, flow_params, optimization_params, designs, post_optimization, wandb
- **Checkpoints**: Pickle files containing flow_params, xi_params, post_params, post_samples (JAX arrays)
- **Logging**: Weights & Biases (wandb) for metrics, plus local CSV/JSON

### Existing CLI

No dedicated CLI. Users run scripts directly with Hydra overrides:
```bash
python BMP.py experiment.design_rounds=5 optimization_params.learning_rate=0.01
```

---

## Phase 2 — CLI Architecture

### Command Groups

```
cli-anything-lfiax experiment run <type> [--config <yaml>] [-o key=value] [--dry-run]
cli-anything-lfiax experiment list [--output-dir <dir>]
cli-anything-lfiax experiment status <workdir>

cli-anything-lfiax config show <path>
cli-anything-lfiax config validate <path> [--type <experiment_type>]
cli-anything-lfiax config diff <a> <b>

cli-anything-lfiax results list [--output-dir <dir>]
cli-anything-lfiax results show <run_dir>
cli-anything-lfiax results checkpoint <path>

cli-anything-lfiax simulator list
cli-anything-lfiax simulator info <name>

cli-anything-lfiax env
```

### Global Flags

| Flag | Purpose |
|---|---|
| `--json` | Machine-readable JSON output |
| `--project <path>` | Path to project JSON file |

### State Model

Session file at `~/.cli-anything-lfiax/session.json` tracks active project and command history. File locking via fcntl prevents concurrent corruption.

### Backend Strategy

The "real software" is lfiax itself. The CLI:
1. Invokes experiment scripts via subprocess (preserving Hydra compatibility)
2. Imports lfiax directly for lightweight ops (simulator listing, checkpoint inspection)
3. Scans filesystem for run outputs and checkpoints

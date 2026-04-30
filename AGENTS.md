# Repository Guide

Welcome! This repository contains experiments and reusable JAX + Haiku components for **Bayesian Optimal Experimental Design (BOED)** and simulation‑based inference.

> **Experiment = script + config + simulator**
> Each Python file in the repo root (e.g. `BMP.py`, `sir.py`) launches one complete BOED loop—choosing designs, simulating data, inferring posteriors, and logging everything to **Weights & Biases (wandb)**.

---

## Top‑level layout (abridged)

```
.
├── BMP.py, sir.py, ...        # Experiment entry‑points
├── config_*.yaml              # Default configs
├── outputs/                   # Generated runs & artefacts
├── src/                       # Installable package "lfiax"
│   └── lfiax/                 # Core library (flows, losses, utils)
└── tests/                     # PyTest suite
```

You’ll also find notebooks, figures, and archived results from various paper deadlines—kept for provenance.

---

## Running an experiment

```bash
# 1. Install the library
pip install -e .

# 2. Launch the BMP design loop
python BMP.py \
    --config config_bmp.yaml \
    --workdir ./outputs/bmp_run
```

The script will:

* start a **wandb** run (logs live in your browser),
* write checkpoints/CSVs under the directory you supplied with `--workdir`.

**Common CLI flags** (shared by every script):

```
--config   Path to YAML config file (defaults to matching config_*.yaml)
--workdir  Output directory (defaults to ./outputs/<timestamp>)
--seed     RNG seed
--device   cpu | gpu | tpu  (optional; falls back to JAX default)
```

---

## Configuration template

Each YAML follows the same minimal schema—override whatever you need:

```yaml
simulator:
  name: bmp                    # bmp | sir | linear_regression | ...
  num_observations: 10

design:
  rounds: 5                    # sequential design rounds
  num_candidates: 64           # candidate batch per round

model:
  flow: nsf                    # neural spline flow (see src/lfiax/flows/nsf.py)
  hidden_dims: [64, 64, 64]

loss:
  name: kl_mi_mix              # defined in utils/oed_losses.py
  beta: 0.01                   # KL/MI weighting

training:
  optimiser: adam
  lr: 1e-3
  steps_per_round: 2000
```

Omitted fields fall back to sensible defaults inside each script.

---

## Key modules in `src/lfiax`

| Path                   | What’s inside                                 |
| ---------------------- | --------------------------------------------- |
| `utils/oed_losses.py`  | Differentiable EIG, MI, and SBC estimators    |
| `flows/nsf.py`         | Neural‑Spline Flow implementation             |
| `minebed/`             | Mutual‑information neural estimator (MINEBed) |
| `utils/simulators.py`  | Generic simulator helpers                     |
| `utils/sir_utils.py`   | SIR‑specific routines                         |
| `nets/conditioners.py` | Conditional network blocks                    |

---

## Built‑in simulators

* **BMP** – ODE model of the **bone morphogenetic protein** signalling pathway (`bmp_simulator/`).
* **SIR** – Stochastic compartmental epidemic model.
* **Linear regression** – Closed‑form toy problem for smoke tests.

All are written in pure JAX; they run on CPU, GPU, or TPU without modification.

---

## Developing

### Adding a new experiment

1. Copy an existing script (e.g. `BMP.py`) or start from the minimal template in `examples/`.
2. Register your simulator in `utils/simulators.py` (or add a new helper module).
3. Create a YAML config with any special hyper‑parameters.
4. Add tests in `tests/` to keep CI green.
5. Open a PR against **`origin2/aistats_hotfixes`**.

### Branches & remotes

* **Local active branch**  : `aistats_hotfixes`
* **Primary upstream**     : `origin2/aistats_hotfixes`
* **Public mirror**        : `lfiax-public`

Feature branches should be based on `aistats_hotfixes` and include passing tests.

---

## FAQ

<details>
<summary>How do I resume an interrupted run?</summary>
The script scans `<workdir>` for the most recent checkpoint and continues automatically. Use `--resume false` to force a fresh start.
</details>

<details>
<summary>Can I disable wandb?</summary>
Set the environment variable `WANDB_MODE=dryrun` before launching an experiment.
</details>

<details>
<summary>Does it run on a cluster?</summary>
Yes. See `gpu_slurm.sh`, `cpu_two_moons.sh`, etc. for ready‑made SLURM templates. They forward all CLI flags unchanged.
</details>

Happy (Bayesian Optimal) experimenting! 🧪

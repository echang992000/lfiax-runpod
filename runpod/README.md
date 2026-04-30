# lfiax SIR baselines on RunPod

This folder packages everything needed to run `sir.py`'s baseline sweep
(the same one in `gpu_slurm.sh`) on a RunPod CUDA pod, plus the
`cli-anything-lfiax` agent harness so the boed_agent comparison runner
can also drive it from there.

## Why RunPod (not local)

The slurm script targets an A30 in the UCI cluster. Locally we hit two
walls:

1. **Mac GPU (jax-metal) is missing ops.** `jax-metal` 0.1.1 is the
   only published Metal backend and fails on `mhlo.cholesky`,
   `default_memory_space`, etc. — operations that distrax's
   `MultivariateNormalFullCovariance` and the NSF flow rely on. Even
   on a sandboxed `jax==0.4.34` install it dies inside
   `sample_lognormal_with_log_probs`.
2. **CPU is too slow.** At default config (`training_steps=10_000`,
   `design_rounds=2`) one invocation is ~5 hours on the M4 Max. 24
   invocations ≈ 120 hours.

A single A100/H100 does the same sweep in ~12 hours; an A6000/3090 in
~24 hours.

## What's in this folder

| File | Purpose |
|------|---------|
| `requirements.txt` | Pinned JAX (CUDA 12), distrax/haiku/optax, sbi, hydra, wandb, blackjax, torchSDE — same versions verified locally to import cleanly. |
| `Dockerfile` | Builds on `runpod/pytorch:2.1.0-py3.10-cuda12.1.1` (matches RunPod's standard PyTorch CUDA 12.1 template). Installs deps + lfiax + harness, generates the test pkl. |
| `setup.sh` | In-place setup if you start from RunPod's PyTorch template instead of building the image. |
| `gen_test_pkl.py` | Builds `test_prior_sde_numpy.pkl` (sir.py opens this from cwd; it's not in the repo). Idempotent. |
| `run_sir_baselines.sh` | Strips SLURM directives from `gpu_slurm.sh` and runs all 24 `python sir.py …` invocations sequentially with `device=cuda`, `wandb=false`, `hpc=false`. |

## Quick start (RunPod PyTorch template, no Docker)

1. **Spin up a pod.** Pick a CUDA 12.x PyTorch template
   (e.g. `runpod/pytorch:2.1.0-py3.10-cuda12.1.1-devel-ubuntu22.04`).
   Single GPU. Mount a 50–100 GB volume at `/workspace` so the
   `runpod_runs/` artifacts persist if the pod restarts.
2. **Push this repo to the pod.** Either:
   - `git clone <your fork>/lfiax-code.git` from inside the pod, or
   - `runpodctl send /Users/eric/Downloads/lfiax-code` from your laptop,
     then `runpodctl receive` on the pod.
3. **Install + verify.**
   ```bash
   cd lfiax-code
   bash runpod/setup.sh
   ```
   This pip-installs the requirements, the editable `lfiax` package,
   the editable `agent-harness` (so `cli-anything-lfiax` is on PATH),
   generates `test_prior_sde_numpy.pkl`, and prints the JAX devices.
   You should see `devices: [CudaDevice(id=0)]`.
4. **Run the sweep.**
   ```bash
   bash runpod/run_sir_baselines.sh
   ```
   Per-invocation logs go under `runpod_runs/<tag>/stdout.log`. Hydra
   writes its own per-run dir at `runpod_runs/<tag>/hydra/`.

   Override the budget with env vars for a cheap pass:
   ```bash
   LFIAX_TRAINING_STEPS=2000 LFIAX_DESIGN_ROUNDS=1 \
     bash runpod/run_sir_baselines.sh
   ```

## Quick start (build the Docker image)

```bash
cd /Users/eric/Downloads/lfiax-code
docker build -f runpod/Dockerfile -t lfiax-runpod:latest .
docker push <your registry>/lfiax-runpod:latest
```

Then on RunPod, choose "Deploy → Custom Container" and point at the
pushed image. The container's default `CMD` runs the full sweep.

## What you get back

After the sweep finishes:

- `runpod_runs/seed{1,2,3}_{random,sobol}_{nle,infonce_lambda_lam{0.01,0.1,1.0}}/`
  — one dir per invocation, with the full `sir.py` Hydra output,
  including learned flow params (if `experiment.save_params=True`),
  L-C2ST p-values, and the design EIG history.
- `stdout.log` per invocation for the per-step `STEP: …; Loss: …;
  EIG: …` log.

To pull results back to the laptop:
```bash
runpodctl send /workspace/lfiax-code/runpod_runs
# then on laptop:
runpodctl receive <code>
```

## Comparing against the local boed_agent harness runs

The harness comparison data lives in
`boed-agent-sir/artifacts/sir_seq_lfiax_harness_{asis,fixed}/`. After
the RunPod sweep finishes you can pull `runpod_runs/` back to the
laptop and run a 3-way comparison plotter (sir.py baselines + harness
as-is + harness mask-fix). That plotter isn't included here yet — let
me know if you want it scaffolded.

## Knobs you'll probably want

| Override | Default | Notes |
|----------|---------|-------|
| `LFIAX_TRAINING_STEPS` | 10000 | Per design round. Halve for ~50% time. |
| `LFIAX_DESIGN_ROUNDS` | 2 | sir.py's full sequential horizon. |
| `LFIAX_LOG_ROOT` | `runpod_runs/` | Where per-invocation dirs go. |
| `wandb.use_wandb=true entity=<…> project=<…>` (CLI override in `run_sir_baselines.sh`) | off | Re-enable WandB if you have credentials. |
| `experiment.save_params=true` | true (config default) | Saves flow checkpoints per round; large but useful for downstream evaluation. |

## Troubleshooting

- **`jaxlib not found` on import.** Wrong CUDA wheel. RunPod ships
  CUDA 12.1; if you're on a different image, edit `requirements.txt`
  to pin `jax[cuda12]` or `jax[cuda11_pip]` accordingly.
- **OOM during MCMC.** `mcmc_params.num_mcmc_samples` and
  `contrastive_sampling.N` are the main memory drivers. Halving N is
  cheap.
- **`No module named sbi`.** `setup.sh` should have installed it; rerun
  if you skipped step 3.
- **`test_prior_sde_numpy.pkl` missing on a fresh pod.** Re-run
  `python runpod/gen_test_pkl.py` from the repo root.

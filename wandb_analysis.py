# aggregate_wandb_sir.py
import math
import numpy as np
import pandas as pd
import wandb

ENTITY = "vz_uci"
PROJECT = "sir_UAI_rebuttal2"

# Put None to auto-detect numeric summary metrics.
# Or specify manually, e.g.:
# METRICS = ["boed_1/final_design_EIG", "boed_1/final_test_log_prob"]
METRICS = None

SEEDS = [1, 2, 3]
DESIGN_POLICIES = ["random", "sobol"]
OBJECTIVES = ["nle", "infonce_lambda"]

OUT_PREFIX = "sir_wandb"

api = wandb.Api()


def cfg_get(config, dotted_key, default=None):
    """Read either Hydra-style flat keys or nested W&B config dicts."""
    if dotted_key in config:
        return config[dotted_key]

    cur = config
    for part in dotted_key.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def fmt_lambda(x):
    if x is None:
        return None
    try:
        return f"{float(x):g}"
    except Exception:
        return str(x)


def condition_from_config(cfg):
    policy = cfg_get(cfg, "baseline.design_policy")
    objective = cfg_get(cfg, "baseline.likelihood_objective")
    seed = cfg_get(cfg, "seed")

    # Key point: ignore eig_lambda for nle, even if the config has a default value.
    if objective == "nle":
        eig_lambda = None
        condition = f"{policy} | nle"
    else:
        eig_lambda = fmt_lambda(cfg_get(cfg, "optimization_params.eig_lambda"))
        condition = f"{policy} | {objective} | lambda={eig_lambda}"

    return {
        "condition": condition,
        "design_policy": policy,
        "objective": objective,
        "eig_lambda": eig_lambda,
        "seed": seed,
    }


def is_number(x):
    return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool)


def clean_metric_name(k):
    bad_prefixes = ("_", "system/", "sys/", "runtime")
    bad_exact = {
        "epoch",
        "global_step",
        "trainer/global_step",
        "seed",
    }
    return not k.startswith(bad_prefixes) and k not in bad_exact


runs = api.runs(
    path=f"{ENTITY}/{PROJECT}",
    filters={
        "config.seed": {"$in": SEEDS},
        "config.baseline.design_policy": {"$in": DESIGN_POLICIES},
        "config.baseline.likelihood_objective": {"$in": OBJECTIVES},
    },
)

runs = list(runs)
print(f"Found {len(runs)} runs")

# Auto-detect metrics from run.summary.
if METRICS is None:
    metric_set = set()
    for run in runs:
        for k, v in dict(run.summary).items():
            if clean_metric_name(k) and is_number(v):
                metric_set.add(k)
    METRICS = sorted(metric_set)

print(f"Using {len(METRICS)} metrics:")
for m in METRICS:
    print("  ", m)

# -----------------------------
# 1. Final-value aggregation
# -----------------------------
raw_rows = []

for run in runs:
    meta = condition_from_config(run.config)
    row = {
        "run_id": run.id,
        "run_name": run.name,
        **meta,
    }

    for metric in METRICS:
        val = run.summary.get(metric, np.nan)
        row[metric] = val if is_number(val) else np.nan

    raw_rows.append(row)

raw = pd.DataFrame(raw_rows)

id_cols = [
    "condition",
    "design_policy",
    "objective",
    "eig_lambda",
    "seed",
    "run_id",
    "run_name",
]

raw_long = raw.melt(
    id_vars=id_cols,
    value_vars=METRICS,
    var_name="metric",
    value_name="value",
).dropna(subset=["value"])

agg = (
    raw_long
    .groupby(["condition", "design_policy", "objective", "eig_lambda", "metric"], dropna=False)
    ["value"]
    .agg(mean="mean", std="std", n="count")
    .reset_index()
)

agg["sem"] = agg["std"] / np.sqrt(agg["n"])

# Nice wide version: one row per condition, columns metric_mean / metric_sem.
wide_mean = agg.pivot(index="condition", columns="metric", values="mean")
wide_sem = agg.pivot(index="condition", columns="metric", values="sem")

wide_mean.columns = [f"{c}__mean" for c in wide_mean.columns]
wide_sem.columns = [f"{c}__sem" for c in wide_sem.columns]

wide = pd.concat([wide_mean, wide_sem], axis=1).reset_index()

raw.to_csv(f"{OUT_PREFIX}_raw_by_seed.csv", index=False)
agg.to_csv(f"{OUT_PREFIX}_final_long.csv", index=False)
wide.to_csv(f"{OUT_PREFIX}_final_wide.csv", index=False)

print()
print("Wrote:")
print(f"  {OUT_PREFIX}_raw_by_seed.csv")
print(f"  {OUT_PREFIX}_final_long.csv")
print(f"  {OUT_PREFIX}_final_wide.csv")
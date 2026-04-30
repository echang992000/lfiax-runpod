#!/usr/bin/env python3
"""
Make a compact summary table from sir_wandb_final_long.csv.

This expects the CSV produced by the earlier W&B aggregation script, with columns:
  design_policy, objective, eig_lambda, metric, mean, sem

It extracts these exact metrics:
  - boed_1/final_design_EIG
  - boed_1/LC2ST_statistic
  - boed_1/median_distance

and writes one row per condition:
  random / nle
  random / infonce_lambda, lambda=0.01
  random / infonce_lambda, lambda=0.1
  random / infonce_lambda, lambda=1
  sobol / nle
  sobol / infonce_lambda, lambda=0.01
  sobol / infonce_lambda, lambda=0.1
  sobol / infonce_lambda, lambda=1

Usage:
  python summarize_sir_wandb_selected.py

Optional:
  python summarize_sir_wandb_selected.py --input sir_wandb_final_long.csv --output sir_wandb_selected_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = {
    "EIG": "boed_1/final_design_EIG",
    "LC2ST": "boed_1/LC2ST_statistic",
    "Median distance": "boed_1/median_distance",
}

DESIGN_ORDER = ["random", "sobol"]
LAMBDA_ORDER = ["0.01", "0.1", "1"]

EXPECTED_COLUMNS = {
    "design_policy",
    "objective",
    "eig_lambda",
    "metric",
    "mean",
    "sem",
}


def normalize_lambda(x) -> str:
    """Normalize lambda values so 1.0 and 1 become the same label."""
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in {"", "nan", "none", "null"}:
        return ""
    try:
        return f"{float(s):g}"
    except ValueError:
        return s


def format_mean_pm_sem(mean, sem, digits: int = 4) -> str:
    """Return 'mean ± sem' using significant digits."""
    if pd.isna(mean):
        return ""
    if pd.isna(sem):
        return f"{mean:.{digits}g}"
    return f"{mean:.{digits}g} ± {sem:.{digits}g}"


def method_label(objective: str, eig_lambda: str) -> str:
    if objective == "nle":
        return "nle"
    if objective == "infonce_lambda":
        return f"infonce_lambda, lambda={eig_lambda}"
    return objective if not eig_lambda else f"{objective}, lambda={eig_lambda}"


def make_expected_rows() -> pd.DataFrame:
    rows = []

    for design in DESIGN_ORDER:
        rows.append({
            "design_policy": design,
            "objective": "nle",
            "eig_lambda_clean": "",
            "method": "nle",
        })

        for lam in LAMBDA_ORDER:
            rows.append({
                "design_policy": design,
                "objective": "infonce_lambda",
                "eig_lambda_clean": lam,
                "method": f"infonce_lambda, lambda={lam}",
            })

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="sir_wandb_final_long.csv")
    parser.add_argument("--output", default="sir_wandb_selected_summary.csv")
    parser.add_argument("--markdown", default="sir_wandb_selected_summary.md")
    parser.add_argument("--numeric-output", default="sir_wandb_selected_summary_numeric.csv")
    parser.add_argument("--digits", type=int, default=4)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Could not find input file: {input_path}")

    df = pd.read_csv(input_path)

    missing_cols = EXPECTED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Input is missing required columns: {sorted(missing_cols)}\n"
            f"Found columns: {list(df.columns)}"
        )

    wanted_metric_names = set(METRICS.values())
    found_metric_names = set(df["metric"].dropna().astype(str))
    missing_metrics = wanted_metric_names - found_metric_names

    if missing_metrics:
        print("Could not find these requested metric names:")
        for m in sorted(missing_metrics):
            print(f"  {m}")
        print("\nAvailable metrics containing EIG, LC2ST, median, or distance:")
        keywords = ("eig", "lc2st", "median", "distance")
        for m in sorted(found_metric_names):
            if any(k in m.lower() for k in keywords):
                print(f"  {m}")
        raise SystemExit("\nPlease check the metric names in the input CSV.")

    df = df[df["metric"].isin(wanted_metric_names)].copy()

    # Normalize condition fields.
    df["design_policy"] = df["design_policy"].astype(str)
    df["objective"] = df["objective"].astype(str)
    df["eig_lambda_clean"] = df["eig_lambda"].apply(normalize_lambda)

    # Important: lambda is irrelevant for nle.
    df.loc[df["objective"] == "nle", "eig_lambda_clean"] = ""

    # Rename metrics to readable labels.
    reverse_metrics = {v: k for k, v in METRICS.items()}
    df["metric_clean"] = df["metric"].map(reverse_metrics)

    # Keep only the intended conditions.
    expected = make_expected_rows()

    df = df.merge(
        expected[["design_policy", "objective", "eig_lambda_clean"]],
        on=["design_policy", "objective", "eig_lambda_clean"],
        how="inner",
    )

    # If duplicate rows exist for the same condition and metric, keep the first and warn.
    duplicate_mask = df.duplicated(
        ["design_policy", "objective", "eig_lambda_clean", "metric_clean"],
        keep=False,
    )
    if duplicate_mask.any():
        print("Warning: found duplicate rows for some condition/metric pairs. Keeping the first.")
        print(
            df.loc[
                duplicate_mask,
                ["design_policy", "objective", "eig_lambda_clean", "metric_clean", "mean", "sem"],
            ].sort_values(["design_policy", "objective", "eig_lambda_clean", "metric_clean"])
        )
        df = df.drop_duplicates(
            ["design_policy", "objective", "eig_lambda_clean", "metric_clean"],
            keep="first",
        )

    # Numeric wide table, useful for further plotting or LaTeX later.
    numeric_mean = df.pivot_table(
        index=["design_policy", "objective", "eig_lambda_clean"],
        columns="metric_clean",
        values="mean",
        aggfunc="first",
    )
    numeric_sem = df.pivot_table(
        index=["design_policy", "objective", "eig_lambda_clean"],
        columns="metric_clean",
        values="sem",
        aggfunc="first",
    )

    numeric = pd.concat(
        [
            numeric_mean.add_suffix(" mean"),
            numeric_sem.add_suffix(" sem"),
        ],
        axis=1,
    ).reset_index()

    numeric = expected.merge(
        numeric,
        on=["design_policy", "objective", "eig_lambda_clean"],
        how="left",
    )

    # Formatted human-readable table.
    rows = []
    for _, cond in expected.iterrows():
        row = {
            "design_policy": cond["design_policy"],
            "method": cond["method"],
        }

        subset = df[
            (df["design_policy"] == cond["design_policy"])
            & (df["objective"] == cond["objective"])
            & (df["eig_lambda_clean"] == cond["eig_lambda_clean"])
        ]

        for clean_name in METRICS.keys():
            match = subset[subset["metric_clean"] == clean_name]
            if match.empty:
                row[clean_name] = ""
            else:
                r = match.iloc[0]
                row[clean_name] = format_mean_pm_sem(r["mean"], r["sem"], args.digits)

        rows.append(row)

    summary = pd.DataFrame(rows)

    output_path = Path(args.output)
    markdown_path = Path(args.markdown)
    numeric_output_path = Path(args.numeric_output)

    summary.to_csv(output_path, index=False)
    numeric.to_csv(numeric_output_path, index=False)

    markdown = summary.to_markdown(index=False)
    markdown_path.write_text(markdown + "\n")

    print("\nCompact summary:")
    print(markdown)

    # Let the user know if any expected condition/metric cells are empty.
    empty_cells = []
    for _, row in summary.iterrows():
        for metric_label in METRICS.keys():
            if row[metric_label] == "":
                empty_cells.append((row["design_policy"], row["method"], metric_label))

    if empty_cells:
        print("\nWarning: these expected cells were not found in the input:")
        for design, method, metric_label in empty_cells:
            print(f"  {design} | {method} | {metric_label}")

    print("\nWrote:")
    print(f"  {output_path}")
    print(f"  {markdown_path}")
    print(f"  {numeric_output_path}")


if __name__ == "__main__":
    main()

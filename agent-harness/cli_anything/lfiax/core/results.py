"""Results inspection — find runs, load checkpoints, compare metrics."""
import os
import glob
import pickle
from datetime import datetime, timezone


def find_runs(base_dir="./outputs"):
    """Scan for completed experiment runs.

    Args:
        base_dir: Base output directory.

    Returns:
        list[dict]: List of run summaries.
    """
    if not os.path.isdir(base_dir):
        return []

    runs = []
    for entry in sorted(os.listdir(base_dir)):
        run_dir = os.path.join(base_dir, entry)
        if not os.path.isdir(run_dir):
            continue

        checkpoints = glob.glob(os.path.join(run_dir, "**/*.pkl"), recursive=True)
        wandb_dirs = glob.glob(os.path.join(run_dir, "**/wandb"), recursive=True)

        status = "completed" if checkpoints else "empty"
        exp_type = _detect_experiment_type(run_dir)

        runs.append({
            "name": entry,
            "path": run_dir,
            "experiment_type": exp_type,
            "status": status,
            "num_checkpoints": len(checkpoints),
            "has_wandb": len(wandb_dirs) > 0,
        })

    return runs


def _detect_experiment_type(run_dir):
    """Attempt to detect experiment type from run directory contents."""
    for f in os.listdir(run_dir):
        fl = f.lower()
        if "bmp" in fl:
            return "bmp"
        if "sir" in fl:
            return "sir"
        if "two_moons" in fl or "twomoons" in fl:
            return "two_moons"
    # Check for hydra config
    hydra_dir = os.path.join(run_dir, ".hydra")
    if os.path.isdir(hydra_dir):
        config_file = os.path.join(hydra_dir, "config.yaml")
        if os.path.isfile(config_file):
            try:
                import yaml
                with open(config_file) as fh:
                    cfg = yaml.safe_load(fh) or {}
                sim = cfg.get("simulator", {})
                if isinstance(sim, dict):
                    return sim.get("name", "unknown")
            except Exception:
                pass
    return "unknown"


def load_checkpoint(checkpoint_path):
    """Load a pickle checkpoint and return summary.

    Args:
        checkpoint_path: Path to .pkl file.

    Returns:
        dict: {path, size_bytes, contents: {key: info}}
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    size = os.path.getsize(checkpoint_path)

    with open(checkpoint_path, "rb") as f:
        data = pickle.load(f)

    contents = {}
    if isinstance(data, dict):
        for key, value in data.items():
            info = {"type": type(value).__name__}
            if hasattr(value, "shape"):
                info["shape"] = list(value.shape)
                info["dtype"] = str(value.dtype)
            elif isinstance(value, (list, tuple)):
                info["length"] = len(value)
            elif isinstance(value, dict):
                info["keys"] = list(value.keys())[:10]
            contents[key] = info
    else:
        contents["_root"] = {"type": type(data).__name__}

    return {
        "path": checkpoint_path,
        "size_bytes": size,
        "num_keys": len(contents),
        "contents": contents,
    }


def get_run_summary(run_dir):
    """Return a structured summary of a run.

    Args:
        run_dir: Path to run directory.

    Returns:
        dict: Summary info.
    """
    if not os.path.isdir(run_dir):
        return {"error": f"Directory not found: {run_dir}"}

    checkpoints = sorted(glob.glob(os.path.join(run_dir, "**/*.pkl"), recursive=True))
    exp_type = _detect_experiment_type(run_dir)

    # Count design rounds from checkpoint names
    design_rounds = set()
    for cp in checkpoints:
        name = os.path.basename(cp)
        if "design_round_" in name:
            try:
                round_num = int(name.split("design_round_")[1].split("_")[0])
                design_rounds.add(round_num)
            except (ValueError, IndexError):
                pass

    stat = os.stat(run_dir)

    return {
        "run_dir": run_dir,
        "experiment_type": exp_type,
        "num_checkpoints": len(checkpoints),
        "design_rounds_completed": len(design_rounds),
        "checkpoint_files": [os.path.basename(c) for c in checkpoints],
        "created": datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc).isoformat(),
        "modified": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    }


def compare_runs(run_dirs):
    """Compare summaries across multiple runs.

    Args:
        run_dirs: List of run directory paths.

    Returns:
        dict: Comparison data.
    """
    summaries = []
    for rd in run_dirs:
        summaries.append(get_run_summary(rd))

    return {
        "num_runs": len(summaries),
        "runs": summaries,
    }

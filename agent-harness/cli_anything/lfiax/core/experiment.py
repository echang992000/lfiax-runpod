"""Experiment management — run, resume, status, list experiments."""
import os
import sys
import subprocess
import glob
from datetime import datetime, timezone

EXPERIMENT_SCRIPTS = {
    "bmp": "BMP.py",
    "sir": "sir.py",
    "two_moons": "two_moons.py",
    "two_moons_active_learning": "two_moons_active_learning.py",
}

EXPERIMENT_CONFIGS = {
    "bmp": "config_bmp.yaml",
    "sir": "config_sir.yaml",
    "two_moons": "config_two_moons.yaml",
    "two_moons_active_learning": "config_two_moons_active_learning.yaml",
}


def find_lfiax_root():
    """Find the lfiax repository root directory.

    Checks LFIAX_ROOT env var first, then walks up from cwd.

    Returns:
        str: Absolute path to lfiax root.

    Raises:
        RuntimeError: If root cannot be found.
    """
    root = os.environ.get("LFIAX_ROOT")
    if root and os.path.isdir(os.path.join(root, "src", "lfiax")):
        return os.path.abspath(root)

    current = os.path.abspath(os.getcwd())
    for _ in range(10):
        if os.path.isdir(os.path.join(current, "src", "lfiax")):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent

    raise RuntimeError(
        "Cannot find lfiax repository root. Set LFIAX_ROOT environment variable "
        "or run from within the lfiax directory."
    )


def run_experiment(experiment_type, config_path=None, workdir=None,
                   seed=None, device=None, overrides=None, dry_run=False):
    """Run an experiment by invoking its script.

    Args:
        experiment_type: One of bmp, sir, two_moons, two_moons_active_learning.
        config_path: Custom config YAML path.
        workdir: Output directory.
        seed: RNG seed.
        device: Compute device (cpu/gpu/tpu).
        overrides: List of Hydra override strings.
        dry_run: If True, return command without executing.

    Returns:
        dict: {status, command, workdir, returncode, stdout, stderr}
    """
    if experiment_type not in EXPERIMENT_SCRIPTS:
        return {
            "status": "error",
            "error": f"Unknown experiment type: {experiment_type}",
            "valid_types": list(EXPERIMENT_SCRIPTS.keys()),
        }

    try:
        root = find_lfiax_root()
    except RuntimeError as e:
        return {"status": "error", "error": str(e)}

    script = os.path.join(root, EXPERIMENT_SCRIPTS[experiment_type])
    if not os.path.isfile(script):
        return {"status": "error", "error": f"Script not found: {script}"}

    cmd = [sys.executable, script]

    # Config handling
    if config_path:
        config_dir = os.path.dirname(os.path.abspath(config_path))
        config_name = os.path.splitext(os.path.basename(config_path))[0]
        cmd.extend(["--config-dir", config_dir, "--config-name", config_name])

    # Hydra overrides
    hydra_overrides = list(overrides) if overrides else []
    if seed is not None:
        hydra_overrides.append(f"experiment.seed={seed}")
    if device:
        hydra_overrides.append(f"experiment.device={device}")
    if workdir:
        hydra_overrides.append(f"experiment.workdir={workdir}")
    cmd.extend(hydra_overrides)

    result = {
        "experiment_type": experiment_type,
        "command": cmd,
        "workdir": workdir or os.path.join(root, "outputs"),
        "script": script,
    }

    if dry_run:
        result["status"] = "dry_run"
        result["command_str"] = " ".join(cmd)
        return result

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, cwd=root
        )
        result["status"] = "completed" if proc.returncode == 0 else "failed"
        result["returncode"] = proc.returncode
        result["stdout"] = proc.stdout[-2000:] if proc.stdout else ""
        result["stderr"] = proc.stderr[-2000:] if proc.stderr else ""
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)

    return result


def get_run_status(workdir):
    """Check status of an experiment run directory.

    Args:
        workdir: Path to the run output directory.

    Returns:
        dict: Status information.
    """
    if not os.path.isdir(workdir):
        return {"status": "not_found", "workdir": workdir}

    checkpoints = glob.glob(os.path.join(workdir, "**/*.pkl"), recursive=True)
    wandb_dirs = glob.glob(os.path.join(workdir, "**/wandb"), recursive=True)
    has_latest = os.path.isfile(os.path.join(workdir, "latest.pt"))

    status = "unknown"
    if has_latest:
        status = "resumable"
    elif checkpoints:
        status = "completed"
    else:
        status = "empty"

    return {
        "workdir": workdir,
        "status": status,
        "num_checkpoints": len(checkpoints),
        "checkpoints": [os.path.basename(c) for c in sorted(checkpoints)],
        "has_wandb": len(wandb_dirs) > 0,
        "has_latest_pt": has_latest,
    }


def list_runs(output_base="./outputs"):
    """List all experiment runs in the output directory.

    Args:
        output_base: Base output directory to scan.

    Returns:
        list[dict]: List of run info dicts.
    """
    if not os.path.isdir(output_base):
        return []

    runs = []
    for entry in sorted(os.listdir(output_base)):
        run_dir = os.path.join(output_base, entry)
        if not os.path.isdir(run_dir):
            continue

        # Detect experiment type from checkpoints or config
        exp_type = "unknown"
        for script_type, script_name in EXPERIMENT_SCRIPTS.items():
            base_name = script_name.replace(".py", "").lower()
            if base_name in entry.lower():
                exp_type = script_type
                break

        checkpoints = glob.glob(os.path.join(run_dir, "**/*.pkl"), recursive=True)
        stat = os.stat(run_dir)

        runs.append({
            "name": entry,
            "type": exp_type,
            "status": "completed" if checkpoints else "empty",
            "created": datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc).isoformat(),
            "num_checkpoints": len(checkpoints),
            "path": run_dir,
        })

    return runs

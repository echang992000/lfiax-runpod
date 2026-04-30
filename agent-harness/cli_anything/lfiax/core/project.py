"""Project management — create, load, save experiment project state."""
import json
import os
from datetime import datetime, timezone

VALID_EXPERIMENT_TYPES = ("bmp", "sir", "two_moons", "two_moons_active_learning")


def create_project(name, experiment_type, config_path=None, output_dir=None):
    """Create a new project dict.

    Args:
        name: Project name.
        experiment_type: One of bmp, sir, two_moons, two_moons_active_learning.
        config_path: Path to the config YAML file.
        output_dir: Base output directory for runs.

    Returns:
        dict: Project state.

    Raises:
        ValueError: If experiment_type is not recognized.
    """
    if experiment_type not in VALID_EXPERIMENT_TYPES:
        raise ValueError(
            f"Unknown experiment type '{experiment_type}'. "
            f"Valid types: {', '.join(VALID_EXPERIMENT_TYPES)}"
        )
    return {
        "name": name,
        "experiment_type": experiment_type,
        "config_path": config_path,
        "output_dir": output_dir or "./outputs",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "created",
        "runs": [],
    }


def save_project(project, path):
    """Save project dict to a JSON file.

    Args:
        project: Project dict.
        path: File path to write.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(project, f, indent=2, default=str)


def load_project(path):
    """Load project from a JSON file.

    Args:
        path: Path to project JSON.

    Returns:
        dict: Project state.

    Raises:
        FileNotFoundError: If file doesn't exist.
        json.JSONDecodeError: If file is not valid JSON.
    """
    with open(path) as f:
        return json.load(f)


def get_project_info(project):
    """Return a summary dict of project state.

    Args:
        project: Project dict.

    Returns:
        dict: Summary info.
    """
    return {
        "name": project.get("name", "unknown"),
        "experiment_type": project.get("experiment_type", "unknown"),
        "config_path": project.get("config_path"),
        "output_dir": project.get("output_dir"),
        "status": project.get("status", "unknown"),
        "created_at": project.get("created_at"),
        "num_runs": len(project.get("runs", [])),
    }


def add_run(project, run_id, config_overrides=None):
    """Add a run entry to the project.

    Args:
        project: Project dict (modified in place).
        run_id: Unique run identifier.
        config_overrides: Optional dict of config overrides for this run.

    Returns:
        dict: The new run entry.
    """
    run = {
        "run_id": run_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "pending",
        "config_overrides": config_overrides or {},
    }
    project.setdefault("runs", []).append(run)
    project["status"] = "active"
    return run

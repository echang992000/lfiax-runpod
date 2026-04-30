"""Configuration management — load, validate, diff, create configs."""
import os
import yaml


# Required top-level sections per experiment type
CONFIG_SCHEMA = {
    "bmp": {
        "required": ["experiment", "flow_params", "optimization_params"],
        "optional": ["designs", "post_optimization", "wandb"],
    },
    "sir": {
        "required": ["experiment", "flow_params", "optimization_params"],
        "optional": ["designs", "post_optimization", "wandb"],
    },
    "two_moons": {
        "required": ["flow_params", "optimization_params"],
        "optional": ["experiment", "wandb"],
    },
    "two_moons_active_learning": {
        "required": ["flow_params", "optimization_params"],
        "optional": ["experiment", "wandb"],
    },
}

# Default config templates
_DEFAULTS = {
    "bmp": {
        "experiment": {
            "design_rounds": 5,
            "refine_rounds": 1,
            "sbi_prior_samples": 1000,
            "device": "cpu",
        },
        "flow_params": {
            "num_layers": 5,
            "mlp_num_layers": 2,
            "hidden_size": 64,
            "num_bins": 8,
            "activation": "gelu",
            "dropout_rate": 0.0,
        },
        "optimization_params": {
            "learning_rate": 1e-3,
            "training_steps": 2000,
        },
    },
    "sir": {
        "experiment": {
            "design_rounds": 5,
            "refine_rounds": 1,
            "device": "cpu",
        },
        "flow_params": {
            "num_layers": 5,
            "mlp_num_layers": 2,
            "hidden_size": 64,
            "num_bins": 8,
            "activation": "gelu",
        },
        "optimization_params": {
            "learning_rate": 1e-3,
            "training_steps": 2000,
        },
    },
    "two_moons": {
        "flow_params": {
            "num_layers": 5,
            "mlp_num_layers": 2,
            "hidden_size": 64,
            "num_bins": 8,
        },
        "optimization_params": {
            "learning_rate": 1e-3,
            "training_steps": 5000,
        },
    },
    "two_moons_active_learning": {
        "flow_params": {
            "num_layers": 5,
            "mlp_num_layers": 2,
            "hidden_size": 64,
            "num_bins": 8,
        },
        "optimization_params": {
            "learning_rate": 1e-3,
            "training_steps": 5000,
        },
    },
}


def load_config(config_path):
    """Load a YAML config file.

    Args:
        config_path: Path to config YAML.

    Returns:
        dict: Parsed config.

    Raises:
        FileNotFoundError: If file doesn't exist.
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def show_config(config_path):
    """Load and return config as dict.

    Args:
        config_path: Path to config YAML.

    Returns:
        dict: Parsed config.
    """
    return load_config(config_path)


def validate_config(config, experiment_type=None):
    """Validate a config dict.

    Args:
        config: Parsed config dict.
        experiment_type: Optional experiment type for type-specific validation.

    Returns:
        dict: {valid: bool, errors: list, warnings: list}
    """
    errors = []
    warnings = []

    if not isinstance(config, dict):
        return {"valid": False, "errors": ["Config is not a dict"], "warnings": []}

    if not config:
        errors.append("Config is empty")

    if experiment_type and experiment_type in CONFIG_SCHEMA:
        schema = CONFIG_SCHEMA[experiment_type]
        for section in schema["required"]:
            if section not in config:
                errors.append(f"Missing required section: '{section}'")
        for section in schema.get("optional", []):
            if section not in config:
                warnings.append(f"Optional section not present: '{section}'")

    # Check for common issues
    flow = config.get("flow_params", {})
    if isinstance(flow, dict):
        if flow.get("num_layers", 1) > 20:
            warnings.append("flow_params.num_layers > 20 is unusually high")
        if flow.get("hidden_size", 64) > 1024:
            warnings.append("flow_params.hidden_size > 1024 is unusually large")

    opt = config.get("optimization_params", {})
    if isinstance(opt, dict):
        lr = opt.get("learning_rate", 1e-3)
        if isinstance(lr, (int, float)):
            if lr > 0.1:
                warnings.append("optimization_params.learning_rate > 0.1 is unusually high")
            if lr < 1e-7:
                warnings.append("optimization_params.learning_rate < 1e-7 is unusually low")

    return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}


def diff_configs(config_a_path, config_b_path):
    """Compare two config files.

    Args:
        config_a_path: Path to first config.
        config_b_path: Path to second config.

    Returns:
        dict: {differences: {key: {a: val, b: val}}, a_only: [...], b_only: [...]}
    """
    a = load_config(config_a_path)
    b = load_config(config_b_path)

    differences = {}
    a_only = []
    b_only = []

    def _flatten(d, prefix=""):
        items = {}
        if isinstance(d, dict):
            for k, v in d.items():
                key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    items.update(_flatten(v, key))
                else:
                    items[key] = v
        return items

    flat_a = _flatten(a)
    flat_b = _flatten(b)

    all_keys = set(flat_a.keys()) | set(flat_b.keys())
    for key in sorted(all_keys):
        if key in flat_a and key not in flat_b:
            a_only.append(key)
        elif key in flat_b and key not in flat_a:
            b_only.append(key)
        elif flat_a.get(key) != flat_b.get(key):
            differences[key] = {"a": flat_a[key], "b": flat_b[key]}

    return {
        "differences": differences,
        "a_only": a_only,
        "b_only": b_only,
        "config_a": config_a_path,
        "config_b": config_b_path,
    }


def create_config(experiment_type, overrides=None):
    """Generate a default config for an experiment type.

    Args:
        experiment_type: One of bmp, sir, two_moons, two_moons_active_learning.
        overrides: Optional dict of overrides to apply.

    Returns:
        dict: Config dict.

    Raises:
        ValueError: If experiment_type is unknown.
    """
    if experiment_type not in _DEFAULTS:
        raise ValueError(
            f"Unknown experiment type '{experiment_type}'. "
            f"Valid types: {', '.join(_DEFAULTS.keys())}"
        )
    import copy
    config = copy.deepcopy(_DEFAULTS[experiment_type])

    if overrides:
        for key, value in overrides.items():
            parts = key.split(".")
            d = config
            for part in parts[:-1]:
                d = d.setdefault(part, {})
            d[parts[-1]] = value

    return config

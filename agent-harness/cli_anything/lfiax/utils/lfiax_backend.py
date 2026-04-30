"""lfiax backend — wraps the actual lfiax library for CLI operations.

The 'real software' for this harness is the lfiax Python package itself.
Unlike GUI-app harnesses that invoke external executables, we import and
call the lfiax library directly.
"""
import os
import sys
import subprocess


def find_lfiax_root():
    """Find the lfiax repository root directory.

    Checks LFIAX_ROOT env var first, then walks up from cwd looking for src/lfiax/.

    Returns:
        str: Absolute path to the lfiax root directory.

    Raises:
        RuntimeError: If lfiax root cannot be found.
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
        "or run from within the lfiax directory.\n"
        "  Install: git clone <lfiax-repo> && cd lfiax && pip install -e ."
    )


def check_lfiax_installed():
    """Check if the lfiax package is importable.

    Returns:
        dict: {installed, version, path}
    """
    try:
        import lfiax
        return {
            "installed": True,
            "version": getattr(lfiax, "__version__", "unknown"),
            "path": os.path.dirname(lfiax.__file__),
        }
    except ImportError:
        return {"installed": False, "version": None, "path": None}


def check_jax_available():
    """Check if JAX is available and report device info.

    Returns:
        dict: {available, version, devices, default_backend}
    """
    try:
        import jax
        devices = jax.devices()
        return {
            "available": True,
            "version": jax.__version__,
            "devices": [{"platform": d.platform, "id": d.id} for d in devices],
            "default_backend": jax.default_backend(),
        }
    except ImportError:
        return {
            "available": False,
            "version": None,
            "devices": [],
            "default_backend": None,
        }


def run_experiment_script(script_path, config_dir=None, config_name=None,
                          overrides=None, workdir=None, blocking=True):
    """Run an lfiax experiment script via subprocess.

    Args:
        script_path: Path to the experiment Python script.
        config_dir: Hydra config directory.
        config_name: Hydra config name (without .yaml).
        overrides: List of Hydra override strings.
        workdir: Working directory for the process.
        blocking: If True, wait for completion.

    Returns:
        dict: {returncode, stdout, stderr, command, pid}
    """
    cmd = [sys.executable, script_path]

    if config_dir:
        cmd.extend(["--config-dir", config_dir])
    if config_name:
        cmd.extend(["--config-name", config_name])
    if overrides:
        cmd.extend(overrides)

    env = os.environ.copy()
    cwd = workdir or os.path.dirname(script_path)

    if blocking:
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=cwd, env=env
        )
        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "command": cmd,
            "pid": None,
        }
    else:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, cwd=cwd, env=env
        )
        return {
            "returncode": None,
            "stdout": None,
            "stderr": None,
            "command": cmd,
            "pid": proc.pid,
        }


def get_environment_info():
    """Get full environment info for diagnostics.

    Returns:
        dict: lfiax, jax, python, and system info.
    """
    return {
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "lfiax": check_lfiax_installed(),
        "jax": check_jax_available(),
        "platform": sys.platform,
    }

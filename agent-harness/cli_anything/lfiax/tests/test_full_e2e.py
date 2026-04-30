"""E2E and CLI subprocess tests for cli-anything-lfiax.

Tests the installed CLI command via subprocess and real workflow scenarios.
"""
import json
import os
import pickle
import subprocess
import sys
import tempfile

import pytest
import yaml


def _resolve_cli(name):
    """Resolve installed CLI command; falls back to python -m for dev.

    Set env CLI_ANYTHING_FORCE_INSTALLED=1 to require the installed command.
    """
    import shutil
    force = os.environ.get("CLI_ANYTHING_FORCE_INSTALLED", "").strip() == "1"
    path = shutil.which(name)
    if path:
        print(f"[_resolve_cli] Using installed command: {path}")
        return [path]
    if force:
        raise RuntimeError(f"{name} not found in PATH. Install with: pip install -e .")
    module = "cli_anything.lfiax.lfiax_cli"
    print(f"[_resolve_cli] Falling back to: {sys.executable} -m {module}")
    return [sys.executable, "-m", "cli_anything.lfiax"]


# ── E2E Workflow Tests ───────────────────────────────────────────────

class TestProjectLifecycleE2E:
    """Test full project create/save/load lifecycle."""

    def test_project_roundtrip(self, tmp_path):
        from cli_anything.lfiax.core.project import (
            create_project, save_project, load_project, add_run, get_project_info,
        )
        proj = create_project("e2e_test", "bmp", config_path="config_bmp.yaml",
                              output_dir=str(tmp_path / "outputs"))
        add_run(proj, "run_001", config_overrides={"lr": 0.001})
        add_run(proj, "run_002", config_overrides={"lr": 0.01})

        path = str(tmp_path / "project.json")
        save_project(proj, path)

        loaded = load_project(path)
        info = get_project_info(loaded)
        assert info["num_runs"] == 2
        assert loaded["runs"][0]["run_id"] == "run_001"
        assert loaded["runs"][1]["config_overrides"]["lr"] == 0.01
        print(f"\n  Project JSON: {path} ({os.path.getsize(path)} bytes)")


class TestConfigWorkflowE2E:
    """Test config load/validate/diff with real YAML files."""

    def test_config_workflow(self, tmp_path):
        from cli_anything.lfiax.core.config import (
            load_config, validate_config, diff_configs, create_config,
        )
        # Create a valid BMP config
        cfg = create_config("bmp")
        path_a = str(tmp_path / "config_a.yaml")
        with open(path_a, "w") as f:
            yaml.dump(cfg, f)

        # Load and validate
        loaded = load_config(path_a)
        result = validate_config(loaded, "bmp")
        assert result["valid"] is True

        # Create a modified config
        cfg_b = create_config("bmp", overrides={"flow_params.num_layers": 10})
        path_b = str(tmp_path / "config_b.yaml")
        with open(path_b, "w") as f:
            yaml.dump(cfg_b, f)

        # Diff them
        diff = diff_configs(path_a, path_b)
        assert "flow_params.num_layers" in diff["differences"]
        print(f"\n  Config A: {path_a}")
        print(f"  Config B: {path_b}")
        print(f"  Differences: {len(diff['differences'])}")


class TestExperimentDryRunE2E:
    """Test dry-run for all experiment types."""

    @pytest.mark.parametrize("exp_type", ["bmp", "sir", "two_moons", "two_moons_active_learning"])
    def test_dry_run(self, tmp_path, exp_type):
        from cli_anything.lfiax.core.experiment import run_experiment
        # Create fake lfiax root
        (tmp_path / "src" / "lfiax").mkdir(parents=True)
        script_name = {"bmp": "BMP.py", "sir": "sir.py",
                       "two_moons": "two_moons.py",
                       "two_moons_active_learning": "two_moons_active_learning.py"}[exp_type]
        (tmp_path / script_name).touch()

        os.environ["LFIAX_ROOT"] = str(tmp_path)
        try:
            result = run_experiment(exp_type, dry_run=True)
            assert result["status"] == "dry_run"
            assert "command" in result
            assert any(script_name in str(c) for c in result["command"])
            print(f"\n  {exp_type} dry-run command: {result.get('command_str', '')}")
        finally:
            del os.environ["LFIAX_ROOT"]


class TestResultsDiscoveryE2E:
    """Test scanning for runs and loading checkpoints."""

    def test_find_and_inspect_runs(self, tmp_path):
        from cli_anything.lfiax.core.results import find_runs, load_checkpoint, get_run_summary

        # Create fake run directories with checkpoints
        run_dir = tmp_path / "outputs" / "bmp_run_001"
        run_dir.mkdir(parents=True)
        ckpt_data = {
            "design_round_0_flow_params_sbi_0": [[0.1, 0.2], [0.3, 0.4]],
            "design_round_0_xi_params_sbi_0": [0.5, 1.0],
        }
        ckpt_path = run_dir / "design_round_0_flow_params_sbi_0.pkl"
        with open(ckpt_path, "wb") as f:
            pickle.dump(ckpt_data, f)

        # Find runs
        runs = find_runs(str(tmp_path / "outputs"))
        assert len(runs) == 1
        assert runs[0]["num_checkpoints"] == 1

        # Load checkpoint
        ckpt = load_checkpoint(str(ckpt_path))
        assert ckpt["num_keys"] == 2
        assert ckpt["size_bytes"] > 0
        print(f"\n  Checkpoint: {ckpt_path} ({ckpt['size_bytes']:,} bytes)")

        # Get run summary
        summary = get_run_summary(str(run_dir))
        assert summary["num_checkpoints"] == 1
        assert summary["design_rounds_completed"] == 1


class TestSimulatorListingE2E:
    """Test simulator discovery and info."""

    def test_all_simulators(self):
        from cli_anything.lfiax.core.simulator import list_simulators, get_simulator_info
        sims = list_simulators()
        assert len(sims) >= 4

        for sim in sims:
            info = get_simulator_info(sim["name"])
            assert "parameters" in info
            assert len(info["parameters"]) == info["num_params"]
            print(f"\n  Simulator: {info['full_name']} ({info['num_params']} params)")


# ── CLI Subprocess Tests ─────────────────────────────────────────────

class TestCLISubprocess:
    """Test the installed CLI command via subprocess."""

    CLI_BASE = _resolve_cli("cli-anything-lfiax")

    def _run(self, args, check=True):
        return subprocess.run(
            self.CLI_BASE + args,
            capture_output=True, text=True, check=check,
        )

    def test_help(self):
        result = self._run(["--help"])
        assert result.returncode == 0
        assert "cli-anything-lfiax" in result.stdout

    def test_env_json(self):
        result = self._run(["--json", "env"])
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert "python" in data
        assert "lfiax" in data
        assert "jax" in data

    def test_simulator_list_json(self):
        result = self._run(["--json", "simulator", "list"])
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert isinstance(data, list)
        assert len(data) >= 4
        names = [s["name"] for s in data]
        assert "bmp" in names

    def test_simulator_info_json(self):
        result = self._run(["--json", "simulator", "info", "sir"])
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["name"] == "sir"
        assert data["num_params"] == 2

    def test_simulator_info_invalid(self):
        result = self._run(["--json", "simulator", "info", "nonexistent"], check=False)
        # Should still exit 0 but with error in output
        data = json.loads(result.stdout)
        assert "error" in data

    def test_config_show(self, tmp_path):
        cfg = {"flow_params": {"num_layers": 5}, "optimization_params": {"lr": 0.001}}
        cfg_path = str(tmp_path / "test_config.yaml")
        with open(cfg_path, "w") as f:
            yaml.dump(cfg, f)
        result = self._run(["--json", "config", "show", cfg_path])
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["flow_params"]["num_layers"] == 5

    def test_config_validate(self, tmp_path):
        cfg = {
            "experiment": {"design_rounds": 5},
            "flow_params": {"num_layers": 5},
            "optimization_params": {"lr": 0.001},
        }
        cfg_path = str(tmp_path / "valid_config.yaml")
        with open(cfg_path, "w") as f:
            yaml.dump(cfg, f)
        result = self._run(["--json", "config", "validate", cfg_path, "--type", "bmp"])
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["valid"] is True

    def test_config_diff(self, tmp_path):
        cfg_a = {"flow_params": {"num_layers": 5}}
        cfg_b = {"flow_params": {"num_layers": 10}}
        path_a = str(tmp_path / "a.yaml")
        path_b = str(tmp_path / "b.yaml")
        with open(path_a, "w") as f:
            yaml.dump(cfg_a, f)
        with open(path_b, "w") as f:
            yaml.dump(cfg_b, f)
        result = self._run(["--json", "config", "diff", path_a, path_b])
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert "flow_params.num_layers" in data["differences"]

    def test_experiment_list_json(self, tmp_path):
        result = self._run(["--json", "experiment", "list",
                            "--output-dir", str(tmp_path)])
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert isinstance(data, list)

    def test_results_list_json(self, tmp_path):
        result = self._run(["--json", "results", "list",
                            "--output-dir", str(tmp_path)])
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert isinstance(data, list)

    def test_oed_describe_json(self):
        result = self._run(["--json", "oed", "describe"])
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["name"] == "lfiax"
        assert "lf_pce_eig_scan" in data["capabilities"]["supported_estimators"]

    def test_oed_validate_example_spec(self):
        import cli_anything.lfiax.examples as ex_pkg
        spec_path = os.path.join(os.path.dirname(ex_pkg.__file__), "linear_spec.json")
        result = self._run(["--json", "oed", "validate", spec_path])
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["valid"] is True
        assert data["backend"] == "lfiax"

    def test_oed_optimize_dry_run_json(self):
        import cli_anything.lfiax.examples as ex_pkg
        spec_path = os.path.join(os.path.dirname(ex_pkg.__file__), "linear_spec.json")
        result = self._run(["--json", "oed", "optimize", spec_path, "--dry-run"])
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["status"] == "dry_run"
        assert data["estimator"] == "lf_pce_eig_scan"
        assert data["validation"]["valid"] is True

    def test_oed_init_scaffold(self, tmp_path):
        out = tmp_path / "scaffold"
        result = self._run(["--json", "oed", "init", str(out)])
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["status"] == "ok"
        assert (out / "linear_problem.py").exists()
        assert (out / "linear_spec.json").exists()

    def test_experiment_dry_run_json(self, tmp_path):
        """Test experiment dry run via subprocess."""
        # Create fake lfiax root
        (tmp_path / "src" / "lfiax").mkdir(parents=True)
        (tmp_path / "BMP.py").touch()

        env = os.environ.copy()
        env["LFIAX_ROOT"] = str(tmp_path)
        result = subprocess.run(
            self.CLI_BASE + ["--json", "experiment", "run", "bmp", "--dry-run"],
            capture_output=True, text=True, env=env,
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["status"] == "dry_run"
        assert "command" in data

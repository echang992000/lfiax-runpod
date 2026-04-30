"""Unit tests for cli-anything-lfiax core modules.

Tests use synthetic data only — no external dependencies required.
"""
import json
import os
import pickle
import tempfile

import pytest
import yaml


# ── project.py ───────────────────────────────────────────────────────

class TestProject:
    def test_create_project(self):
        from cli_anything.lfiax.core.project import create_project
        proj = create_project("test_exp", "bmp", config_path="config_bmp.yaml")
        assert proj["name"] == "test_exp"
        assert proj["experiment_type"] == "bmp"
        assert proj["status"] == "created"
        assert proj["runs"] == []
        assert "created_at" in proj

    def test_create_project_all_types(self):
        from cli_anything.lfiax.core.project import create_project, VALID_EXPERIMENT_TYPES
        for t in VALID_EXPERIMENT_TYPES:
            proj = create_project(f"test_{t}", t)
            assert proj["experiment_type"] == t

    def test_create_project_invalid_type(self):
        from cli_anything.lfiax.core.project import create_project
        with pytest.raises(ValueError, match="Unknown experiment type"):
            create_project("bad", "nonexistent_type")

    def test_save_and_load_project(self):
        from cli_anything.lfiax.core.project import create_project, save_project, load_project
        with tempfile.TemporaryDirectory() as td:
            proj = create_project("roundtrip", "sir")
            path = os.path.join(td, "project.json")
            save_project(proj, path)
            loaded = load_project(path)
            assert loaded["name"] == "roundtrip"
            assert loaded["experiment_type"] == "sir"

    def test_get_project_info(self):
        from cli_anything.lfiax.core.project import create_project, get_project_info
        proj = create_project("info_test", "two_moons")
        info = get_project_info(proj)
        assert info["name"] == "info_test"
        assert info["num_runs"] == 0

    def test_add_run(self):
        from cli_anything.lfiax.core.project import create_project, add_run
        proj = create_project("run_test", "bmp")
        run = add_run(proj, "run_001", config_overrides={"lr": 0.01})
        assert run["run_id"] == "run_001"
        assert proj["status"] == "active"
        assert len(proj["runs"]) == 1

    def test_add_multiple_runs(self):
        from cli_anything.lfiax.core.project import create_project, add_run
        proj = create_project("multi_run", "sir")
        add_run(proj, "run_001")
        add_run(proj, "run_002")
        add_run(proj, "run_003")
        assert len(proj["runs"]) == 3


# ── config.py ────────────────────────────────────────────────────────

class TestConfig:
    def _write_yaml(self, td, name, data):
        path = os.path.join(td, name)
        with open(path, "w") as f:
            yaml.dump(data, f)
        return path

    def test_load_config(self):
        from cli_anything.lfiax.core.config import load_config
        with tempfile.TemporaryDirectory() as td:
            path = self._write_yaml(td, "test.yaml", {"flow_params": {"num_layers": 5}})
            cfg = load_config(path)
            assert cfg["flow_params"]["num_layers"] == 5

    def test_load_config_missing_file(self):
        from cli_anything.lfiax.core.config import load_config
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")

    def test_show_config(self):
        from cli_anything.lfiax.core.config import show_config
        with tempfile.TemporaryDirectory() as td:
            path = self._write_yaml(td, "show.yaml", {"key": "value"})
            data = show_config(path)
            assert data["key"] == "value"

    def test_validate_config_valid(self):
        from cli_anything.lfiax.core.config import validate_config
        cfg = {
            "experiment": {"design_rounds": 5},
            "flow_params": {"num_layers": 5},
            "optimization_params": {"learning_rate": 0.001},
        }
        result = validate_config(cfg, "bmp")
        assert result["valid"] is True
        assert len(result["errors"]) == 0

    def test_validate_config_missing_section(self):
        from cli_anything.lfiax.core.config import validate_config
        cfg = {"flow_params": {"num_layers": 5}}
        result = validate_config(cfg, "bmp")
        assert result["valid"] is False
        assert any("experiment" in e for e in result["errors"])

    def test_validate_config_warnings(self):
        from cli_anything.lfiax.core.config import validate_config
        cfg = {
            "experiment": {},
            "flow_params": {"num_layers": 25, "hidden_size": 2048},
            "optimization_params": {"learning_rate": 0.5},
        }
        result = validate_config(cfg, "bmp")
        assert len(result["warnings"]) >= 2

    def test_diff_configs_identical(self):
        from cli_anything.lfiax.core.config import diff_configs
        with tempfile.TemporaryDirectory() as td:
            data = {"flow_params": {"num_layers": 5}}
            pa = self._write_yaml(td, "a.yaml", data)
            pb = self._write_yaml(td, "b.yaml", data)
            result = diff_configs(pa, pb)
            assert len(result["differences"]) == 0
            assert len(result["a_only"]) == 0
            assert len(result["b_only"]) == 0

    def test_diff_configs_different(self):
        from cli_anything.lfiax.core.config import diff_configs
        with tempfile.TemporaryDirectory() as td:
            pa = self._write_yaml(td, "a.yaml", {"flow_params": {"num_layers": 5}})
            pb = self._write_yaml(td, "b.yaml", {"flow_params": {"num_layers": 10}})
            result = diff_configs(pa, pb)
            assert "flow_params.num_layers" in result["differences"]

    def test_create_config(self):
        from cli_anything.lfiax.core.config import create_config
        cfg = create_config("bmp")
        assert "experiment" in cfg
        assert "flow_params" in cfg
        assert "optimization_params" in cfg

    def test_create_config_with_overrides(self):
        from cli_anything.lfiax.core.config import create_config
        cfg = create_config("bmp", overrides={"flow_params.num_layers": 10})
        assert cfg["flow_params"]["num_layers"] == 10

    def test_create_config_invalid_type(self):
        from cli_anything.lfiax.core.config import create_config
        with pytest.raises(ValueError):
            create_config("nonexistent")


# ── experiment.py ────────────────────────────────────────────────────

class TestExperiment:
    def test_experiment_scripts_mapping(self):
        from cli_anything.lfiax.core.experiment import EXPERIMENT_SCRIPTS
        assert "bmp" in EXPERIMENT_SCRIPTS
        assert "sir" in EXPERIMENT_SCRIPTS
        assert "two_moons" in EXPERIMENT_SCRIPTS
        assert "two_moons_active_learning" in EXPERIMENT_SCRIPTS

    def test_find_lfiax_root_with_env(self, tmp_path):
        from cli_anything.lfiax.core.experiment import find_lfiax_root
        # Create fake lfiax structure
        (tmp_path / "src" / "lfiax").mkdir(parents=True)
        os.environ["LFIAX_ROOT"] = str(tmp_path)
        try:
            root = find_lfiax_root()
            assert root == str(tmp_path)
        finally:
            del os.environ["LFIAX_ROOT"]

    def test_run_experiment_dry_run(self, tmp_path):
        from cli_anything.lfiax.core.experiment import run_experiment
        (tmp_path / "src" / "lfiax").mkdir(parents=True)
        (tmp_path / "BMP.py").touch()
        os.environ["LFIAX_ROOT"] = str(tmp_path)
        try:
            result = run_experiment("bmp", dry_run=True)
            assert result["status"] == "dry_run"
            assert "command" in result
        finally:
            del os.environ["LFIAX_ROOT"]

    def test_run_experiment_invalid_type(self):
        from cli_anything.lfiax.core.experiment import run_experiment
        result = run_experiment("nonexistent", dry_run=True)
        assert result["status"] == "error"

    def test_get_run_status_no_checkpoints(self, tmp_path):
        from cli_anything.lfiax.core.experiment import get_run_status
        result = get_run_status(str(tmp_path))
        assert result["status"] == "empty"
        assert result["num_checkpoints"] == 0

    def test_get_run_status_with_checkpoints(self, tmp_path):
        from cli_anything.lfiax.core.experiment import get_run_status
        (tmp_path / "checkpoint.pkl").write_bytes(pickle.dumps({"test": 1}))
        result = get_run_status(str(tmp_path))
        assert result["num_checkpoints"] == 1

    def test_list_runs_empty(self, tmp_path):
        from cli_anything.lfiax.core.experiment import list_runs
        runs = list_runs(str(tmp_path))
        assert runs == []

    def test_list_runs_nonexistent(self):
        from cli_anything.lfiax.core.experiment import list_runs
        runs = list_runs("/nonexistent/path")
        assert runs == []


# ── results.py ───────────────────────────────────────────────────────

class TestResults:
    def test_find_runs_empty(self, tmp_path):
        from cli_anything.lfiax.core.results import find_runs
        runs = find_runs(str(tmp_path))
        assert runs == []

    def test_load_checkpoint(self, tmp_path):
        from cli_anything.lfiax.core.results import load_checkpoint
        data = {
            "flow_params": {"layer_0": [[0.0] * 4] * 3},
            "xi_params": [1.0, 1.0, 1.0, 1.0, 1.0],
        }
        path = str(tmp_path / "test.pkl")
        with open(path, "wb") as f:
            pickle.dump(data, f)
        result = load_checkpoint(path)
        assert result["num_keys"] == 2
        assert "flow_params" in result["contents"]
        assert "xi_params" in result["contents"]

    def test_load_checkpoint_missing(self):
        from cli_anything.lfiax.core.results import load_checkpoint
        with pytest.raises(FileNotFoundError):
            load_checkpoint("/nonexistent/checkpoint.pkl")

    def test_get_run_summary(self, tmp_path):
        from cli_anything.lfiax.core.results import get_run_summary
        (tmp_path / "design_round_0_flow_params_sbi_0.pkl").write_bytes(
            pickle.dumps({"test": 1})
        )
        result = get_run_summary(str(tmp_path))
        assert result["num_checkpoints"] == 1
        assert result["design_rounds_completed"] == 1

    def test_compare_runs(self, tmp_path):
        from cli_anything.lfiax.core.results import compare_runs
        d1 = tmp_path / "run1"
        d2 = tmp_path / "run2"
        d1.mkdir()
        d2.mkdir()
        result = compare_runs([str(d1), str(d2)])
        assert result["num_runs"] == 2


# ── simulator.py ─────────────────────────────────────────────────────

class TestSimulator:
    def test_list_simulators(self):
        from cli_anything.lfiax.core.simulator import list_simulators
        sims = list_simulators()
        assert len(sims) >= 4
        names = [s["name"] for s in sims]
        assert "bmp" in names
        assert "sir" in names

    def test_get_simulator_info_valid(self):
        from cli_anything.lfiax.core.simulator import get_simulator_info
        info = get_simulator_info("bmp")
        assert info["name"] == "bmp"
        assert info["num_params"] == 4
        assert len(info["parameters"]) == 4

    def test_get_simulator_info_invalid(self):
        from cli_anything.lfiax.core.simulator import get_simulator_info
        with pytest.raises(ValueError, match="Unknown simulator"):
            get_simulator_info("nonexistent_sim")


# ── session.py ───────────────────────────────────────────────────────

class TestSession:
    def test_session_create(self, tmp_path):
        from cli_anything.lfiax.core.session import Session
        s = Session(session_file=str(tmp_path / "session.json"))
        assert s.active_project is None
        assert s.history == []

    def test_session_set_project(self, tmp_path):
        from cli_anything.lfiax.core.session import Session
        s = Session(session_file=str(tmp_path / "session.json"))
        s.set_project("/path/to/project.json")
        assert s.active_project == "/path/to/project.json"

    def test_session_save_load(self, tmp_path):
        from cli_anything.lfiax.core.session import Session
        path = str(tmp_path / "session.json")
        s = Session(session_file=path)
        s.set_project("/test/project.json")
        s.add_to_history("experiment run bmp")
        s.save()

        s2 = Session(session_file=path)
        assert s2.active_project == "/test/project.json"
        assert len(s2.history) == 1

    def test_session_history(self, tmp_path):
        from cli_anything.lfiax.core.session import Session
        s = Session(session_file=str(tmp_path / "session.json"))
        s.add_to_history("cmd1")
        s.add_to_history("cmd2")
        s.add_to_history("cmd3")
        assert len(s.history) == 3
        assert s.history[0]["command"] == "cmd1"

    def test_session_to_dict(self, tmp_path):
        from cli_anything.lfiax.core.session import Session
        s = Session(session_file=str(tmp_path / "session.json"))
        d = s.to_dict()
        assert "session_file" in d
        assert "active_project" in d
        assert "history_count" in d


# ── spec.py ──────────────────────────────────────────────────────────

class TestSpec:
    def _sample_spec(self):
        return {
            "backend": "lfiax",
            "simulator_ref": "cli_anything.lfiax.examples.linear_problem:simulator",
            "prior_sampler_ref": "cli_anything.lfiax.examples.linear_problem:prior",
            "design_variables": [{"name": "xi", "lower": -2.0, "upper": 2.0, "initial": 0.25}],
            "objective": {"estimator": "lf_pce_eig_scan"},
        }

    def test_normalize_fills_defaults(self):
        from cli_anything.lfiax.core.spec import normalize_spec
        spec = normalize_spec(self._sample_spec())
        assert spec["compute_budget"]["num_outer_samples"] == 256
        assert spec["compute_budget"]["num_inner_samples"] == 10
        assert spec["artifacts"]["output_dir"] == "artifacts"
        assert spec["objective"]["name"] == "expected_information_gain"

    def test_load_spec_roundtrip(self, tmp_path):
        from cli_anything.lfiax.core.spec import load_spec
        p = tmp_path / "spec.json"
        p.write_text(json.dumps(self._sample_spec()))
        spec = load_spec(str(p))
        assert spec["backend"] == "lfiax"
        assert len(spec["design_variables"]) == 1

    def test_load_spec_missing(self):
        from cli_anything.lfiax.core.spec import load_spec
        with pytest.raises(FileNotFoundError):
            load_spec("/nonexistent/spec.json")

    def test_validate_spec_valid(self):
        from cli_anything.lfiax.core.spec import normalize_spec, validate_spec
        report = validate_spec(normalize_spec(self._sample_spec()))
        assert report["valid"] is True
        assert report["backend"] == "lfiax"

    def test_validate_spec_missing_prior(self):
        from cli_anything.lfiax.core.spec import normalize_spec, validate_spec
        raw = self._sample_spec()
        raw.pop("prior_sampler_ref")
        report = validate_spec(normalize_spec(raw))
        assert report["valid"] is False
        assert any("prior_sampler_ref_or_latent_sampler_ref" in e["path"]
                   for e in report["errors"])

    def test_validate_spec_bad_bounds(self):
        from cli_anything.lfiax.core.spec import normalize_spec, validate_spec
        raw = self._sample_spec()
        raw["design_variables"][0]["lower"] = 5.0
        raw["design_variables"][0]["upper"] = 1.0
        report = validate_spec(normalize_spec(raw))
        assert report["valid"] is False

    def test_effective_initial_design_uses_initial(self):
        from cli_anything.lfiax.core.spec import normalize_spec, effective_initial_design
        spec = normalize_spec(self._sample_spec())
        assert effective_initial_design(spec) == [0.25]

    def test_effective_initial_design_midpoint_fallback(self):
        from cli_anything.lfiax.core.spec import normalize_spec, effective_initial_design
        raw = self._sample_spec()
        raw["design_variables"][0].pop("initial")
        spec = normalize_spec(raw)
        assert effective_initial_design(spec) == [0.0]

    def test_design_bounds(self):
        from cli_anything.lfiax.core.spec import normalize_spec, design_bounds
        spec = normalize_spec(self._sample_spec())
        lower, upper = design_bounds(spec)
        assert lower == [-2.0] and upper == [2.0]

    def test_design_mode_defaults_to_point(self):
        from cli_anything.lfiax.core.spec import normalize_spec, design_mode
        spec = normalize_spec(self._sample_spec())
        assert design_mode(spec) == "point"

    def test_design_mode_distribution_fills_defaults(self):
        from cli_anything.lfiax.core.spec import normalize_spec, design_mode
        raw = self._sample_spec()
        raw["backend_options"] = {"design_mode": "distribution"}
        spec = normalize_spec(raw)
        assert design_mode(spec) == "distribution"
        bo = spec["backend_options"]
        assert "xi_stddev_min" in bo
        assert "end_sigma" in bo
        assert "decay_rate" in bo

    def test_execution_path_defaults_to_black_box(self):
        from cli_anything.lfiax.core.spec import execution_path, normalize_spec
        spec = normalize_spec(self._sample_spec())
        assert execution_path(spec) == "black_box"

    def test_execution_path_uses_differentiable_flag(self):
        from cli_anything.lfiax.core.spec import execution_path, normalize_spec
        raw = self._sample_spec()
        raw["differentiable"] = True
        spec = normalize_spec(raw)
        assert execution_path(spec) == "differentiable"

    def test_validate_rejects_unknown_design_mode(self):
        from cli_anything.lfiax.core.spec import normalize_spec, validate_spec
        raw = self._sample_spec()
        raw["backend_options"] = {"design_mode": "bogus"}
        report = validate_spec(normalize_spec(raw))
        assert report["valid"] is False
        assert any("design_mode" in e["path"] for e in report["errors"])

    def test_initial_design_distribution_defaults(self):
        from cli_anything.lfiax.core.spec import normalize_spec, initial_design_distribution
        raw = self._sample_spec()
        raw["backend_options"] = {"design_mode": "distribution"}
        spec = normalize_spec(raw)
        mu, sd = initial_design_distribution(spec)
        # midpoint of [-2, 2] is 0; stddev default = (4)/4 = 1
        assert mu == [0.0]
        assert sd == [1.0]

    def test_initial_design_distribution_explicit(self):
        from cli_anything.lfiax.core.spec import normalize_spec, initial_design_distribution
        raw = self._sample_spec()
        raw["backend_options"] = {
            "design_mode": "distribution",
            "xi_mu_init": [0.5],
            "xi_stddev_init": [0.2],
        }
        spec = normalize_spec(raw)
        mu, sd = initial_design_distribution(spec)
        assert mu == [0.5] and sd == [0.2]

    def test_initial_design_distribution_wrong_len(self):
        from cli_anything.lfiax.core.spec import normalize_spec, initial_design_distribution
        raw = self._sample_spec()
        raw["backend_options"] = {
            "design_mode": "distribution",
            "xi_mu_init": [0.5, 1.0],
        }
        spec = normalize_spec(raw)
        with pytest.raises(ValueError):
            initial_design_distribution(spec)


# ── problem.py ───────────────────────────────────────────────────────

class TestProblem:
    def test_resolve_reference_module(self):
        from cli_anything.lfiax.core.problem import resolve_reference
        fn = resolve_reference("cli_anything.lfiax.examples.linear_problem:prior")
        assert callable(fn)

    def test_resolve_reference_invalid_format(self):
        from cli_anything.lfiax.core.problem import resolve_reference
        with pytest.raises(ValueError):
            resolve_reference("no_colon_here")

    def test_resolve_reference_bad_attr(self):
        from cli_anything.lfiax.core.problem import resolve_reference
        with pytest.raises(AttributeError):
            resolve_reference("cli_anything.lfiax.examples.linear_problem:nonexistent_fn")

    def test_resolve_reference_from_file(self, tmp_path):
        from cli_anything.lfiax.core.problem import resolve_reference
        py = tmp_path / "prob.py"
        py.write_text("def hello():\n    return 42\n")
        fn = resolve_reference(f"{py}:hello")
        assert fn() == 42

    def test_load_problem_requires_simulator(self):
        from cli_anything.lfiax.core.problem import load_problem
        with pytest.raises(ValueError):
            load_problem({})

    def test_load_problem_resolves(self):
        from cli_anything.lfiax.core.problem import load_problem
        spec = {
            "simulator_ref": "cli_anything.lfiax.examples.linear_problem:simulator",
            "prior_sampler_ref": "cli_anything.lfiax.examples.linear_problem:prior",
        }
        problem = load_problem(spec)
        assert callable(problem["simulator"])
        assert callable(problem["prior"])


# ── oed.py ───────────────────────────────────────────────────────────

class TestOED:
    def test_describe_backend(self):
        from cli_anything.lfiax.core.oed import describe_backend
        info = describe_backend()
        assert info["name"] == "lfiax"
        assert "lf_pce_eig_scan" in info["capabilities"]["supported_estimators"]

    def test_plan_run(self):
        from cli_anything.lfiax.core.spec import normalize_spec
        from cli_anything.lfiax.core.oed import plan_run
        spec = normalize_spec({
            "backend": "lfiax",
            "simulator_ref": "cli_anything.lfiax.examples.linear_problem:simulator",
            "prior_sampler_ref": "cli_anything.lfiax.examples.linear_problem:prior",
            "design_variables": [{"name": "xi", "lower": -2.0, "upper": 2.0, "initial": 0.25}],
            "objective": {"estimator": "lf_pce_eig_scan"},
        })
        plan = plan_run(spec, seed=7)
        assert plan["status"] == "dry_run"
        assert plan["initial_design"] == [0.25]
        assert plan["validation"]["valid"] is True
        assert plan["seed"] == 7
        assert plan["execution_path"] == "black_box"

    def test_optimize_design_dry_run(self):
        from cli_anything.lfiax.core.spec import normalize_spec
        from cli_anything.lfiax.core.oed import optimize_design
        spec = normalize_spec({
            "backend": "lfiax",
            "simulator_ref": "cli_anything.lfiax.examples.linear_problem:simulator",
            "prior_sampler_ref": "cli_anything.lfiax.examples.linear_problem:prior",
            "design_variables": [{"name": "xi", "lower": -2.0, "upper": 2.0, "initial": 0.25}],
            "objective": {"estimator": "lf_pce_eig_scan"},
        })
        result = optimize_design(spec, dry_run=True, write_artifacts=False)
        assert result["status"] == "dry_run"

    def test_plan_run_distribution_mode(self):
        from cli_anything.lfiax.core.spec import normalize_spec
        from cli_anything.lfiax.core.oed import plan_run
        spec = normalize_spec({
            "backend": "lfiax",
            "simulator_ref": "cli_anything.lfiax.examples.linear_problem:simulator",
            "prior_sampler_ref": "cli_anything.lfiax.examples.linear_problem:prior",
            "design_variables": [{"name": "xi", "lower": -2.0, "upper": 2.0}],
            "objective": {"estimator": "lf_pce_eig_scan"},
            "backend_options": {
                "design_mode": "distribution",
                "xi_mu_init": [0.1],
                "xi_stddev_init": [0.3],
            },
        })
        plan = plan_run(spec)
        assert plan["design_mode"] == "distribution"
        assert plan["xi_mu_init"] == [0.1]
        assert plan["xi_stddev_init"] == [0.3]
        assert plan["annealing"]["end_sigma"] == [0.01]
        assert "initial_design" not in plan

    def test_describe_backend_lists_both_modes(self):
        from cli_anything.lfiax.core.oed import describe_backend
        info = describe_backend()
        assert "point" in info["capabilities"]["design_modes"]
        assert "distribution" in info["capabilities"]["design_modes"]

    def test_compact_result_payload_omits_stepwise_history(self):
        from cli_anything.lfiax.core.oed import _compact_result_payload

        result = {
            "history": [
                {"step": index, "design": [float(index)], "eig": float(index)}
                for index in range(8)
            ],
            "artifacts": {},
        }

        compact = _compact_result_payload(result)

        assert compact["history_summary"]["num_steps"] == 8
        assert "history" not in compact
        assert "checkpoints" not in compact["history_summary"]

    def test_optimize_design_invalid_spec(self):
        from cli_anything.lfiax.core.spec import normalize_spec
        from cli_anything.lfiax.core.oed import optimize_design
        spec = normalize_spec({
            "backend": "lfiax",
            "simulator_ref": "",
            "design_variables": [],
            "objective": {},
        })
        result = optimize_design(spec, write_artifacts=False)
        assert result["status"] == "invalid_spec"
        assert result["validation"]["valid"] is False

    def test_optimize_design_black_box_smoke(self, tmp_path):
        pytest.importorskip("jax")
        pytest.importorskip("haiku")
        pytest.importorskip("optax")
        spec = {
            "backend": "lfiax",
            "simulator_ref": "cli_anything.lfiax.examples.linear_problem:simulator",
            "prior_sampler_ref": "cli_anything.lfiax.examples.linear_problem:prior",
            "differentiable": False,
            "design_variables": [{"name": "xi", "lower": -2.0, "upper": 2.0, "initial": 0.25}],
            "objective": {"estimator": "lf_pce_eig_scan"},
            "compute_budget": {
                "num_outer_samples": 8,
                "num_inner_samples": 2,
                "num_optimization_steps": 2,
                "design_learning_rate": 0.01,
                "flow_learning_rate": 0.001,
            },
            "artifacts": {"output_dir": str(tmp_path)},
        }
        from cli_anything.lfiax.core.oed import optimize_design
        from cli_anything.lfiax.core.spec import normalize_spec
        result = optimize_design(normalize_spec(spec), write_artifacts=True)
        assert result["status"] == "completed"
        assert result["execution_path"] == "black_box"
        assert os.path.exists(result["artifacts"]["likelihood_checkpoint"])
        assert os.path.exists(result["artifacts"]["likelihood_metadata_json"])
        assert os.path.exists(result["artifacts"]["optimization_history_npz"])

    def test_optimize_design_differentiable_smoke(self, tmp_path):
        pytest.importorskip("jax")
        pytest.importorskip("haiku")
        pytest.importorskip("optax")
        spec = {
            "backend": "lfiax",
            "simulator_ref": "cli_anything.lfiax.examples.linear_problem:simulator",
            "prior_sampler_ref": "cli_anything.lfiax.examples.linear_problem:prior",
            "differentiable": True,
            "design_variables": [{"name": "xi", "lower": -2.0, "upper": 2.0, "initial": 0.25}],
            "objective": {"estimator": "lf_pce_eig_scan"},
            "compute_budget": {
                "num_outer_samples": 8,
                "num_inner_samples": 2,
                "num_optimization_steps": 2,
                "design_learning_rate": 0.01,
                "flow_learning_rate": 0.001,
            },
            "artifacts": {"output_dir": str(tmp_path)},
        }
        from cli_anything.lfiax.core.oed import optimize_design
        from cli_anything.lfiax.core.spec import normalize_spec
        result = optimize_design(normalize_spec(spec), write_artifacts=True)
        assert result["status"] == "completed"
        assert result["execution_path"] == "differentiable"
        assert os.path.exists(result["artifacts"]["likelihood_checkpoint"])
        assert os.path.exists(result["artifacts"]["likelihood_metadata_json"])
        assert os.path.exists(result["artifacts"]["optimization_history_npz"])

    def test_distribution_mode_sigma_history_narrows(self, tmp_path):
        pytest.importorskip("jax")
        pytest.importorskip("haiku")
        pytest.importorskip("optax")
        spec = {
            "backend": "lfiax",
            "simulator_ref": "cli_anything.lfiax.examples.linear_problem:simulator",
            "prior_sampler_ref": "cli_anything.lfiax.examples.linear_problem:prior",
            "differentiable": False,
            "design_variables": [{"name": "xi", "lower": -2.0, "upper": 2.0}],
            "objective": {"estimator": "lf_pce_eig_scan"},
            "compute_budget": {
                "num_outer_samples": 8,
                "num_inner_samples": 2,
                "num_optimization_steps": 4,
                "design_learning_rate": 0.01,
                "flow_learning_rate": 0.001,
            },
            "backend_options": {
                "design_mode": "distribution",
                "xi_mu_init": [0.0],
                "xi_stddev_init": [1.0],
                "end_sigma": 0.05,
                "decay_rate": 10.0,
            },
            "artifacts": {"output_dir": str(tmp_path)},
        }
        from cli_anything.lfiax.core.oed import optimize_design
        from cli_anything.lfiax.core.spec import normalize_spec
        result = optimize_design(normalize_spec(spec), write_artifacts=False)
        assert result["status"] == "completed"
        sigma_history = result["artifacts"]["sigma_history"]
        assert sigma_history[0][0] > sigma_history[-1][0]
        assert "xi_stddev" in result


# ── lfiax_backend.py ─────────────────────────────────────────────────

class TestBackend:
    def test_check_lfiax_installed(self):
        from cli_anything.lfiax.utils.lfiax_backend import check_lfiax_installed
        result = check_lfiax_installed()
        assert "installed" in result
        assert isinstance(result["installed"], bool)

    def test_check_jax_available(self):
        from cli_anything.lfiax.utils.lfiax_backend import check_jax_available
        result = check_jax_available()
        assert "available" in result
        assert isinstance(result["available"], bool)

    def test_get_environment_info(self):
        from cli_anything.lfiax.utils.lfiax_backend import get_environment_info
        info = get_environment_info()
        assert "python" in info
        assert "lfiax" in info
        assert "jax" in info
        assert "platform" in info

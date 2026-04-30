# TEST.md — cli-anything-lfiax Test Plan & Results

## Part 1: Test Plan

### Test Inventory

| File | Type | Estimated Tests |
|------|------|----------------|
| `test_core.py` | Unit tests | ~30 |
| `test_full_e2e.py` | E2E + subprocess tests | ~15 |

---

### Unit Test Plan (`test_core.py`)

#### Module: `core/project.py`
- `test_create_project` — Creates valid project dict with all required fields
- `test_create_project_invalid_type` — Rejects unknown experiment types
- `test_save_and_load_project` — Round-trip JSON serialization
- `test_get_project_info` — Returns summary dict
- `test_add_run` — Adds run entry to project
- `test_add_multiple_runs` — Multiple runs accumulate correctly

#### Module: `core/config.py`
- `test_load_config` — Loads YAML config file
- `test_load_config_missing_file` — Error on missing file
- `test_show_config` — Returns parsed dict
- `test_validate_config_valid` — Valid BMP config passes
- `test_validate_config_missing_section` — Detects missing required sections
- `test_validate_config_warnings` — Reports warnings for unusual values
- `test_diff_configs_identical` — No differences for same config
- `test_diff_configs_different` — Reports differences correctly
- `test_create_config` — Generates valid default config

#### Module: `core/experiment.py`
- `test_experiment_scripts_mapping` — All experiment types mapped
- `test_find_lfiax_root_with_env` — Finds root via LFIAX_ROOT env var
- `test_find_lfiax_root_from_cwd` — Finds root by walking up
- `test_run_experiment_dry_run` — Dry run returns command without executing
- `test_get_run_status_no_checkpoints` — Reports empty run dir
- `test_get_run_status_with_checkpoints` — Reports checkpoint info
- `test_list_runs_empty` — Empty dir returns empty list

#### Module: `core/results.py`
- `test_find_runs_empty` — No runs in empty dir
- `test_load_checkpoint` — Loads and summarizes pickle file
- `test_get_run_summary` — Returns structured summary
- `test_compare_runs` — Compares multiple run dirs

#### Module: `core/simulator.py`
- `test_list_simulators` — Returns all simulators
- `test_get_simulator_info_valid` — Returns info for known simulator
- `test_get_simulator_info_invalid` — Error for unknown simulator

#### Module: `core/session.py`
- `test_session_create` — Creates new session
- `test_session_set_project` — Sets active project
- `test_session_save_load` — Round-trip persistence
- `test_session_history` — Command history tracking
- `test_session_to_dict` — JSON serializable

#### Module: `utils/lfiax_backend.py`
- `test_check_lfiax_installed` — Reports installation status
- `test_check_jax_available` — Reports JAX availability
- `test_get_environment_info` — Returns complete env info

---

### E2E Test Plan (`test_full_e2e.py`)

#### Real Workflow Tests
1. **Project lifecycle** — Create project, add runs, save, reload, verify state
2. **Config workflow** — Load real config, validate, modify, diff against original
3. **Experiment dry run** — Dry-run all experiment types, verify commands are correct
4. **Results discovery** — Scan real outputs directory (if available), load checkpoints
5. **Simulator listing** — List all simulators, verify info completeness

#### CLI Subprocess Tests (TestCLISubprocess)
Uses `_resolve_cli("cli-anything-lfiax")` to test the installed command:
1. `test_help` — `--help` returns 0
2. `test_version_json` — `--json env` returns valid JSON
3. `test_simulator_list_json` — `--json simulator list` returns JSON array
4. `test_config_show` — Shows a real config file
5. `test_config_validate` — Validates a config file
6. `test_experiment_dry_run` — Dry run via subprocess
7. `test_project_lifecycle` — Create/save/load project via subprocess
8. `test_experiment_list` — Lists runs (may be empty)

---

### Realistic Workflow Scenarios

#### Scenario 1: "New BMP Experiment"
**Simulates**: Researcher setting up and running a BMP OED experiment
**Operations chained**:
1. Show default BMP config
2. Validate config
3. Dry-run experiment with custom overrides
4. Check environment is ready (JAX, lfiax installed)
**Verified**: Commands produce valid output, dry-run command is correct

#### Scenario 2: "Results Comparison"
**Simulates**: Comparing results from multiple experiment runs
**Operations chained**:
1. List available runs
2. Show summary for each run
3. Inspect checkpoints
**Verified**: Run discovery works, checkpoint loading succeeds

#### Scenario 3: "Agent-Driven Experiment"
**Simulates**: AI agent running experiments via --json mode
**Operations chained**:
1. `--json env` — Check environment
2. `--json simulator list` — Discover simulators
3. `--json config show config_bmp.yaml` — Read config
4. `--json experiment run bmp --dry-run` — Plan experiment
**Verified**: All outputs are valid JSON, parseable by agent

---

### Added Coverage for `oed` Group (2026-04-04)

The new generic BOED optimizer (`core/spec.py`, `core/problem.py`,
`core/oed.py`) and the `oed` CLI group are covered by:

- **TestSpec** (9) — normalize defaults, load/roundtrip, missing file, valid /
  missing-prior / bad-bounds validation, `effective_initial_design` with and
  without explicit initials, `design_bounds`.
- **TestProblem** (6) — resolve module refs, invalid format, missing attribute,
  file-path ref loader, `load_problem` requires simulator, full resolve.
- **TestOED** (4) — `describe_backend`, `plan_run`, `optimize_design(dry_run=True)`,
  and the invalid-spec guard.
- **TestCLISubprocess** (4 new) — `oed describe`, `oed validate` against the
  shipped example spec, `oed optimize --dry-run`, and `oed init` scaffolding.

## Part 2: Test Results

### Test Run: 2026-04-04 (post-oed reshape)

**Environment:** Python 3.11.10, pytest 8.3.3, macOS Darwin 24.6.0
**Mode:** `CLI_ANYTHING_FORCE_INSTALLED=1`

```
============================== 84 passed in 0.84s ==============================
```

| Metric | Value |
|--------|-------|
| Total tests | 84 |
| Passed | 84 |
| Failed | 0 |
| Pass rate | **100%** |
| New tests vs 2026-04-03 | +23 (spec/problem/oed + oed subprocess) |

### Test Run: 2026-04-03

**Environment:** Python 3.11.10, pytest 8.3.3, macOS Darwin 24.6.0
**Mode:** `CLI_ANYTHING_FORCE_INSTALLED=1` (installed command verified)
**CLI Path:** `/Users/vincentzaballa/anaconda3/bin/cli-anything-lfiax`

```
============================= test session starts ==============================
platform darwin -- Python 3.11.10, pytest-8.3.3, pluggy-1.5.0
collected 61 items

cli_anything/lfiax/tests/test_core.py::TestProject::test_create_project PASSED
cli_anything/lfiax/tests/test_core.py::TestProject::test_create_project_all_types PASSED
cli_anything/lfiax/tests/test_core.py::TestProject::test_create_project_invalid_type PASSED
cli_anything/lfiax/tests/test_core.py::TestProject::test_save_and_load_project PASSED
cli_anything/lfiax/tests/test_core.py::TestProject::test_get_project_info PASSED
cli_anything/lfiax/tests/test_core.py::TestProject::test_add_run PASSED
cli_anything/lfiax/tests/test_core.py::TestProject::test_add_multiple_runs PASSED
cli_anything/lfiax/tests/test_core.py::TestConfig::test_load_config PASSED
cli_anything/lfiax/tests/test_core.py::TestConfig::test_load_config_missing_file PASSED
cli_anything/lfiax/tests/test_core.py::TestConfig::test_show_config PASSED
cli_anything/lfiax/tests/test_core.py::TestConfig::test_validate_config_valid PASSED
cli_anything/lfiax/tests/test_core.py::TestConfig::test_validate_config_missing_section PASSED
cli_anything/lfiax/tests/test_core.py::TestConfig::test_validate_config_warnings PASSED
cli_anything/lfiax/tests/test_core.py::TestConfig::test_diff_configs_identical PASSED
cli_anything/lfiax/tests/test_core.py::TestConfig::test_diff_configs_different PASSED
cli_anything/lfiax/tests/test_core.py::TestConfig::test_create_config PASSED
cli_anything/lfiax/tests/test_core.py::TestConfig::test_create_config_with_overrides PASSED
cli_anything/lfiax/tests/test_core.py::TestConfig::test_create_config_invalid_type PASSED
cli_anything/lfiax/tests/test_core.py::TestExperiment::test_experiment_scripts_mapping PASSED
cli_anything/lfiax/tests/test_core.py::TestExperiment::test_find_lfiax_root_with_env PASSED
cli_anything/lfiax/tests/test_core.py::TestExperiment::test_run_experiment_dry_run PASSED
cli_anything/lfiax/tests/test_core.py::TestExperiment::test_run_experiment_invalid_type PASSED
cli_anything/lfiax/tests/test_core.py::TestExperiment::test_get_run_status_no_checkpoints PASSED
cli_anything/lfiax/tests/test_core.py::TestExperiment::test_get_run_status_with_checkpoints PASSED
cli_anything/lfiax/tests/test_core.py::TestExperiment::test_list_runs_empty PASSED
cli_anything/lfiax/tests/test_core.py::TestExperiment::test_list_runs_nonexistent PASSED
cli_anything/lfiax/tests/test_core.py::TestResults::test_find_runs_empty PASSED
cli_anything/lfiax/tests/test_core.py::TestResults::test_load_checkpoint PASSED
cli_anything/lfiax/tests/test_core.py::TestResults::test_load_checkpoint_missing PASSED
cli_anything/lfiax/tests/test_core.py::TestResults::test_get_run_summary PASSED
cli_anything/lfiax/tests/test_core.py::TestResults::test_compare_runs PASSED
cli_anything/lfiax/tests/test_core.py::TestSimulator::test_list_simulators PASSED
cli_anything/lfiax/tests/test_core.py::TestSimulator::test_get_simulator_info_valid PASSED
cli_anything/lfiax/tests/test_core.py::TestSimulator::test_get_simulator_info_invalid PASSED
cli_anything/lfiax/tests/test_core.py::TestSession::test_session_create PASSED
cli_anything/lfiax/tests/test_core.py::TestSession::test_session_set_project PASSED
cli_anything/lfiax/tests/test_core.py::TestSession::test_session_save_load PASSED
cli_anything/lfiax/tests/test_core.py::TestSession::test_session_history PASSED
cli_anything/lfiax/tests/test_core.py::TestSession::test_session_to_dict PASSED
cli_anything/lfiax/tests/test_core.py::TestBackend::test_check_lfiax_installed PASSED
cli_anything/lfiax/tests/test_core.py::TestBackend::test_check_jax_available PASSED
cli_anything/lfiax/tests/test_core.py::TestBackend::test_get_environment_info PASSED
cli_anything/lfiax/tests/test_full_e2e.py::TestProjectLifecycleE2E::test_project_roundtrip PASSED
cli_anything/lfiax/tests/test_full_e2e.py::TestConfigWorkflowE2E::test_config_workflow PASSED
cli_anything/lfiax/tests/test_full_e2e.py::TestExperimentDryRunE2E::test_dry_run[bmp] PASSED
cli_anything/lfiax/tests/test_full_e2e.py::TestExperimentDryRunE2E::test_dry_run[sir] PASSED
cli_anything/lfiax/tests/test_full_e2e.py::TestExperimentDryRunE2E::test_dry_run[two_moons] PASSED
cli_anything/lfiax/tests/test_full_e2e.py::TestExperimentDryRunE2E::test_dry_run[two_moons_active_learning] PASSED
cli_anything/lfiax/tests/test_full_e2e.py::TestResultsDiscoveryE2E::test_find_and_inspect_runs PASSED
cli_anything/lfiax/tests/test_full_e2e.py::TestSimulatorListingE2E::test_all_simulators PASSED
cli_anything/lfiax/tests/test_full_e2e.py::TestCLISubprocess::test_help PASSED
cli_anything/lfiax/tests/test_full_e2e.py::TestCLISubprocess::test_env_json PASSED
cli_anything/lfiax/tests/test_full_e2e.py::TestCLISubprocess::test_simulator_list_json PASSED
cli_anything/lfiax/tests/test_full_e2e.py::TestCLISubprocess::test_simulator_info_json PASSED
cli_anything/lfiax/tests/test_full_e2e.py::TestCLISubprocess::test_simulator_info_invalid PASSED
cli_anything/lfiax/tests/test_full_e2e.py::TestCLISubprocess::test_config_show PASSED
cli_anything/lfiax/tests/test_full_e2e.py::TestCLISubprocess::test_config_validate PASSED
cli_anything/lfiax/tests/test_full_e2e.py::TestCLISubprocess::test_config_diff PASSED
cli_anything/lfiax/tests/test_full_e2e.py::TestCLISubprocess::test_experiment_list_json PASSED
cli_anything/lfiax/tests/test_full_e2e.py::TestCLISubprocess::test_results_list_json PASSED
cli_anything/lfiax/tests/test_full_e2e.py::TestCLISubprocess::test_experiment_dry_run_json PASSED

============================== 61 passed in 0.64s ==============================
```

### Summary Statistics

| Metric | Value |
|--------|-------|
| Total tests | 61 |
| Passed | 61 |
| Failed | 0 |
| Pass rate | **100%** |
| Execution time | 0.64s |
| Unit tests (test_core.py) | 42 |
| E2E tests (test_full_e2e.py) | 19 |
| Subprocess tests | 11 |

### Coverage Notes

- All core modules fully tested: project, config, experiment, results, simulator, session, backend
- CLI subprocess tests verify `--json` output, `--help`, config operations, experiment dry runs
- Force-installed mode confirmed: `[_resolve_cli] Using installed command: /Users/vincentzaballa/anaconda3/bin/cli-anything-lfiax`
- Environment-dependent features (JAX device probing, lfiax import) tested with graceful handling

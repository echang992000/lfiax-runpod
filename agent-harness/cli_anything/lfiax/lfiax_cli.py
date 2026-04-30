"""cli-anything-lfiax: Unified CLI for Likelihood Free Inference in JAX.

Provides experiment management, config handling, results inspection,
and interactive REPL for the lfiax BOED framework.
"""
import os
import sys
import json
import click


def _json_output(ctx, data):
    """Print JSON if --json flag is set."""
    if ctx.obj.get("json_mode"):
        click.echo(json.dumps(data, indent=2, default=str))
        return True
    return False


def _get_skin():
    from cli_anything.lfiax.utils.repl_skin import ReplSkin
    return ReplSkin("lfiax", version="1.0.0")


# -- Main group --

@click.group(invoke_without_command=True)
@click.option("--json", "json_mode", is_flag=True, help="Output in JSON format")
@click.option("--project", type=click.Path(), help="Path to project JSON file")
@click.pass_context
def cli(ctx, json_mode, project):
    """cli-anything-lfiax: CLI harness for Likelihood Free Inference in JAX."""
    ctx.ensure_object(dict)
    ctx.obj["json_mode"] = json_mode
    ctx.obj["project"] = project
    if ctx.invoked_subcommand is None:
        ctx.invoke(repl)


# ── experiment group ─────────────────────────────────────────────────

@cli.group()
@click.pass_context
def experiment(ctx):
    """Manage and run BOED experiments."""
    pass


@experiment.command("run")
@click.argument("experiment_type", type=click.Choice(
    ["bmp", "sir", "two_moons", "two_moons_active_learning"]))
@click.option("--config", "config_path", type=click.Path(exists=True),
              help="Custom config YAML")
@click.option("--workdir", type=click.Path(), help="Output directory")
@click.option("--seed", type=int, help="RNG seed")
@click.option("--device", type=click.Choice(["cpu", "gpu", "tpu"]),
              help="Compute device")
@click.option("--override", "-o", multiple=True,
              help="Hydra config overrides (key=value)")
@click.option("--dry-run", is_flag=True, help="Print command without executing")
@click.pass_context
def experiment_run(ctx, experiment_type, config_path, workdir, seed,
                   device, override, dry_run):
    """Run an experiment (BMP, SIR, two_moons, etc.)."""
    from cli_anything.lfiax.core.experiment import run_experiment
    result = run_experiment(
        experiment_type, config_path=config_path, workdir=workdir,
        seed=seed, device=device, overrides=list(override), dry_run=dry_run,
    )
    if not _json_output(ctx, result):
        skin = _get_skin()
        if dry_run:
            skin.info(f"Dry run command: {' '.join(result.get('command', []))}")
        elif result.get("status") == "error":
            skin.error(result.get("error", "Unknown error"))
        elif result.get("returncode", 1) == 0:
            skin.success(f"Experiment '{experiment_type}' completed")
        else:
            skin.error(f"Experiment failed (exit code {result.get('returncode')})")
            if result.get("stderr"):
                click.echo(result["stderr"][:500])


@experiment.command("status")
@click.argument("workdir", type=click.Path(exists=True))
@click.pass_context
def experiment_status(ctx, workdir):
    """Check status of an experiment run."""
    from cli_anything.lfiax.core.experiment import get_run_status
    result = get_run_status(workdir)
    if not _json_output(ctx, result):
        skin = _get_skin()
        skin.status_block(result)


@experiment.command("list")
@click.option("--output-dir", default="./outputs", help="Base output directory")
@click.pass_context
def experiment_list(ctx, output_dir):
    """List all experiment runs."""
    from cli_anything.lfiax.core.experiment import list_runs
    runs = list_runs(output_dir)
    if not _json_output(ctx, runs):
        skin = _get_skin()
        if not runs:
            skin.info("No runs found")
            return
        headers = ["Run", "Type", "Status", "Created"]
        rows = [
            [r.get("name", "?"), r.get("type", "?"),
             r.get("status", "?"), r.get("created", "?")]
            for r in runs
        ]
        skin.table(headers, rows)


# ── config group ─────────────────────────────────────────────────────

@cli.group()
@click.pass_context
def config(ctx):
    """Manage experiment configurations."""
    pass


@config.command("show")
@click.argument("config_path", type=click.Path(exists=True))
@click.pass_context
def config_show(ctx, config_path):
    """Display a configuration file."""
    from cli_anything.lfiax.core.config import show_config
    data = show_config(config_path)
    if not _json_output(ctx, data):
        import yaml
        click.echo(yaml.dump(data, default_flow_style=False))


@config.command("validate")
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--type", "experiment_type",
              type=click.Choice(["bmp", "sir", "two_moons", "two_moons_active_learning"]))
@click.pass_context
def config_validate(ctx, config_path, experiment_type):
    """Validate a configuration file."""
    from cli_anything.lfiax.core.config import load_config, validate_config
    cfg = load_config(config_path)
    result = validate_config(cfg, experiment_type)
    if not _json_output(ctx, result):
        skin = _get_skin()
        if result["valid"]:
            skin.success("Configuration is valid")
        else:
            skin.error("Configuration has errors:")
            for e in result["errors"]:
                skin.error(f"  {e}")
        for w in result.get("warnings", []):
            skin.warning(f"  {w}")


@config.command("diff")
@click.argument("config_a", type=click.Path(exists=True))
@click.argument("config_b", type=click.Path(exists=True))
@click.pass_context
def config_diff(ctx, config_a, config_b):
    """Compare two configuration files."""
    from cli_anything.lfiax.core.config import diff_configs
    result = diff_configs(config_a, config_b)
    if not _json_output(ctx, result):
        skin = _get_skin()
        diffs = result.get("differences", {})
        a_only = result.get("a_only", [])
        b_only = result.get("b_only", [])
        if not diffs and not a_only and not b_only:
            skin.success("Configurations are identical")
        else:
            if diffs:
                skin.section("Value Differences")
                for key, vals in diffs.items():
                    skin.status(key, f"{vals.get('a', '<missing>')} -> {vals.get('b', '<missing>')}")
            if a_only:
                skin.section("Only in A")
                for key in a_only:
                    skin.info(key)
            if b_only:
                skin.section("Only in B")
                for key in b_only:
                    skin.info(key)


# ── results group ────────────────────────────────────────────────────

@cli.group()
@click.pass_context
def results(ctx):
    """Inspect experiment results and checkpoints."""
    pass


@results.command("list")
@click.option("--output-dir", default="./outputs", help="Base output directory")
@click.pass_context
def results_list(ctx, output_dir):
    """List available experiment results."""
    from cli_anything.lfiax.core.results import find_runs
    runs = find_runs(output_dir)
    if not _json_output(ctx, runs):
        skin = _get_skin()
        if not runs:
            skin.info("No results found")
            return
        headers = ["Run", "Experiment", "Checkpoints", "Status"]
        rows = [
            [r.get("name", "?"), r.get("experiment_type", "?"),
             str(r.get("num_checkpoints", 0)), r.get("status", "?")]
            for r in runs
        ]
        skin.table(headers, rows)


@results.command("show")
@click.argument("run_dir", type=click.Path(exists=True))
@click.pass_context
def results_show(ctx, run_dir):
    """Show detailed results for a run."""
    from cli_anything.lfiax.core.results import get_run_summary
    result = get_run_summary(run_dir)
    if not _json_output(ctx, result):
        skin = _get_skin()
        skin.status_block(result)


@results.command("checkpoint")
@click.argument("checkpoint_path", type=click.Path(exists=True))
@click.pass_context
def results_checkpoint(ctx, checkpoint_path):
    """Inspect a checkpoint file."""
    from cli_anything.lfiax.core.results import load_checkpoint
    result = load_checkpoint(checkpoint_path)
    if not _json_output(ctx, result):
        skin = _get_skin()
        skin.section(f"Checkpoint: {os.path.basename(checkpoint_path)}")
        skin.status("Size", f"{result.get('size_bytes', 0):,} bytes")
        skin.status("Keys", str(result.get("num_keys", 0)))
        for key, info in result.get("contents", {}).items():
            details = ", ".join(f"{k}={v}" for k, v in info.items())
            skin.status(f"  {key}", details)


# ── simulator group ──────────────────────────────────────────────────

@cli.group()
@click.pass_context
def simulator(ctx):
    """Inspect and run simulators."""
    pass


@simulator.command("list")
@click.pass_context
def simulator_list(ctx):
    """List available simulators."""
    from cli_anything.lfiax.core.simulator import list_simulators
    sims = list_simulators()
    if not _json_output(ctx, sims):
        skin = _get_skin()
        headers = ["Name", "Parameters", "Description"]
        rows = [[s["name"], str(s["num_params"]), s["description"]] for s in sims]
        skin.table(headers, rows)


@simulator.command("info")
@click.argument("name")
@click.pass_context
def simulator_info(ctx, name):
    """Show detailed info about a simulator."""
    from cli_anything.lfiax.core.simulator import get_simulator_info
    try:
        info = get_simulator_info(name)
        if not _json_output(ctx, info):
            skin = _get_skin()
            skin.section(info.get("full_name", name))
            skin.status("Description", info.get("description", ""))
            skin.status("Parameters", str(info.get("num_params", 0)))
            skin.status("Design space", info.get("design_space", ""))
            skin.status("Output dim", str(info.get("output_dim", 0)))
            if info.get("parameters"):
                skin.section("Parameters")
                for p in info["parameters"]:
                    skin.status(
                        f"  {p['name']}",
                        f"{p['description']} [{p['range'][0]}, {p['range'][1]}]"
                    )
    except ValueError as e:
        if not _json_output(ctx, {"error": str(e)}):
            skin = _get_skin()
            skin.error(str(e))


# ── oed group (generic BOED optimizer) ───────────────────────────────

@cli.group()
@click.pass_context
def oed(ctx):
    """Generic BOED optimizer — run LF-PCE on user-defined problems."""
    pass


@oed.command("describe")
@click.pass_context
def oed_describe(ctx):
    """Show the lfiax OED backend capabilities."""
    from cli_anything.lfiax.core.oed import describe_backend
    info = describe_backend()
    if not _json_output(ctx, info):
        skin = _get_skin()
        skin.section(info["name"])
        skin.status("Description", info["description"])
        skin.status("Status", info["status"])
        skin.section("Capabilities")
        for k, v in info["capabilities"].items():
            skin.status(f"  {k}", str(v))
        skin.section("Required fields")
        for f in info["required_fields"]:
            skin.info(f"  {f}")


@oed.command("validate")
@click.argument("spec_path", type=click.Path(exists=True))
@click.pass_context
def oed_validate(ctx, spec_path):
    """Validate an ExperimentSpec JSON against the lfiax backend."""
    from cli_anything.lfiax.core.spec import load_spec, validate_spec
    spec = load_spec(spec_path)
    report = validate_spec(spec)
    if not _json_output(ctx, report):
        skin = _get_skin()
        if report["valid"]:
            skin.success("Spec is valid")
        else:
            skin.error("Spec has errors:")
            for e in report["errors"]:
                skin.error(f"  {e['path']}: {e['message']}")
        for w in report.get("warnings", []):
            skin.warning(f"  {w['path']}: {w['message']}")


@oed.command("optimize")
@click.argument("spec_path", type=click.Path(exists=True))
@click.option("--seed", type=int, default=0, help="RNG seed")
@click.option("--dry-run", is_flag=True, help="Return the plan without running JAX")
@click.option("--no-artifacts", is_flag=True,
              help="Do not write result.json/spec.json to the artifacts dir")
@click.pass_context
def oed_optimize(ctx, spec_path, seed, dry_run, no_artifacts):
    """Run generic BOED optimization (LF-PCE) against a user spec."""
    from cli_anything.lfiax.core.spec import load_spec
    from cli_anything.lfiax.core.oed import optimize_design
    spec = load_spec(spec_path)
    result = optimize_design(
        spec, seed=seed, dry_run=dry_run, write_artifacts=not no_artifacts,
    )
    if not _json_output(ctx, result):
        skin = _get_skin()
        status = result.get("status", "?")
        if status == "dry_run":
            skin.info("Dry run — optimization plan:")
            skin.status("Backend", result.get("backend", "?"))
            skin.status("Estimator", result.get("estimator", "?"))
            skin.status("Initial design", str(result.get("initial_design")))
            skin.status("Bounds", str(result.get("design_bounds")))
        elif status == "completed":
            skin.success("Optimization complete")
            skin.status("Final design", str(result.get("design")))
            skin.status("Final EIG", f"{result.get('eig'):.4f}" if result.get("eig") is not None else "n/a")
            if result.get("artifacts", {}).get("run_dir"):
                skin.status("Artifacts", result["artifacts"]["run_dir"])
        elif status == "invalid_spec":
            skin.error("Spec is invalid:")
            for e in result.get("validation", {}).get("errors", []):
                skin.error(f"  {e['path']}: {e['message']}")
        else:
            skin.error(result.get("error", f"Status: {status}"))


@oed.command("init")
@click.argument("out_dir", type=click.Path())
@click.pass_context
def oed_init(ctx, out_dir):
    """Scaffold an example problem.py + spec.json in OUT_DIR."""
    import shutil
    src_dir = os.path.join(os.path.dirname(__file__), "examples")
    os.makedirs(out_dir, exist_ok=True)
    files = []
    for name in ("linear_problem.py", "linear_spec.json"):
        src = os.path.join(src_dir, name)
        dst = os.path.join(out_dir, name)
        shutil.copyfile(src, dst)
        files.append(dst)
    result = {"status": "ok", "files": files, "out_dir": out_dir}
    if not _json_output(ctx, result):
        skin = _get_skin()
        skin.success(f"Scaffolded example in {out_dir}")
        for f in files:
            skin.info(f"  {f}")


# ── env command ──────────────────────────────────────────────────────

@cli.command("env")
@click.pass_context
def env_cmd(ctx):
    """Show environment and dependency info."""
    from cli_anything.lfiax.utils.lfiax_backend import get_environment_info
    info = get_environment_info()
    if not _json_output(ctx, info):
        skin = _get_skin()
        skin.section("Environment")
        skin.status("Python", info["python"]["version"].split()[0])
        skin.status("Platform", info["platform"])
        lf = info["lfiax"]
        if lf["installed"]:
            skin.success(f"lfiax {lf['version']} installed at {lf['path']}")
        else:
            skin.error("lfiax not installed - run: pip install -e . from lfiax root")
        jx = info["jax"]
        if jx["available"]:
            skin.success(f"JAX {jx['version']} ({jx['default_backend']})")
            for d in jx["devices"]:
                skin.status(f"  Device", f"{d['platform']}:{d['id']}")
        else:
            skin.error("JAX not available")


# ── REPL command ─────────────────────────────────────────────────────

@cli.command("repl", hidden=True)
@click.pass_context
def repl(ctx):
    """Interactive REPL mode."""
    skin = _get_skin()
    skin.print_banner()

    pt_session = skin.create_prompt_session()

    commands = {
        "experiment run <type>": "Run a BOED experiment",
        "experiment list": "List experiment runs",
        "experiment status <dir>": "Check run status",
        "config show <path>": "Display config file",
        "config validate <path>": "Validate config",
        "config diff <a> <b>": "Compare configs",
        "results list": "List experiment results",
        "results show <dir>": "Show run results",
        "results checkpoint <path>": "Inspect checkpoint",
        "simulator list": "List simulators",
        "simulator info <name>": "Simulator details",
        "oed describe": "Show OED backend capabilities",
        "oed validate <spec>": "Validate an ExperimentSpec JSON",
        "oed optimize <spec>": "Run generic BOED on a user spec",
        "oed init <dir>": "Scaffold example problem + spec",
        "env": "Show environment info",
        "help": "Show this help",
        "quit": "Exit REPL",
    }

    project_name = ""
    if ctx.obj.get("project"):
        project_name = os.path.basename(ctx.obj["project"])

    while True:
        try:
            line = skin.get_input(pt_session, project_name=project_name)
            if not line:
                continue
            if line.lower() in ("quit", "exit", "q"):
                skin.print_goodbye()
                break
            if line.lower() == "help":
                skin.help(commands)
                continue

            # Parse and dispatch to Click commands
            args = line.split()
            if ctx.obj.get("json_mode"):
                args = ["--json"] + args
            if ctx.obj.get("project"):
                args = ["--project", ctx.obj["project"]] + args
            try:
                cli.main(args, standalone_mode=False, parent=ctx.parent)
            except click.exceptions.UsageError as e:
                skin.error(str(e))
            except SystemExit:
                pass
            except Exception as e:
                skin.error(f"Error: {e}")
        except (KeyboardInterrupt, EOFError):
            skin.print_goodbye()
            break


if __name__ == "__main__":
    cli()

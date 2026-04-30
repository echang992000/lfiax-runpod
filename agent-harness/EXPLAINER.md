# cli-anything-lfiax: How It Works

A walkthrough of the CLI harness architecture for future reference.

## The Big Picture

Think of it as a **unified frontend** to your scattered experiment scripts. Instead of remembering `python BMP.py experiment.design_rounds=5 ...` vs `python sir.py ...` vs checkpoint file paths vs YAML layouts, you have one command: `cli-anything-lfiax`.

Under the hood it's **Click** (a Python CLI framework) + **subprocess calls** to your existing scripts + **direct imports** of lfiax utilities.

---

## Layer 1: The Entry Point

`setup.py` registers `cli-anything-lfiax` as a console script pointing at `lfiax_cli.py:cli`. When you type `cli-anything-lfiax`, the shell runs that function.

The `cli` function is a Click **group** — a container for subcommands. The key trick:

```python
@click.group(invoke_without_command=True)
def cli(ctx, json_mode, project):
    if ctx.invoked_subcommand is None:
        ctx.invoke(repl)   # ← no args? drop into REPL
```

So `cli-anything-lfiax` (no args) → REPL, but `cli-anything-lfiax experiment run bmp` → one-shot command.

---

## Layer 2: Command Groups

There are 5 command groups, each mapping to a **logical domain** of lfiax:

| Group | What it wraps | Backend call |
|-------|---------------|--------------|
| `experiment` | BMP.py, sir.py, two_moons.py | `subprocess.run([python, script, ...hydra_overrides])` |
| `config` | YAML config files | `yaml.safe_load()` + dict comparison |
| `results` | outputs/ dir + .pkl checkpoints | `os.walk` + `pickle.load` |
| `simulator` | lfiax.utils.simulators | Static metadata dict (+ optional JAX sampling) |
| `env` | Installation diagnostics | `import jax; import lfiax` |

Each command is ~10 lines: parse args → call a core function → print result.

---

## Layer 3: The `--json` Flag Pattern

Every command ends with this idiom:

```python
def experiment_list(ctx, output_dir):
    runs = list_runs(output_dir)              # pure function, returns dict/list
    if not _json_output(ctx, runs):           # if --json, print JSON and return True
        skin = _get_skin()
        skin.table(headers, rows)             # otherwise, pretty-print with colors
```

**Why this matters:** AI agents (or scripts) run `cli-anything-lfiax --json ...` and get parseable JSON. Humans run it without `--json` and get colored tables. Same codepath, same data.

---

## Layer 4: How `experiment run` Actually Works

This is the most interesting one. When you run:

```bash
cli-anything-lfiax experiment run bmp --seed 42 -o experiment.design_rounds=3
```

Here's the flow in `core/experiment.py`:

1. **Find the lfiax repo root** — walks up from cwd looking for `src/lfiax/`, or uses `$LFIAX_ROOT`. This lets the CLI work from any directory.
2. **Look up the script** — `EXPERIMENT_SCRIPTS["bmp"] = "BMP.py"`
3. **Build the Hydra command line:**
   ```
   [python, /path/to/lfiax/BMP.py, experiment.seed=42, experiment.design_rounds=3]
   ```
4. **Hand it to `subprocess.run`** with `cwd=lfiax_root` so Hydra finds its configs.
5. **Capture stdout/stderr** and return a structured dict.

The CLI doesn't reimplement your training loop — it just **constructs the right `python BMP.py ...` invocation** for you and runs it. Hydra still does all the config magic; you're just saved from typing the full path and remembering override syntax.

`--dry-run` returns the command without executing — useful for agents to preview actions.

---

## Layer 5: State and Session

`core/session.py` keeps a JSON file at `~/.cli-anything-lfiax/session.json` with:

- `active_project` — which project JSON you're working on
- `history` — last 100 commands
- Timestamps

It uses **fcntl file locking** (`flock`) so two REPLs can't corrupt the file. You mostly don't interact with this directly; it's used by the REPL to remember context between commands.

---

## Layer 6: The REPL

When you run `cli-anything-lfiax` with no args, `ctx.invoke(repl)` kicks off an interactive loop:

```python
while True:
    line = skin.get_input(pt_session, project_name=project_name)
    args = line.split()
    cli.main(args, standalone_mode=False, parent=ctx.parent)
```

The clever part: the REPL **re-invokes the same Click group** with the user's typed args. So `experiment list` inside the REPL goes through the exact same code path as `cli-anything-lfiax experiment list` at the shell. Zero duplication.

`prompt_toolkit` gives you history (up arrow), search (Ctrl-R), and persistent history across sessions. `ReplSkin` wraps all the ANSI color codes, box-drawing, and the startup banner.

---

## Layer 7: Backend Abstraction

`utils/lfiax_backend.py` is where the "is the real software installed?" logic lives:

```python
def check_lfiax_installed():
    try:
        import lfiax
        return {"installed": True, "version": ..., "path": ...}
    except ImportError:
        return {"installed": False, ...}
```

Same pattern for JAX. This is what powers `cli-anything-lfiax env` and lets the CLI give clear "install this first" errors instead of mysterious crashes.

---

## Key Design Choices

1. **Subprocess, not import, for experiments.** Your scripts are 600-1000 lines of Hydra + JAX + wandb. Reimplementing them in-process would be fragile. Calling them as subprocesses preserves all their existing behavior.

2. **Import, not subprocess, for metadata.** For `simulator list`, config parsing, checkpoint inspection — these are quick lookups where importing Python modules is cleaner.

3. **Deferred imports.** Heavy imports (`jax`, `numpy`, `lfiax`) are inside function bodies, not at module top. This means `cli-anything-lfiax --help` starts instantly even without JAX installed.

4. **Namespace package.** `cli_anything/` has no `__init__.py`. This is PEP 420 — it lets `cli-anything-lfiax`, `cli-anything-blender`, etc. all live under `cli_anything.*` without conflict.

5. **Pure functions return dicts.** Every core function returns a JSON-serializable dict. The CLI layer is just glue between user input, these functions, and either JSON or pretty-printed output.

---

## A Concrete Trace

```bash
cli-anything-lfiax --json config validate config_bmp.yaml --type bmp
```

1. Shell runs `/anaconda3/bin/cli-anything-lfiax` (the entry point).
2. Click parses `--json` → sets `ctx.obj["json_mode"] = True`.
3. Click routes to `config_validate` command.
4. `config_validate` calls `load_config(path)` → `yaml.safe_load()` returns a dict.
5. Calls `validate_config(cfg, "bmp")` → walks the schema, returns `{valid, errors, warnings}`.
6. `_json_output(ctx, result)` sees `json_mode=True`, prints `json.dumps(result)`, returns True.
7. Function exits. Exit code 0.

That's it. ~15 lines of CLI glue wrapping ~30 lines of core logic.

---

## File Map Reference

| File | Role |
|------|------|
| `lfiax_cli.py` | Click CLI definition, all command groups, REPL loop |
| `core/experiment.py` | Build/run Hydra subprocess commands |
| `core/config.py` | YAML load/validate/diff |
| `core/project.py` | Project JSON state (create/load/save) |
| `core/results.py` | Scan outputs/, inspect .pkl checkpoints |
| `core/simulator.py` | Simulator metadata registry |
| `core/session.py` | REPL session state + fcntl locking |
| `utils/lfiax_backend.py` | lfiax/JAX installation checks |
| `utils/repl_skin.py` | Terminal styling, banner, prompts |
| `setup.py` | Namespace package + console_scripts entry |
| `skills/SKILL.md` | AI-agent discoverable capability doc |

---

## Zoom-In Points

Places where it's easy to get lost:

- **Click command decorators** — how `@cli.group()`, `@experiment.command()`, and `@click.pass_context` chain together
- **Hydra override building** — how the `-o key=value` flags become positional args to the subprocess
- **REPL dispatch** — how `cli.main(args, standalone_mode=False)` re-enters Click without exiting
- **Namespace package mechanics** — why omitting `cli_anything/__init__.py` is load-bearing

# AGENTS.md

Guidance for agentic coding tools operating in this repository.

## Project Scope

- This is a Python/JAX research codebase for offline goal-conditioned RL.
- Main entry points are `main.py`, `eval_pretrained.py`, and `rliable_agents.py`.
- Dependency management uses `uv` with `pyproject.toml` and `uv.lock`.
- Python target is `>=3.11`.
- Core packages include JAX, Flax, Optax, ml-collections, Orbax, and W&B.
- Experiment launch commands live in `hyperparameters.sh`.

## Rules Files

- Repository-local Cursor rules: none found in `.cursor/rules/`.
- Repository-local `.cursorrules`: none found.
- Repository-local Copilot rules: none found in `.github/copilot-instructions.md`.
- If any of those files are added later, treat them as higher-priority repository guidance.

## Repository Layout

- `agents/`: algorithm implementations and agent registries.
- `agents/__init__.py`: central registry mapping agent names to classes.
- `agents/*/*.py`: agent variants typically expose a class plus `get_config()`.
- `utils/`: datasets, networks, env creation, eval helpers, logging, checkpoint helpers.
- `main.py`: offline training loop, logging, checkpointing, and evaluation.
- `eval_pretrained.py`: checkpoint evaluation and RLiable-style reporting.
- `hyperparameters.sh`: canonical paper and baseline launch commands.
- `results/`, `eval_results/`, and similar output folders may contain generated artifacts; do not edit them unless asked.

## Environment Setup

- Sync dependencies: `uv sync`
- Run commands in the project environment: `uv run <command>`
- Optional shell activation: `source .venv/bin/activate`
- Quick import smoke test: `uv run python -c "import agents, utils; print('ok')"`

## Build And Validation Commands

- There is no package build pipeline, wheel, or Makefile.
- Treat these as build-equivalent checks:
- Bytecode compilation: `uv run python -m compileall .`
- Training CLI health check: `uv run python main.py --help`
- Eval CLI health check: `uv run python eval_pretrained.py --help`
- Agent registry import smoke test: `uv run python -c "from agents import agents; print(sorted(agents)[:3])"`

## Lint And Format Commands

- Format all files: `uv run ruff format .`
- Check formatting only: `uv run ruff format . --check`
- Lint all files: `uv run ruff check .`
- Auto-fix lint issues when safe: `uv run ruff check . --fix`
- Preferred local validation after edits: `uv run ruff format . && uv run ruff check .`

## Test Commands

- There is currently no committed `tests/` directory or `pytest.ini`.
- If tests are added, use `pytest` through `uv run`.
- Run all tests: `uv run pytest`
- Run a single test file: `uv run pytest path/to/test_file.py`
- Run a single test function: `uv run pytest path/to/test_file.py::test_name`
- Run a single test method: `uv run pytest path/to/test_file.py::TestClass::test_name`
- Run tests by keyword: `uv run pytest -k "keyword"`
- When adding a new test, verify at least one targeted test invocation before finishing.

## Practical Smoke Checks

- Because formal tests are absent, always run at least one lightweight smoke command relevant to the change.
- Minimal train dry-run:
- `uv run python main.py --env_name=pointmaze-medium-navigate-v0 --train_steps=5 --log_interval=5 --eval_interval=5 --save_interval=1000000 --eval_episodes=1 --video_episodes=0 --wandb_mode=offline`
- Minimal eval dry-run with an existing checkpoint:
- `uv run python eval_pretrained.py --env_name=pointmaze-medium-navigate-v0 --agent=agents/crl/dual.py --restore_path=<checkpoints_dir> --restore_epoch=<step> --eval_episodes=2 --num_bootstraps=100`
- RLiable comparison utility:
- `uv run python rliable_agents.py --results_dirs <dir1> <dir2> --labels "A" "B" --output_dir comparison_results/<name>`

## Experiment Commands

- For baseline or paper reproduction runs, start from `hyperparameters.sh` rather than inventing new defaults.
- For new methods, choose defaults that are consistent with nearby agents and environment families.
- Avoid launching full training runs unless explicitly requested; default to smoke-scale checks.

## Coding Conventions

### Imports

- Use three import groups: standard library, third-party, local project.
- Separate groups with one blank line.
- Prefer one import per module or a short grouped import line when readable.
- Avoid duplicate imports and mixed ordering.
- Prefer explicit imports over wildcard imports.

### Formatting

- Use `ruff format` as the canonical formatter.
- Keep lines readable; split long calls and literals across lines.
- Use trailing commas in multi-line literals and function calls.
- Keep docstrings concise and focused on behavior or shape assumptions.
- Add comments only for non-obvious math, JAX behavior, or tricky dataset semantics.

### Types And Signatures

- Match the repository's lightweight typing style.
- Use `Any` when Flax/JAX pytrees make precise typing noisy.
- Add type hints to new public helpers where practical.
- Preserve established public method names and signatures such as `create`, `update`, `sample_actions`, and `get_config`.
- Keep `get_config()` returning `ml_collections.ConfigDict` with explicit keys.

### Naming

- Filenames and modules: `snake_case.py`
- Classes: `PascalCase`
- Agent classes should use the `Agent` suffix.
- Functions, variables, and config keys: `snake_case`
- True constants: `UPPER_CASE`
- `absl.flags` names should remain lower snake case.
- Log keys should keep slash namespaces such as `actor/loss` or `value/v_mean`.

### JAX And Flax Patterns

- Keep hot paths JIT-friendly.
- Use `@jax.jit` on performance-critical update and loss functions when consistent with neighboring code.
- Keep RNG handling explicit via `jax.random.split`.
- Prefer vectorized operations like `vmap`, `einsum`, and array ops over Python loops in model math.
- Use `jax.lax.stop_gradient` only when intentional; document why if the reason is not obvious.
- Avoid Python-side mutation inside jitted regions except for established train-state patterns.

### Configuration And Agents

- New agent variants should follow the existing structure in `agents/crl`, `agents/gcfbc`, or `agents/gcivl`.
- Register any new agent in `agents/__init__.py`.
- Add new hyperparameters to `get_config()` with sensible defaults.
- Preserve compatibility keys in configs unless a coordinated cleanup is requested.
- Keep dataset goal-sampling probabilities normalized to sum to `1.0`.

### Data, Evaluation, And Reproducibility

- Preserve seed plumbing across Python `random`, NumPy, and JAX.
- Do not silently change save directory structure, experiment naming, or checkpoint conventions.
- Keep evaluation outputs compatible with downstream RLiable analysis.
- Preserve stable metric names when possible so previous result-processing scripts continue to work.

### Error Handling And Validation

- Fail fast on invalid configuration or unsupported modes.
- Prefer `ValueError` or clear assertions for impossible states and shape assumptions.
- Include the offending key, path, shape, or value in error messages when practical.
- Keep CLI interruption behavior graceful when touching long-running scripts.
- Validate boundary assumptions at dataset, action-space, and representation interfaces.

## Change Strategy For Agents

- Prefer minimal, targeted diffs over broad refactors.
- Preserve existing research behavior unless the task explicitly changes algorithm logic.
- If you modify algorithmic behavior, note the rationale in your final summary.
- Do not rewrite generated artifacts, checkpoints, W&B files, or plot outputs unless asked.
- If you add tests, also mention the exact single-test command that should be used to rerun them.

## Recommended Completion Checklist

- Sync dependencies if needed: `uv sync`
- Format and lint: `uv run ruff format . && uv run ruff check .`
- Run one relevant smoke command for the touched path.
- If tests exist for the touched code, run at least one targeted `pytest` invocation.
- State any assumptions if full training or full evaluation was not run.

# AGENTS.md

Guidance for autonomous coding agents working in this repository.

## Scope and Ground Truth

- This project is a Python/JAX research codebase for offline goal-conditioned RL.
- Primary entry points are `main.py`, `eval_pretrained.py`, and `rliable_agents.py`.
- Dependency management uses `uv` (`pyproject.toml` + `uv.lock`).
- There are currently no committed `tests/` files in the repository.
- No Cursor rules were found in `.cursor/rules/` or `.cursorrules`.
- No Copilot rules were found in `.github/copilot-instructions.md`.

## Environment Setup Commands

- Install/sync deps: `uv sync`
- Run with project env: `uv run <command>`
- Python version target: `>=3.11` (see `pyproject.toml`).
- Quick import smoke test: `uv run python -c "import agents, utils; print('ok')"`

## Build Commands

- There is no package build pipeline (no wheel/sdist/Makefile configured).
- Treat these as build-equivalent checks:
- Bytecode compile all Python: `uv run python -m compileall .`
- CLI health check (train script flags): `uv run python main.py --help`
- CLI health check (eval script flags): `uv run python eval_pretrained.py --help`

## Lint and Format Commands

- Ruff cache is present, so Ruff is the expected linter.
- Lint all files: `uv run ruff check .`
- Auto-fix lint issues: `uv run ruff check . --fix`
- Format all files: `uv run ruff format .`
- Check formatting only: `uv run ruff format . --check`

## Test Commands

- There is no formal test suite checked in right now.
- Default full test command (once tests exist): `uv run pytest`
- Run a single test file: `uv run pytest path/to/test_file.py`
- Run a single test case: `uv run pytest path/to/test_file.py::test_name`
- Run a single test method: `uv run pytest path/to/test_file.py::TestClass::test_name`
- Filter tests by keyword: `uv run pytest -k "keyword"`

## Practical Validation Commands (use today)

- Because tests are absent, validate changes with lightweight smoke runs.
- Minimal train dry-run (very short):
- `uv run python main.py --env_name=pointmaze-medium-navigate-v0 --train_steps=5 --log_interval=5 --eval_interval=5 --save_interval=1000000 --eval_episodes=1 --video_episodes=0 --wandb_mode=offline`
- Minimal eval dry-run (requires checkpoint):
- `uv run python eval_pretrained.py --env_name=pointmaze-medium-navigate-v0 --agent=agents/crl/dual.py --restore_path=<checkpoints_dir> --restore_epoch=<step> --eval_episodes=2 --num_bootstraps=100`
- RLiable comparison utility:
- `uv run python rliable_agents.py --results_dirs <dir1> <dir2> --labels "A" "B" --output_dir comparison_results/<name>`

## Full experiment run
- Take optimal hyperparameters and launch arguments from `hyperparameters.sh` file for specific baseline.
- If you are running/implementing not a baseline method, choose best optimal hyperparameters that are suitable.

## Repository Structure to Preserve

- `agents/`: algorithm implementations and config (`get_config`) per agent.
- `utils/`: datasets, env creation, networks, evaluation, logging, checkpoint helpers.
- `main.py`: training orchestration, eval loop, checkpointing, W&B logging.
- `eval_pretrained.py`: checkpoint evaluation + RLiable metrics/plots.
- `hyperparameters.sh`: canonical experiment command matrix.

## Code Style Guidelines

### Imports

- Order imports in three groups: stdlib, third-party, local project.
- Separate groups with one blank line.
- Avoid duplicate imports (e.g., same module imported twice).
- Prefer explicit imports over wildcard imports.

### Formatting

- Use `ruff format` style as canonical formatter.
- Keep lines readable; prefer multi-line call formatting over dense one-liners.
- Use trailing commas in multi-line literals/calls to reduce noisy diffs.
- Keep comments focused on non-obvious math or JAX-specific behavior.

### Types and Signatures

- Follow existing pattern: lightweight typing, `Any` where Flax/JAX pytrees are complex.
- Add type hints for new public functions when practical.
- Preserve existing public method signatures (`create`, `update`, `sample_actions`, `get_config`).
- Any new agent implementation must follow template from `agents` implementations.
- Keep `get_config()` returning `ml_collections.ConfigDict` with explicit keys.

### Naming Conventions

- Filenames/modules: `snake_case.py`.
- Classes: `PascalCase` with `Agent` suffix for agents.
- Functions/variables/config keys: `snake_case`.
- Constants/flags: UPPER_CASE only for true constants; `absl.flags` use lower snake case.
- Log keys use slash namespaces like `actor/loss`, `value/v_mean`.

### JAX/Flax Conventions

- Keep hot paths JIT-friendly (`@jax.jit` on `update`, `total_loss`, action sampling).
- Avoid Python-side mutation inside jitted code except established patterns.
- Keep RNG handling explicit (`new_rng, rng = jax.random.split(...)`).
- Use `jax.lax.stop_gradient` intentionally and document why when added.
- Prefer vectorized ops (`vmap`, `einsum`) over Python loops in model math.

### Config and Hyperparameters

- Add new hyperparameters in `get_config()` with sensible defaults and comments.
- Keep dataset sampling probabilities normalized (sum to `1.0`).
- Respect existing config compatibility keys even if currently unused.
- When adding agent variants, register them in `agents/__init__.py`.

### Error Handling and Validation

- Fail fast on invalid configuration with `ValueError`/`assert` (consistent with codebase).
- Validate shape assumptions at boundaries (dataset batch, action dims, goal reps).
- Keep user-facing CLI interruptions graceful (`KeyboardInterrupt` handling pattern).
- Prefer clear exception messages including offending key/path/value.

### Logging, Checkpointing, and Reproducibility

- Keep training/eval metrics both in W&B and CSV/JSON outputs where applicable.
- Use existing Orbax helpers in `utils/flax_utils.py` for save/restore.
- Preserve seed plumbing (`random`, `numpy`, JAX RNG).
- Do not silently change default experiment naming or save directory patterns.

### Evaluation and Metrics

- Maintain RLiable-compatible outputs (`score_matrix.npy`, aggregate JSON, plots).
- Keep per-task metric naming stable for downstream comparison scripts.
- Use small bootstrap counts only for quick smoke checks; keep large defaults for final runs.

## Change Management for Agents

- Prefer minimal, targeted diffs; avoid broad refactors unless requested.
- If touching algorithm logic, include a short rationale in PR/commit notes.
- Run lint/format and at least one smoke command after edits.
- If adding tests, also document exact single-test invocation in this file.
- Never commit large generated artifacts (plots, checkpoints, W&B run files) unless asked.

## Known Gaps / Caveats

- The repository currently has no first-class unit tests; rely on smoke validation.
- Some scripts are compute-heavy by default; always downscale steps/episodes for CI-like checks.

## Quick Agent Checklist
- Ensure you have loaded uv venv as `source .venv/bin/activate`
- Sync deps: `uv sync`
- Format + lint: `uv run ruff format . && uv run ruff check .`
- Run smoke command relevant to your change.
- For test additions, verify one targeted case with `uv run pytest path::test_name`.
- Summarize any assumptions when full training/eval was not executed.

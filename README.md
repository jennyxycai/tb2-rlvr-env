# tb2-rlvr-env

RLVR environment for [terminal-bench-2](https://github.com/harbor-framework/terminal-bench-2) built on the [Harbor](https://www.harborframework.com/) framework. The env wraps tb-format tasks, runs them via Harbor's `terminus-2` harness, and computes an 85/15 reward (pytest-based test verification + LLM-as-judge on the trajectory). Intended to be plugged into an RL trainer (SkyRL, verifiers, etc.) via `HarborRolloutInterface`.


## Install

Python 3.12+ required (Harbor's minimum).

```bash
git clone <this-repo>
cd tb2-rlvr-env

python3.12 -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"
```

This pulls Harbor from source (not on PyPI yet), plus our deps (litellm, anthropic, scipy, pyyaml, pydantic, omegaconf, python-dotenv).

## Configure

```bash
cp .env.example .env  # then fill in
```

Required:
- `ANTHROPIC_API_KEY` — for the LLM-as-judge (Claude Sonnet)
- A reachable Docker daemon if you're using `sandbox_provider: docker` (the default, set in [configs/env_config.yaml](configs/env_config.yaml)).


## Common commands

```bash
pytest -m "not slow"          # fast unit tests (~2s)
pytest -m slow                # judge fixtures (~12s, ~$0.30) + smoke (~2 min on Docker)
ruff check src tests          # lint
ruff format src tests         # format

# Stage 5 (inert until JudgeValidator is implemented):
tb2-validate-judge --config configs/judge_config.yaml
```

## Smoke test

```bash
pytest tests/test_smoke.py -v -m slow
```

Runs one tb2 task end-to-end on local Docker (default: `fix-git` with `claude-haiku-4-5`). Costs ~$0.05–0.50 and 1–2 min wall-clock. Override via env vars:

```bash
TB2_SMOKE_TASK=<task-name> TB2_SMOKE_MODEL=<model-id> pytest tests/test_smoke.py -v -m slow
```

## Reward function

Each rollout produces a reward in `[0, 1]`:

- **0.85 × test verification**: pytest from the task's `tests/test.sh` runs against the agent's final container state after `complete()`. Per-test weights are applied if the task specifies them; otherwise uniform `1/N`.
- **0.15 × LLM-as-judge**: Claude Sonnet 4.6 scores the trajectory on 4 dimensions (action output success, planning quality, phase adherence, tool effectiveness); the 4 are averaged.

Combined via `GroupRewards`: weighted sum, then group-demean across the 16 rollouts per prompt to produce the GRPO advantage.

## Where the trainer plugs in

`src/env/rollout.py::HarborRolloutInterface` is the contract called by the RL trainer (SkyRL or `verifiers`). The trainer lives in a separate repo; this env plugs in by implementing that interface. On init it lazily resolves task pools via `DatasetResolver`; per-rollout it builds a `harbor.TrialConfig`, runs `Trial.create(config).run()` (async, wrapped in `asyncio.run`), extracts trajectory + verifier output from the trial dir, dispatches reward fns, returns `RolloutResult`.
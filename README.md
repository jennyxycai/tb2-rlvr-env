# tb2-rlvr-env

RLVR environment for [terminal-bench-2](https://github.com/harbor-framework/terminal-bench-2) built on the [Harbor](https://www.harborframework.com/) framework. The env wraps tb-format tasks, runs them via a Harbor-managed agent harness (Terminus), and computes a 85/15 reward (pytest-based test verification + LLM-as-judge on the trajectory). Intended to be plugged into an RL trainer (SkyRL, verifiers, etc.) via `HarborRolloutInterface`.

## Install

Python 3.12+ required (Harbor's minimum).

```bash
git clone <this-repo>
cd tb2-rlvr-env

python3.12 -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"
```

This pulls Harbor from source (it's not on PyPI yet), plus our deps (anthropic, scipy, pyyaml, pydantic, omegaconf, python-dotenv).

## Configure

Copy `.env.example` to `.env` and fill in:

```bash
cp .env.example .env
```

Required:
- `ANTHROPIC_API_KEY` — for the LLM-as-judge (Claude Sonnet)
- `HARBOR_SANDBOX_PROVIDER` — one of `modal`, `daytona`, `e2b`, `runloop`, `tensorlake`
- Provider-specific credentials (e.g., `MODAL_TOKEN_ID` + `MODAL_TOKEN_SECRET` for Modal)

## Smoke test

```bash
pytest tests/test_smoke.py -v -m smoke
```

Runs one tb2 task end-to-end against the base model, computes the 85/15 reward, prints the result. ~$1-5 in API costs, 1-2 minutes wall-clock.

## Reward function

Each rollout produces a reward in `[0, 1]`:

- **0.85 × test verification**: pytest from the task's `tests/test.sh` runs against the agent's final container state after `complete()`. Per-test weights are applied if the task specifies them; otherwise uniform `1/N`.
- **0.15 × LLM-as-judge**: Claude Sonnet 4 scores the trajectory on 4 dimensions (action output success, planning quality, phase adherence, tool effectiveness); the 4 are averaged.

Combined via the `GroupRewards` pattern from `gypsum/src/tasks/rewards.py`: weighted sum, then group-demean across the 16 rollouts per prompt to produce the GRPO advantage.

## Notes on tb2 task filesystem

The agent's working directory inside the container is whatever the task's `environment/Dockerfile` sets as `WORKDIR` (varies per task; e.g., `bn-fit-modify` uses `/app/`). The task instruction is delivered to the agent in the initial user message of the chat template, not as a file in the container. Harbor mounts `/logs/verifier/`, `/logs/agent/`, `/solution/`, and `/tests/` at trial time per its spec; the trained agent sees only the WORKDIR (the others are not in Terminus's tool catalog).

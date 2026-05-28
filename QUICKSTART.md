# Quickstart

## 1. Install

Requires Python >=3.12.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

`harbor` is installed from source from `github.com/laude-institute/harbor` (not yet on PyPI).

## 2. Configure secrets

```bash
cp .env.example .env  # then fill in keys
```

Required:
- `ANTHROPIC_API_KEY` — for the LLM-as-judge (Claude Sonnet)
- `HARBOR_SANDBOX_PROVIDER` — e.g. `modal`, plus its credentials

## 3. Repo layout

```
src/
  env/       rollout interface + trajectory compaction
  rewards/   test verifier, llm judge, group-rewards combine, judge calibration
  prompts/   agent system prompt + judge rubric (markdown, loaded at runtime)
configs/     env / reward / judge YAMLs
tests/       pytest unit + smoke tests, judge eval fixtures
```

## 4. Common commands

Run unit tests (skips the slow smoke test):

```bash
pytest -m "not slow"
```

Run Phase 0 judge calibration (Spearman corr on ~50 held-out tasks):

```bash
tb2-validate-judge --config configs/judge_config.yaml
```

Run the slow end-to-end smoke (one tb2 task, real container, real API):

```bash
pytest -m slow
```

## 5. Where the trainer plugs in

`src/env/rollout.py::HarborRolloutInterface` is the contract called by the RL trainer
(SkyRL or `verifiers`). The trainer lives in a separate repo; this env plugs in by
implementing that interface.

See [README.md](README.md) for the full design doc.

# Stage 5: Judge calibration (Phase 0b)

Before plugging the LLM judge into training as 15% of the reward, confirm it actually correlates with the test verifier. Otherwise we'd be optimizing against noise.

**Method**: ~50 held-out rollouts through `HarborRolloutInterface.rollout()` with the base model, score each with both `TestVerifierReward` and `LLMJudgeReward`, compute Spearman ρ between the two score lists.

**Decision**:

| ρ | Action |
|---|---|
| `≥ 0.7` | Proceed — judge is well-calibrated, plug into training |
| `0.5–0.7` | Iterate the rubric ([judge_rubric.md](src/prompts/judge_rubric.md)); add fixtures to [tests/judge_eval_cases/](tests/judge_eval_cases/) covering the disagreements |
| `< 0.5` | Swap judge model (Sonnet → Opus / Gemini 2.5 Pro) — don't start training |

Why Spearman: we care about *ranking*, not absolute magnitude. A judge that always outputs `[0.4, 0.6]` but ranks correctly is fine.

**Status**: not implemented. Scaffolds in [src/rewards/judge_validation.py](src/rewards/judge_validation.py) (`JudgeValidator`, `JudgeValidationCLI`) and [src/rewards/types.py](src/rewards/types.py) (`JudgeValidationReport`). The `tb2-validate-judge` console entry is wired but inert. Implementation is ~30 lines once picked up: the loop above + `scipy.stats.spearmanr` + a table lookup.

**Cost**: ~$5-10 in judge API calls + ~5-12h of local Docker time for the 50 rollouts (or much less on a remote sandbox). Iterate on the rubric via the cheap Phase 0a fixture loop ([`OfflineEvalCaseRunner`](src/rewards/judge_validation.py), ~$0.30/run) before burning live calibration calls.

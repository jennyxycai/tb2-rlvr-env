# Judge calibration: offline runner (Phase 0a, working) + scaffolds for the
# live Spearman calibration step (Phase 0b, blocked on the Harbor wire-up in Stage 4).

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.rewards.types import CaseResult, JudgeValidationReport


class OfflineEvalCaseRunner:
    """Runs the judge against pre-recorded trajectories in tests/judge_eval_cases/.

    Used before live calibration: verifies the rubric isn't broken by checking that
    each fixture's judge score falls within its declared `expected_range`.
    """

    def __init__(self, cases_dir: str, judge: Any) -> None:
        self.cases_dir = Path(cases_dir)
        self.judge = judge

    def run(self) -> list[CaseResult]:
        """Score every fixture in cases_dir and return per-case results."""
        results: list[CaseResult] = []
        for path in sorted(self.cases_dir.glob("*.json")):
            case = json.loads(path.read_text())
            name = case.get("name", path.stem)
            er = tuple(case.get("expected_range", [0.0, 1.0]))
            try:
                score = self.judge(case)
                dims = getattr(self.judge, "last_dimensions", None)
                in_range = er[0] <= score <= er[1]
                results.append(
                    CaseResult(
                        name=name,
                        expected_range=er,
                        actual_score=score,
                        dimensions=dims,
                        in_range=in_range,
                    )
                )
            except Exception as e:
                results.append(
                    CaseResult(
                        name=name,
                        expected_range=er,
                        actual_score=None,
                        dimensions=None,
                        in_range=False,
                        error=f"{type(e).__name__}: {e}",
                    )
                )
        return results


class JudgeValidator:
    """Phase 0b: run base model on ~50 held-out tb2 tasks via Harbor, score each
    trajectory with the judge + test verifier, compute Spearman ρ between them.

    Decision thresholds (per the design doc):
      rho >= 0.7 -> proceed to training
      0.5 <= rho < 0.7 -> iterate on rubric
      rho < 0.5 -> swap judge model / simplify rubric

    TODO(stage-5): not implemented yet. Blocked on HarborRolloutInterface
    (Stage 4) since this needs to drive `harbor.run_trial(task_id, agent=...)`
    for each held-out task to collect a (trajectory, test_score) pair.
    """

    def __init__(self, task_ids: list[str], base_model: str, judge: Any) -> None: ...

    def run(self) -> JudgeValidationReport: ...

    def _decision_from_rho(self, rho: float) -> str: ...


class JudgeValidationCLI:
    """Argparse + invocation glue around JudgeValidator. Registered as the
    `tb2-validate-judge` console entry point in pyproject.toml.

    TODO(stage-5): not implemented yet. Add once JudgeValidator.run is real.
    """

    def __init__(self, argv: list[str] | None = None) -> None: ...

    def parse_args(self) -> dict[str, object]: ...

    @classmethod
    def run(cls) -> int: ...


if __name__ == "__main__":
    raise SystemExit(JudgeValidationCLI.run())

# Phase 0 judge calibration: compute Spearman rank correlation between judge scores
# and ground-truth test pass scores across a held-out task set. Decides whether the
# judge is trustworthy enough to use as part of the training reward.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class JudgeValidationReport:
    """Output of one calibration run."""

    spearman_rho: float
    p_value: float
    n_tasks: int
    per_task_scores: list[dict[str, Any]]
    decision: str


class JudgeValidator:
    """Runs base model on held-out tasks, scores trajectories with judge + tests, correlates.

    Decision thresholds (per the design doc):
      rho >= 0.7 -> proceed to training
      0.5 <= rho < 0.7 -> iterate on rubric
      rho < 0.5 -> swap judge model / simplify rubric
    """

    def __init__(self, task_ids: list[str], base_model: str, judge: Any) -> None: ...

    def run(self) -> JudgeValidationReport: ...

    def _decision_from_rho(self, rho: float) -> str: ...


class OfflineEvalCaseRunner:
    """Runs the judge against pre-recorded trajectories in tests/judge_eval_cases/.

    Used before live calibration: verifies the rubric isn't broken by checking that
    each fixture's judge score falls within its declared `expected_range`.
    """

    def __init__(self, cases_dir: str, judge: Any) -> None: ...

    def run(self) -> list[dict[str, Any]]: ...

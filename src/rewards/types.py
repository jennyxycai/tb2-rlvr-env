# Shared types for the rewards layer: dataclasses, exceptions, dimension constant.
# No logic. Anything that imports from rewards/* and only needs a type or a result
# shape should import from here to keep the dependency graph shallow.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DIMENSIONS: tuple[str, str, str, str] = (
    "action_success",
    "planning",
    "phase_adherence",
    "tool_effectiveness",
)


class JudgeParseError(ValueError):
    """Raised when the judge model's response can't be parsed into the 4 expected scores."""


@dataclass
class CaseResult:
    """One offline eval case scored: judge output + pass/fail against expected_range."""

    name: str
    expected_range: tuple[float, float]
    actual_score: float | None
    dimensions: dict[str, float] | None
    in_range: bool
    error: str | None = None


@dataclass
class JudgeValidationReport:
    """Output of one Phase 0b live calibration run (Spearman correlation across held-out tasks).
    TODO(stage-5): produced by JudgeValidator.run() once that's wired against Harbor."""

    spearman_rho: float
    p_value: float
    n_tasks: int
    per_task_scores: list[dict[str, Any]]
    decision: str

# Test-verifier reward (85% component): reads Harbor's pytest verifier output from
# /logs/verifier/reward.json and returns a weighted sum of per-test pass/fail in [0, 1].

from __future__ import annotations

from typing import Any


class TestVerifierReward:
    """Weighted-test reward component.

    Pulls per-test results written by Harbor's verifier and applies the per-task
    test weights (defaults to uniform 1/N if no weights configured).
    """

    def __init__(self, weights_by_task: dict[str, dict[str, float]] | None = None) -> None: ...

    def __call__(self, rollout: Any) -> float: ...

    def _load_verifier_output(self, rollout: Any) -> dict[str, bool]: ...

    def _weighted_sum(self, results: dict[str, bool], weights: dict[str, float]) -> float: ...

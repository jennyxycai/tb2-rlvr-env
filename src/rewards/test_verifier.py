# Test-verifier reward: reads per-test pass/fail from Harbor's
# verifier output and returns a weighted sum in [0, 1].

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.rewards.utils import get_field


class TestVerifierReward:
    """Weighted-test reward component.

    Pulls per-test results written by Harbor's verifier and applies the per-task
    test weights (defaults to uniform 1/N if no weights configured).
    """

    def __init__(self, weights_by_task: dict[str, dict[str, float]] | None = None) -> None:
        self.weights_by_task = weights_by_task or {}

    def __call__(self, rollout: Any) -> float:
        results = self._load_verifier_output(rollout)
        task_id = get_field(rollout, "task_id")
        weights = self.weights_by_task.get(task_id) if task_id else None
        return self._weighted_sum(results, weights)

    def _load_verifier_output(self, rollout: Any) -> dict[str, bool]:
        """Return {test_name: passed}. Resolution order:
          1) rollout.verifier_output (direct field)
          2) rollout.metadata['verifier_output']
          3) rollout.metadata['verifier_path'] (path to reward.json on disk)

        TODO(stage-4): align with Harbor's actual reward.json schema once
        HarborRolloutInterface is wired and we know the real key names.
        """
        out = get_field(rollout, "verifier_output")
        if out is None:
            meta = get_field(rollout, "metadata") or {}
            out = meta.get("verifier_output")
            if out is None and "verifier_path" in meta:
                out = json.loads(Path(meta["verifier_path"]).read_text())
        if out is None:
            raise ValueError(
                "rollout has no verifier_output / metadata.verifier_output / metadata.verifier_path"
            )
        return {str(k): bool(v) for k, v in out.items()}

    def _weighted_sum(
        self, results: dict[str, bool], weights: dict[str, float] | None
    ) -> float:
        """Sum of `weight * passed` over all tests. weights=None -> uniform 1/N.
        Returns 0.0 if results is empty. Tests not in weights contribute 0."""
        if not results:
            return 0.0
        if weights is None:
            weights = {k: 1.0 / len(results) for k in results}
        return sum(weights.get(name, 0.0) for name, passed in results.items() if passed)

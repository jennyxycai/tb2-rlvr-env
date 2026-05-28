# End-to-end smoke test: one easy tb2 task, base model, full reward pipeline.
# Marked slow because it pulls a Docker image and calls live model APIs.

from __future__ import annotations

import pytest


@pytest.mark.slow
class TestSmokeRollout:
    """Single-task rollout that exercises the whole stack (env + rewards + combine)."""

    def test_one_easy_task_produces_valid_reward(self) -> None: ...

    def test_rollout_terminates_within_budget(self) -> None: ...

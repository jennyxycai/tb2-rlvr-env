# End-to-end smoke test: one easy tb2 task, base model, full reward pipeline.
# Marked slow because it pulls a Docker image, builds a container, and calls
# live model APIs.

from __future__ import annotations

import os
import shutil
import subprocess

import pytest
import yaml

from src.env.rollout import HarborRolloutInterface
from src.rewards.test_verifier import TestVerifierReward


def _docker_reachable() -> bool:
    """True if the local Docker daemon answers a `docker info` query."""
    if shutil.which("docker") is None:
        return False
    try:
        subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=5,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return False


# Smoke task + cheap model. Overridable via env vars so you don't have to edit
# this file to swap tasks. tb2 has 89 task dirs (no `hello-world`); pick a
# small-Dockerfile, short-timeout one once you've poked around the repo.
#   list: https://github.com/harbor-framework/terminal-bench-2
SMOKE_TASK_ID = "cancel-async-tasks"
SMOKE_MODEL = "claude-haiku-4-5"


@pytest.mark.slow
@pytest.mark.skipif(not _docker_reachable(), reason="Docker daemon not reachable")
@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY") and not os.path.exists(".env"),
    reason="needs ANTHROPIC_API_KEY (env var or .env)",
)
class TestSmokeRollout:
    """Single-task rollout that exercises the whole stack (env + rewards)."""

    @pytest.fixture
    def iface(self) -> HarborRolloutInterface:
        from dotenv import load_dotenv

        load_dotenv()
        env_config = yaml.safe_load(open("configs/env_config.yaml"))
        return HarborRolloutInterface(
            env_config=env_config,
            reward_fns={"tests": TestVerifierReward()},
        )

    def test_one_easy_task_produces_valid_reward(
        self, iface: HarborRolloutInterface
    ) -> None:
        result = iface.rollout(SMOKE_TASK_ID, policy=SMOKE_MODEL)
        assert result.task_id == SMOKE_TASK_ID
        assert "tests" in result.rewards
        assert 0.0 <= result.rewards["tests"] <= 1.0
        assert result.terminated_by in {"complete", "exception", "no_verifier_result"}
        assert "trial_id" in result.metadata

    def test_rollout_terminates_within_budget(
        self, iface: HarborRolloutInterface
    ) -> None:
        """Trial completes (one way or another) without hanging past Harbor's timeout."""
        result = iface.rollout(SMOKE_TASK_ID, policy=SMOKE_MODEL)
        assert result.terminated_by != "no_verifier_result" or result.metadata.get("exception")

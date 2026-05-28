# HarborRolloutInterface: env contract the RL trainer (SkyRL/verifiers) calls into
# to run one tb2 rollout end-to-end (container reset -> agent loop -> verifier -> reward).

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

import harbor

from src.env.dataset_resolver import DatasetResolver, ResolvedTaskPool

log = logging.getLogger(__name__)


@dataclass
class RolloutResult:
    """One completed rollout: trajectory + per-component rewards + termination cause.

    Shape is duck-compatible with the eval-case fixtures in
    tests/judge_eval_cases/, so reward fns work on both live trials and fixtures.
    """

    task_id: str
    trajectory: list[dict[str, Any]]
    rewards: dict[str, float]
    terminated_by: str
    task_instruction: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class RewardFn(Protocol):
    """Callable contract for any reward component (test verifier, llm judge, ...)."""

    def __call__(self, rollout: RolloutResult) -> float: ...


class HarborRolloutInterface:
    """Adapter between Harbor's Trial runner and the RL trainer's rollout loop.

    On __init__: stores env_config + reward_fns and creates a DatasetResolver.
    On rollout(task_id, policy): builds a TrialConfig, runs Harbor's Trial,
    extracts the trajectory and verifier output, dispatches reward_fns, and
    returns a RolloutResult.
    """

    def __init__(self, env_config: dict[str, Any], reward_fns: dict[str, RewardFn]) -> None:
        self.env_config = env_config
        self.reward_fns = reward_fns
        self.resolver = DatasetResolver(env_config)
        self._task_pools: dict[str, ResolvedTaskPool] | None = None

        self.harness = env_config.get("harness", "terminus")
        self.base_model = env_config.get("base_model", "claude-haiku-4-5")
        self.trials_dir = Path(env_config.get("trials_dir", "trials"))

        # Where Harbor fetches task directories from.
        self.harbor_task_repo: str = env_config["harbor_task_repo"]
        self.harbor_task_revision: str | None = env_config.get("harbor_task_revision")

        sandbox = env_config.get("sandbox_provider", "docker")
        self.environment_type = harbor.EnvironmentType[sandbox.upper()]

    def task_pools(self) -> dict[str, ResolvedTaskPool]:
        """Lazily resolve all pools declared in env_config['datasets']."""
        if self._task_pools is None:
            self._task_pools = self.resolver.resolve_all()
        return self._task_pools

    def rollout(self, task_id: str, policy: str | None = None) -> RolloutResult:
        """Run one rollout: build TrialConfig -> Harbor Trial -> dispatch
        reward_fns -> RolloutResult. Wraps Harbor's async API in asyncio.run so
        callers (incl. the smoke test) stay sync."""
        config = self._build_trial_config(task_id, policy)
        result, task_instruction = asyncio.run(self._run_trial(config))

        trial_dir = self._trial_dir_from_result(result)
        trajectory = self._read_trajectory(trial_dir)
        verifier_output = self._read_ctrf(trial_dir)
        terminated_by = self._extract_termination(result)

        partial = RolloutResult(
            task_id=task_id,
            trajectory=trajectory,
            rewards={},
            terminated_by=terminated_by,
            task_instruction=task_instruction,
            metadata={
                "trial_id": str(result.id),
                "trial_uri": result.trial_uri,
                "trial_dir": str(trial_dir),
                "verifier_output": verifier_output,
                "verifier_aggregate": (
                    dict(result.verifier_result.rewards) if result.verifier_result else None
                ),
                "exception": str(result.exception_info) if result.exception_info else None,
            },
        )

        rewards: dict[str, float] = {}
        for name, fn in self.reward_fns.items():
            try:
                rewards[name] = float(fn(partial))
            except Exception as e:
                log.warning("reward fn %r failed on task %s: %s", name, task_id, e)
                rewards[name] = 0.0
        partial.rewards = rewards
        return partial

    async def _run_trial(
        self, config: harbor.TrialConfig
    ) -> tuple[harbor.TrialResult, str]:
        """Async glue: `Trial.create(config)` loads the task + picks the right
        subclass (SingleStep vs MultiStep). We snapshot `trial.task.instruction`
        before running because the TrialResult doesn't carry the Task object
        back; reading the instruction afterward would require re-loading."""
        trial = await harbor.Trial.create(config)
        task_instruction = getattr(trial.task, "instruction", "") or ""
        result = await trial.run()
        return result, task_instruction

    def reset(self, task_id: str) -> None:
        """No-op: Harbor's Trial creates a fresh container per call so there's
        no persistent state to reset between rollouts."""

    def close(self) -> None:
        """No-op: nothing persistent to release."""

    def _build_trial_config(
        self, task_id: str, policy: str | None
    ) -> harbor.TrialConfig:
        return harbor.TrialConfig(
            task=harbor.TrialTaskConfig(
                git_url=self.harbor_task_repo,
                git_commit_id=self.harbor_task_revision,
                # `path` is the subdirectory inside the git repo holding the
                # task's task.toml + environment/ + tests/ (not the agent's
                # working dir). Harbor's TaskConfig disallows both `path` and
                # `name`; `name` is for org/name-style package tasks.
                path=Path(task_id),
            ),
            agent=harbor.TrialAgentConfig(
                name=self.harness,
                model_name=policy or self.base_model,
            ),
            environment=harbor.TrialEnvironmentConfig(type=self.environment_type),
            trials_dir=self.trials_dir,
        )

    def _trial_dir_from_result(self, result: harbor.TrialResult) -> Path:
        """Trial dirs are reported back as `file://` URIs; strip the scheme."""
        parsed = urlparse(result.trial_uri)
        return Path(parsed.path) if parsed.scheme == "file" else Path(result.trial_uri)

    def _read_trajectory(self, trial_dir: Path) -> list[dict[str, Any]]:
        """Read Harbor's ATIF-v1.7 trajectory dump from `agent/trajectory.json`.
        Returns the `steps` array directly (already in the shape LLMJudgeReward's
        trajectory formatter expects: step_id / source / tool_calls / observation.results)."""
        path = trial_dir / "agent" / "trajectory.json"
        if not path.exists():
            log.warning("trajectory.json missing at %s; returning empty list", path)
            return []
        data = json.loads(path.read_text())
        return data.get("steps") or []

    def _read_ctrf(self, trial_dir: Path) -> dict[str, bool] | None:
        """Read per-test pass/fail from `verifier/ctrf.json` (CTRF = Common Test
        Report Format). Returns {test_name: passed_bool}, suitable as
        `metadata.verifier_output` for TestVerifierReward.

        Falls back to None if the file is missing (e.g., the verifier never ran
        because the agent errored out)."""
        path = trial_dir / "verifier" / "ctrf.json"
        if not path.exists():
            log.warning("ctrf.json missing at %s; verifier_output will be None", path)
            return None
        data = json.loads(path.read_text())
        tests = (data.get("results") or {}).get("tests") or []
        return {t["name"]: t.get("status") == "passed" for t in tests if "name" in t}

    def _extract_termination(self, result: harbor.TrialResult) -> str:
        """Best-effort termination cause string."""
        if result.exception_info:
            return "exception"
        if result.verifier_result is None:
            return "no_verifier_result"
        return "complete"

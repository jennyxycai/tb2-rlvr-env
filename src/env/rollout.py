# HarborRolloutInterface: env contract the RL trainer (SkyRL/verifiers) calls into
# to run one tb2 rollout end-to-end (container reset -> agent loop -> verifier -> reward).

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class RolloutResult:
    """One completed rollout: trajectory + per-component rewards + termination cause."""

    task_id: str
    trajectory: list[dict[str, Any]]
    rewards: dict[str, float]
    terminated_by: str
    metadata: dict[str, Any]


class RewardFn(Protocol):
    """Callable contract for any reward component (test verifier, llm judge, ...)."""

    def __call__(self, rollout: RolloutResult) -> float: ...


class HarborRolloutInterface:
    """Adapter between a Harbor task runner and the RL trainer's rollout loop.

    The trainer calls `rollout(task_id, policy)` to collect one trajectory; this class
    owns container lifecycle (via Harbor), the Terminus harness, and reward dispatch.
    """

    def __init__(self, env_config: dict[str, Any], reward_fns: dict[str, RewardFn]) -> None: ...

    def rollout(self, task_id: str, policy: Any) -> RolloutResult: ...

    def reset(self, task_id: str) -> None: ...

    def close(self) -> None: ...

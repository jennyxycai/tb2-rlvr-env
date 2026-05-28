# GroupRewards-style combiner: weighted sum of reward components + group-demean
# across the G=16 rollouts of one prompt to produce the GRPO advantage.

from __future__ import annotations


class RewardTransform:
    """Base class for per-component scalar transforms (clamp, normalize, etc.)."""

    def __call__(self, value: float) -> float: ...


class Clamp(RewardTransform):
    """Clamps a reward into [floor, ceil]."""

    def __init__(self, floor: float, ceil: float) -> None: ...

    def __call__(self, value: float) -> float: ...


class GroupRewards:
    """Combines per-component rewards into a single GRPO advantage.

    Mirrors gypsum's `GroupRewards` pattern: weighted sum across components, optional
    per-component transforms, then demean across the G rollouts of the same prompt.
    """

    def __init__(
        self,
        group_size: int,
        reward_names: list[str],
        weights: dict[str, float],
        transforms: dict[str, list[RewardTransform]] | None = None,
        add_group_demean: bool = True,
    ) -> None: ...

    def combine(self, per_rollout_rewards: list[dict[str, float]]) -> list[float]: ...

    def _apply_transforms(self, name: str, value: float) -> float: ...

    def _demean(self, values: list[float]) -> list[float]: ...

# GroupRewards-style combiner: weighted sum of reward components + group-demean
# across the G=16 rollouts of one prompt to produce the GRPO advantage.

from __future__ import annotations


class RewardTransform:
    """Base class for per-component scalar transforms (clamp, normalize, etc.).
    Subclasses must override `__call__`."""

    def __call__(self, value: float) -> float:
        raise NotImplementedError


class Clamp(RewardTransform):
    """Clamps a reward into [floor, ceil]. Used to enforce the [0, 1] convention
    on `tests` and `llmaj` rewards before they get combined into the advantage."""

    def __init__(self, floor: float, ceil: float) -> None:
        if floor > ceil:
            raise ValueError(f"floor={floor} > ceil={ceil}")
        self.floor = floor
        self.ceil = ceil

    def __call__(self, value: float) -> float:
        if value < self.floor:
            return self.floor
        if value > self.ceil:
            return self.ceil
        return value


class GroupRewards:
    """Combines per-component rewards into a single GRPO advantage.

    Weighted sum across components, optional per-component transforms, then demean across the G rollouts of the same prompt.
    """

    def __init__(
        self,
        group_size: int,
        reward_names: list[str],
        weights: dict[str, float],
        transforms: dict[str, list[RewardTransform]] | None = None,
        add_group_demean: bool = True,
    ) -> None:
        missing = set(reward_names) - set(weights.keys())
        if missing:
            raise ValueError(f"weights missing for declared components: {missing}")
        self.group_size = group_size
        self.reward_names = list(reward_names)
        self.weights = weights
        self.transforms = transforms or {}
        self.add_group_demean = add_group_demean

    def combine(self, per_rollout_rewards: list[dict[str, float]]) -> list[float]:
        """Take per-component rewards for G*K rollouts (K consecutive groups of
        G=group_size rollouts each) and return G*K advantages in the same order.

        For each rollout:
          score = sum_i weights[i] * apply_transforms(name_i, reward_i)
        Then, if add_group_demean, subtract each group's mean from its members.
        """
        if len(per_rollout_rewards) % self.group_size != 0:
            raise ValueError(
                f"got {len(per_rollout_rewards)} rollouts; "
                f"must be a multiple of group_size={self.group_size}"
            )

        scores: list[float] = []
        for rewards in per_rollout_rewards:
            total = 0.0
            for name in self.reward_names:
                if name not in rewards:
                    raise ValueError(
                        f"rollout missing reward component {name!r}; "
                        f"got {list(rewards.keys())}"
                    )
                total += self.weights[name] * self._apply_transforms(name, rewards[name])
            scores.append(total)

        if not self.add_group_demean:
            return scores

        out: list[float] = []
        for i in range(0, len(scores), self.group_size):
            out.extend(self._demean(scores[i : i + self.group_size]))
        return out

    def _apply_transforms(self, name: str, value: float) -> float:
        for t in self.transforms.get(name) or []:
            value = t(value)
        return value

    def _demean(self, values: list[float]) -> list[float]:
        """Subtract the mean of `values` from each entry. Empty input -> []."""
        if not values:
            return []
        mean = sum(values) / len(values)
        return [v - mean for v in values]

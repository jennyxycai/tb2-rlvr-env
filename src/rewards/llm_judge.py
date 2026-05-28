# LLM-as-judge reward (15% component): scores a trajectory across 4 rubric
# dimensions (action success / planning / phase adherence / tool effectiveness)
# and averages to a single [0, 1] reward. Backend chain handles API rate-limit failover.

from __future__ import annotations

from typing import Any


class JudgeBackend:
    """One judge model endpoint (e.g., Claude Sonnet 4). Supports retry + backoff."""

    def __init__(self, model: str, temperature: float, max_retries: int) -> None: ...

    def score(self, prompt: str) -> dict[str, float]: ...


class JudgeBackendChain:
    """Fallback chain of JudgeBackends (Sonnet -> Opus -> Gemini Pro) for rate-limit resilience."""

    def __init__(self, backends: list[JudgeBackend]) -> None: ...

    def score(self, prompt: str) -> dict[str, float]: ...


class LLMJudgeReward:
    """Trajectory-scoring reward component.

    Renders the trajectory into the rubric prompt, calls the backend chain, parses
    the 4 dimension scores, and returns their mean in [0, 1].
    """

    DIMENSIONS = ("action_success", "planning", "phase_adherence", "tool_effectiveness")

    def __init__(self, backend: JudgeBackendChain, rubric_path: str) -> None: ...

    def __call__(self, rollout: Any) -> float: ...

    def _render_prompt(self, rollout: Any) -> str: ...

    def _parse_dimensions(self, response: str) -> dict[str, float]: ...

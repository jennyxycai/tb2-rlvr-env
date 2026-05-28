# LLM-based context compaction: summarizes old (action, observation) pairs when a
# trajectory's prompt exceeds the configured token budget (~32k for Qwen3).

from __future__ import annotations

from typing import Any


class TrajectoryCompactor:
    """Compacts a trajectory prefix into a short summary while preserving the last K turns.

    Invoked by the rollout loop when prompt tokens exceed `max_tokens`; the prefix is
    replaced by a single summary turn so the conversation history stays under budget.
    """

    def __init__(self, summarizer_model: str, max_tokens: int, keep_last_k: int) -> None: ...

    def should_compact(self, trajectory: list[dict[str, Any]], current_tokens: int) -> bool: ...

    def compact(self, trajectory: list[dict[str, Any]]) -> list[dict[str, Any]]: ...

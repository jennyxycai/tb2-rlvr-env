# Dataset resolver: turns env_config.yaml's `datasets` section into a concrete list
# of TaskRefs at trial-pool construction time. Refuses to resolve mutable revisions
# (e.g. 'main') unless explicitly allowed; snapshots the resolved list for run records.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class TaskRef:
    """One resolved task with provenance metadata. Carries everything the rollout
    interface needs to fetch the task + everything the run record needs to trace it."""

    task_id: str
    source_name: str  # e.g. 'Danau5tin/terminal-bench-rl', 'open-thoughts/OpenThoughts-Agent-v1-RL'
    revision: str  # pinned commit SHA / HF revision hash
    difficulty: str | None  # 'easy' | 'medium' | 'hard' | 'extremely_hard' | None
    weights: dict[str, float] | None  # per-test weights if the dataset ships them


@dataclass
class ResolvedTaskPool:
    """One of: training RL pool, SFT warmup pool, eval pool. Reproducibility unit."""

    name: str  # 'rl' | 'sft_warmup' | 'eval'
    tasks: list[TaskRef]

    def snapshot_to(self, path: str) -> None: ...

    @classmethod
    def from_snapshot(cls, path: str) -> ResolvedTaskPool: ...


class DatasetSource(Protocol):
    """One concrete dataset backend. Implementations: GitHubCSVSource, HuggingFaceSource."""

    source_name: str
    revision: str

    def load(self) -> list[TaskRef]: ...


class GitHubCSVSource:
    """Reads a CSV from a pinned GitHub revision (e.g. terminal-bench-rl's latest_verified.csv).
    Applies optional row-level filters (difficulty, etc.) before returning TaskRefs."""

    def __init__(
        self,
        repo: str,
        path: str,
        revision: str,
        filter: dict[str, Any] | None = None,
    ) -> None: ...

    def load(self) -> list[TaskRef]: ...

    def _fetch_csv(self) -> str: ...

    def _apply_filter(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]: ...


class HuggingFaceSource:
    """Loads a pinned HF dataset revision via `datasets.load_dataset`.
    Applies optional row-level filters before returning TaskRefs."""

    def __init__(
        self,
        name: str,
        revision: str,
        filter: dict[str, Any] | None = None,
    ) -> None: ...

    def load(self) -> list[TaskRef]: ...

    def _apply_filter(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]: ...


class DatasetResolver:
    """Top-level orchestrator. Reads env_config dict, returns ResolvedTaskPools.

    Safety:
      - `task_ids_override` short-circuits all dataset loads and returns the override pool.
      - If any source has revision == 'main' (or similar mutable ref) and
        `allow_mutable_revisions` is False, raises before any network call.
    Reproducibility:
      - Caller is expected to immediately snapshot each ResolvedTaskPool to disk
        and commit that snapshot as a run artifact.
    """

    MUTABLE_REVS = {"main", "master", "HEAD", "latest"}

    def __init__(
        self, env_config: dict[str, Any], allow_mutable_revisions: bool = False
    ) -> None: ...

    def resolve(self, pool_name: str) -> ResolvedTaskPool: ...

    def resolve_all(self) -> dict[str, ResolvedTaskPool]: ...

    def _build_source(self, spec: dict[str, Any]) -> DatasetSource: ...

    def _check_revision(self, revision: str) -> None: ...

    def _override_pool(self, name: str) -> ResolvedTaskPool | None: ...

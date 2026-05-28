# Dataset resolver: turns env_config.yaml's `datasets` section into a concrete list
# of TaskRefs at trial-pool construction time. Refuses to resolve mutable revisions
# (e.g. 'main') unless explicitly allowed; snapshots the resolved list for run records.

from __future__ import annotations

import csv
import io
import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import pyarrow.parquet as pq
import requests
from huggingface_hub import hf_hub_download, list_repo_files

log = logging.getLogger(__name__)


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

    def snapshot_to(self, path: str) -> None:
        """Write this pool to a JSON file. Caller checks the file in as a run
        artifact so the exact (task_id, source, revision) set is recoverable later
        even if env_config.yaml or upstream datasets change."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": self.name,
            "task_count": len(self.tasks),
            "snapshot_created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "tasks": [asdict(t) for t in self.tasks],
        }
        p.write_text(json.dumps(payload, indent=2))

    @classmethod
    def from_snapshot(cls, path: str) -> ResolvedTaskPool:
        """Load a pool previously written by `snapshot_to`."""
        data = json.loads(Path(path).read_text())
        return cls(
            name=data["name"],
            tasks=[TaskRef(**t) for t in data["tasks"]],
        )


class DatasetSource(Protocol):
    """One concrete dataset backend. Implementations: GitHubCSVSource, HuggingFaceSource."""

    source_name: str
    revision: str

    def load(self) -> list[TaskRef]: ...


class GitHubCSVSource:
    """Reads a CSV from a pinned GitHub revision (e.g. terminal-bench-rl's
    latest_verified.csv). Applies optional row-level filters and parses the
    `test_weights` column (JSON string -> dict[str, float]) into TaskRef.weights."""

    RAW_URL = "https://raw.githubusercontent.com/{repo}/{revision}/{path}"

    def __init__(
        self,
        repo: str,
        path: str,
        revision: str,
        filter: dict[str, Any] | None = None,
    ) -> None:
        self.repo = repo
        self.path = path
        self.revision = revision
        self.filter = filter or {}
        self.source_name = repo

    def load(self) -> list[TaskRef]:
        rows = list(csv.DictReader(io.StringIO(self._fetch_csv())))
        rows = self._apply_filter(rows)
        tasks: list[TaskRef] = []
        for row in rows:
            task_id = (row.get("task_id") or "").strip()
            if not task_id:
                continue
            tasks.append(
                TaskRef(
                    task_id=task_id,
                    source_name=self.source_name,
                    revision=self.revision,
                    difficulty=(row.get("difficulty") or None),
                    weights=self._parse_weights(row.get("test_weights")),
                )
            )
        return tasks

    def _fetch_csv(self) -> str:
        url = self.RAW_URL.format(repo=self.repo, revision=self.revision, path=self.path)
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.text

    def _apply_filter(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Keep rows where every (column, allowed) pair matches.
        `allowed` may be a single value or a list of values."""
        if not self.filter:
            return rows

        def keep(row: dict[str, Any]) -> bool:
            for col, allowed in self.filter.items():
                allowed_set = {allowed} if isinstance(allowed, (str, int, float)) else set(allowed)
                if row.get(col) not in allowed_set:
                    return False
            return True

        return [r for r in rows if keep(r)]

    @staticmethod
    def _parse_weights(value: str | None) -> dict[str, float] | None:
        """Parse the `test_weights` CSV column (a JSON string) into a {test: weight}
        dict. Empty / unparseable values return None so the verifier falls back to
        uniform 1/N weighting at runtime."""
        if not value or not value.strip():
            return None
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as e:
            log.warning("test_weights JSON parse failed: %s; value=%r", e, value[:120])
            return None
        if not isinstance(parsed, dict):
            return None
        return {str(k): float(v) for k, v in parsed.items()}


class HuggingFaceSource:
    """Loads task IDs from a HuggingFace dataset at a pinned revision.

    Supports two backend modes:
      - `mode="parquet"` (default): downloads a single parquet file at the
        pinned revision and extracts task IDs from a configured column
        (`id_column`, default 'path'). Works for OpenThoughts-Agent-v1-RL.
      - `mode="dir_tree"`: TODO(stage-3-ext) - lists top-level directories
        in the repo and treats each containing `task.toml` as a task. Needed
        for `zai-org/terminal-bench-2-verified` which ships one dir per task.
    """

    def __init__(
        self,
        name: str,
        revision: str,
        mode: str = "parquet",
        id_column: str = "path",
        parquet_file: str | None = None,
        filter: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.revision = revision
        self.mode = mode
        self.id_column = id_column
        self.parquet_file = parquet_file
        self.filter = filter or {}
        self.source_name = name

    def load(self) -> list[TaskRef]:
        if self.mode == "parquet":
            return self._load_parquet()
        if self.mode == "dir_tree":
            raise NotImplementedError(
                "HuggingFaceSource(mode='dir_tree') is not implemented yet. "
                "TODO(stage-3-ext): list top-level dirs at the pinned revision and "
                "filter those containing task.toml; used for tb2-verified."
            )
        raise ValueError(f"unknown HuggingFaceSource mode: {self.mode!r}")

    def _load_parquet(self) -> list[TaskRef]:
        file = self.parquet_file
        if file is None:
            files = list_repo_files(
                repo_id=self.name, repo_type="dataset", revision=self.revision
            )
            parquets = [f for f in files if f.endswith(".parquet")]
            if len(parquets) != 1:
                raise ValueError(
                    f"expected exactly 1 parquet in {self.name}@{self.revision}, "
                    f"found {len(parquets)}: {parquets}. Set `parquet_file` explicitly."
                )
            file = parquets[0]

        local_path = hf_hub_download(
            repo_id=self.name,
            filename=file,
            repo_type="dataset",
            revision=self.revision,
        )

        # Only pull the columns we need (avoids reading task_binary blobs).
        wanted = [self.id_column] + [c for c in self.filter if c != self.id_column]
        wanted = [c for c in wanted if c in pq.read_schema(local_path).names]
        table = pq.read_table(local_path, columns=wanted)
        rows = self._apply_filter(table.to_pylist())

        tasks: list[TaskRef] = []
        for r in rows:
            task_id = r.get(self.id_column)
            if not task_id:
                continue
            tasks.append(
                TaskRef(
                    task_id=str(task_id),
                    source_name=self.source_name,
                    revision=self.revision,
                    difficulty=(str(r["difficulty"]) if r.get("difficulty") else None),
                    weights=None,  # HF datasets don't ship per-test weights
                )
            )
        return tasks

    def _apply_filter(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self.filter:
            return rows

        def keep(row: dict[str, Any]) -> bool:
            for col, allowed in self.filter.items():
                allowed_set = (
                    {allowed} if isinstance(allowed, (str, int, float)) else set(allowed)
                )
                if row.get(col) not in allowed_set:
                    return False
            return True

        return [r for r in rows if keep(r)]


class DatasetResolver:
    """Top-level orchestrator. Reads env_config dict, returns ResolvedTaskPools.

    Caller is expected to snapshot each ResolvedTaskPool to disk as a run artifact
    (via ResolvedTaskPool.snapshot_to) for reproducibility.
    """

    def __init__(self, env_config: dict[str, Any]) -> None:
        self.env_config = env_config

    def resolve(self, pool_name: str) -> ResolvedTaskPool:
        """Resolve one pool to a ResolvedTaskPool. `task_ids_override` (if set in
        env_config) short-circuits the dataset load and applies to every pool name."""
        override = self._override_pool(pool_name)
        if override is not None:
            return override

        datasets = self.env_config.get("datasets") or {}
        if pool_name not in datasets:
            raise KeyError(
                f"pool {pool_name!r} not declared in env_config['datasets']; "
                f"available: {list(datasets.keys())}"
            )

        tasks = [
            t
            for spec in datasets[pool_name]
            for t in self._build_source(spec).load()
        ]
        return ResolvedTaskPool(name=pool_name, tasks=tasks)

    def resolve_all(self) -> dict[str, ResolvedTaskPool]:
        """Resolve every pool declared in env_config['datasets']."""
        datasets = self.env_config.get("datasets") or {}
        return {name: self.resolve(name) for name in datasets}

    def _build_source(self, spec: dict[str, Any]) -> DatasetSource:
        kind = spec.get("source")
        match kind:
            case "github":
                return GitHubCSVSource(
                    repo=spec["repo"],
                    path=spec["path"],
                    revision=spec["revision"],
                    filter=spec.get("filter"),
                )
            case "huggingface":
                return HuggingFaceSource(
                    name=spec["name"],
                    revision=spec["revision"],
                    mode=spec.get("mode", "parquet"),
                    id_column=spec.get("id_column", "path"),
                    parquet_file=spec.get("parquet_file"),
                    filter=spec.get("filter"),
                )
            case _:
                raise ValueError(f"unknown dataset source kind: {kind!r}")

    def _override_pool(self, name: str) -> ResolvedTaskPool | None:
        """If `task_ids_override` is set in env_config, build a synthetic pool from
        those IDs and return it. Applies to every pool name (overrides everything).
        Returns None when no override is configured."""
        override = self.env_config.get("task_ids_override")
        if not override:
            return None
        tasks = [
            TaskRef(
                task_id=str(tid),
                source_name="__override__",
                revision="__override__",
                difficulty=None,
                weights=None,
            )
            for tid in override
        ]
        return ResolvedTaskPool(name=name, tasks=tasks)

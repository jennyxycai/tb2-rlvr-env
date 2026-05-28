# Shared helpers for the rewards layer (string truncation, dict/attr accessor).

from __future__ import annotations

from typing import Any


def get_field(obj: Any, name: str) -> Any:
    """Read `name` from either a dict (via .get) or a dataclass / object (via getattr).
    Returns None if the field is missing. Used so reward fns work against both
    live RolloutResult objects and pre-recorded eval-case fixture dicts."""
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def short(v: Any, limit: int = 300) -> str:
    """Stringify `v` and truncate to `limit` chars with an ellipsis. Lets the
    judge prompt include tool-call args / env results without blowing up on
    long shell output."""
    s = v if isinstance(v, str) else repr(v)
    if len(s) <= limit:
        return s
    return s[: limit - 3] + "..."

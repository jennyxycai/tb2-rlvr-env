# LLM-as-judge reward: renders a rollout into the rubric prompt, calls the
# judge backend, averages the 4 dimension scores into a single [0, 1] reward.

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.rewards.judge_backend import JudgeBackend
from src.rewards.types import DIMENSIONS
from src.rewards.utils import get_field, short


class LLMJudgeReward:
    """Trajectory-scoring reward component.

    Renders the trajectory into the rubric prompt, calls the backend, parses
    the 4 dimension scores, and returns their mean in [0, 1].

    Accepts either a RolloutResult-like object (with .trajectory and .task_instruction
    attrs) or a dict with those keys, so it works against both live rollouts and
    pre-recorded eval-case fixtures.
    """

    def __init__(self, backend: JudgeBackend, rubric_path: str) -> None:
        self.backend = backend
        self.rubric_path = Path(rubric_path)
        self._rubric_cache: str | None = None
        self._last_dimensions: dict[str, float] | None = None

    def __call__(self, rollout: Any) -> float:
        prompt = self._render_prompt(rollout)
        scores = self.backend.score(prompt)
        self._last_dimensions = scores
        return sum(scores.values()) / len(DIMENSIONS)

    @property
    def last_dimensions(self) -> dict[str, float] | None:
        """Per-dimension scores from the most recent __call__. Useful for logging."""
        return self._last_dimensions

    def _render_prompt(self, rollout: Any) -> str:
        rubric = self._load_rubric()
        task_instruction = get_field(rollout, "task_instruction") or ""
        trajectory = get_field(rollout, "trajectory") or []
        traj_text = self._format_trajectory(trajectory)
        return (
            f"{rubric}\n\n"
            f"---\n\n"
            f"# Inputs (actual)\n\n"
            f"## task_instruction\n\n{task_instruction}\n\n"
            f"## trajectory\n\n{traj_text}\n\n"
            f"---\n\n"
            f"Now emit the YAML block with the four score keys. No prose.\n"
        )

    def _format_trajectory(self, trajectory: list[dict[str, Any]]) -> str:
        """Render ATIF steps to readable text using `[Step N by Role]` delimiters
        (parser-friendly, LLM-readable, similar to terminal-bench-rl but not in XML)."""
        if not trajectory:
            return "(empty trajectory)"
        out: list[str] = []
        for step in trajectory:
            step_id = step.get("step_id", "?")
            source = step.get("source", "?")
            if source == "agent":
                role = "Agent"
                out.append(f"[Step {step_id} by {role}]")
                reasoning = (step.get("reasoning_content") or "").strip()
                if reasoning:
                    out.append(f"reasoning: {reasoning}")
                for call in step.get("tool_calls") or []:
                    fn = call.get("function_name")
                    args = call.get("arguments") or {}
                    cid = call.get("tool_call_id", "?")
                    args_str = ", ".join(f"{k}={short(v)}" for k, v in args.items())
                    out.append(f"  -> {fn}({args_str})  [call_id={cid}]")
            elif source == "user":
                role = "Env Response"
                out.append(f"[Step {step_id} by {role}]")
                for r in step.get("results") or []:
                    cid = r.get("source_call_id", "?")
                    content = r.get("content", "")
                    out.append(f"  <- [{cid}] {short(content, limit=600)}")
            else:
                out.append(f"[Step {step_id} by {source}]")
                out.append(short(step, limit=600))
            out.append("")
        return "\n".join(out)

    def _load_rubric(self) -> str:
        if self._rubric_cache is None:
            self._rubric_cache = self.rubric_path.read_text()
        return self._rubric_cache

# Unit tests for the reward stack: TestVerifierReward, JudgeBackend parser,
# GroupRewards combiner, plus offline judge rubric checks against fixtures.

from __future__ import annotations

import json
import os

import pytest

from src.rewards.combine import Clamp, GroupRewards
from src.rewards.judge_backend import JudgeBackend
from src.rewards.test_verifier import TestVerifierReward
from src.rewards.types import JudgeParseError

POSTGRES_WEIGHTS = {
    "test_db_starts": 0.20,
    "test_basic_query": 0.20,
    "test_concurrent_reads": 0.30,
    "test_concurrent_writes": 0.30,
}


class TestTestVerifierReward:
    """Per-test weighted-sum behavior + missing-weights uniform fallback."""

    def test_uniform_weights_when_unspecified(self) -> None:
        r = TestVerifierReward()
        rollout = {"verifier_output": {"a": True, "b": True, "c": False, "d": False}}
        assert r(rollout) == 0.5

    def test_partial_pass_yields_partial_reward(self) -> None:
        r = TestVerifierReward(weights_by_task={"pg": POSTGRES_WEIGHTS})
        rollout = {
            "task_id": "pg",
            "verifier_output": {
                "test_db_starts": True,
                "test_basic_query": True,
                "test_concurrent_reads": False,
                "test_concurrent_writes": False,
            },
        }
        assert r(rollout) == pytest.approx(0.40)

    def test_all_pass_yields_one(self) -> None:
        r = TestVerifierReward(weights_by_task={"pg": POSTGRES_WEIGHTS})
        rollout = {
            "task_id": "pg",
            "verifier_output": dict.fromkeys(POSTGRES_WEIGHTS, True),
        }
        assert r(rollout) == pytest.approx(1.0)

    def test_all_fail_yields_zero(self) -> None:
        r = TestVerifierReward()
        rollout = {"verifier_output": {"a": False, "b": False}}
        assert r(rollout) == 0.0

    def test_load_from_metadata_path(self, tmp_path) -> None:
        """Verifier output can also be loaded from a path in metadata."""
        path = tmp_path / "reward.json"
        path.write_text(json.dumps({"t1": True, "t2": False}))
        r = TestVerifierReward()
        assert r({"metadata": {"verifier_path": str(path)}}) == 0.5


class TestLLMJudgeReward:
    """Dimension parsing. Parser is pure; no API mock needed."""

    def test_dimensions_average_to_final_score(self) -> None:
        be = JudgeBackend(model="placeholder")
        parsed = be._parse(
            "action_success: 0.8\n"
            "planning: 0.6\n"
            "phase_adherence: 0.4\n"
            "tool_effectiveness: 0.2"
        )
        assert parsed == {
            "action_success": 0.8,
            "planning": 0.6,
            "phase_adherence": 0.4,
            "tool_effectiveness": 0.2,
        }
        assert sum(parsed.values()) / 4 == pytest.approx(0.5)

    def test_malformed_response_raises(self) -> None:
        be = JudgeBackend(model="placeholder")

        with pytest.raises(JudgeParseError):
            be._parse("not yaml at all, no keys")

        with pytest.raises(JudgeParseError):
            # Out-of-range value
            be._parse(
                "action_success: 1.5\n"
                "planning: 0.5\n"
                "phase_adherence: 0.5\n"
                "tool_effectiveness: 0.5"
            )

        with pytest.raises(JudgeParseError):
            # Missing one dimension
            be._parse(
                "action_success: 0.5\nplanning: 0.5\nphase_adherence: 0.5"
            )

    def test_yaml_fences_stripped(self) -> None:
        be = JudgeBackend(model="placeholder")
        parsed = be._parse(
            "```yaml\n"
            "action_success: 0.5\n"
            "planning: 0.5\n"
            "phase_adherence: 0.5\n"
            "tool_effectiveness: 0.5\n"
            "```"
        )
        assert parsed["action_success"] == 0.5


class TestGroupRewards:
    """Weighted combine + per-component clamp + group demean."""

    def test_weighted_combine_matches_design_split(self) -> None:
        gr = GroupRewards(
            group_size=4,
            reward_names=["tests", "llmaj"],
            weights={"tests": 0.85, "llmaj": 0.15},
            add_group_demean=False,
        )
        scores = gr.combine(
            [
                {"tests": 1.0, "llmaj": 1.0},   # 0.85 + 0.15 = 1.0
                {"tests": 0.5, "llmaj": 0.0},   # 0.425
                {"tests": 0.0, "llmaj": 0.5},   # 0.075
                {"tests": 0.0, "llmaj": 0.0},   # 0.0
            ]
        )
        assert scores[0] == pytest.approx(1.0)
        assert scores[1] == pytest.approx(0.425)
        assert scores[2] == pytest.approx(0.075)
        assert scores[3] == pytest.approx(0.0)

    def test_clamp_transform_applied(self) -> None:
        gr = GroupRewards(
            group_size=2,
            reward_names=["x"],
            weights={"x": 1.0},
            transforms={"x": [Clamp(0.0, 1.0)]},
            add_group_demean=False,
        )
        scores = gr.combine([{"x": 1.5}, {"x": -0.5}])
        assert scores == [1.0, 0.0]

    def test_group_demean_zero_variance_group(self) -> None:
        gr = GroupRewards(
            group_size=4,
            reward_names=["r"],
            weights={"r": 1.0},
            add_group_demean=True,
        )
        advs = gr.combine([{"r": 0.5}] * 4)
        assert all(a == pytest.approx(0.0) for a in advs)

    def test_demean_across_two_groups_independent(self) -> None:
        gr = GroupRewards(
            group_size=2,
            reward_names=["r"],
            weights={"r": 1.0},
            add_group_demean=True,
        )
        # group 1: [1.0, 0.0] -> mean 0.5 -> [+0.5, -0.5]
        # group 2: [0.8, 0.4] -> mean 0.6 -> [+0.2, -0.2]
        advs = gr.combine([{"r": 1.0}, {"r": 0.0}, {"r": 0.8}, {"r": 0.4}])
        assert advs[0] == pytest.approx(0.5)
        assert advs[1] == pytest.approx(-0.5)
        assert advs[2] == pytest.approx(0.2)
        assert advs[3] == pytest.approx(-0.2)


@pytest.mark.slow
@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY") and not os.path.exists(".env"),
    reason="needs ANTHROPIC_API_KEY (env var or .env)",
)
class TestJudgeOnEvalCases:
    """Loads tests/judge_eval_cases/*.json and asserts each judge score is in
    expected_range. Marked slow because it makes 6 live API calls (~$0.30)."""

    def test_all_cases_within_expected_range(self) -> None:
        from dotenv import load_dotenv

        load_dotenv()

        from src.rewards.judge_validation import OfflineEvalCaseRunner
        from src.rewards.llm_judge import LLMJudgeReward

        judge = LLMJudgeReward(
            JudgeBackend("claude-sonnet-4-6"),
            rubric_path="src/prompts/judge_rubric.md",
        )
        runner = OfflineEvalCaseRunner("tests/judge_eval_cases", judge)
        results = runner.run()

        assert len(results) == 6
        failures = [(r.name, r.actual_score, r.expected_range, r.error) for r in results if not r.in_range]
        assert not failures, f"out-of-range or errored: {failures}"

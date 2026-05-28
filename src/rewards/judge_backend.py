# Judge backend: one litellm endpoint that scores a single prompt and parses
# the 4-dimension YAML response. No retry, no fallback chain.

from __future__ import annotations

import re

import yaml
from litellm import completion

from src.rewards.types import DIMENSIONS, JudgeParseError


class JudgeBackend:
    """One judge model endpoint. Single litellm call, no retry nor fallback.
    If the call fails or the response can't be parsed, the error bubbles up to
    the caller."""

    def __init__(self, model: str, temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature

    def score(self, prompt: str) -> dict[str, float]:
        """Call the model and return the 4 parsed scores. Raises on failure."""
        resp = completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        text = resp.choices[0].message.content or ""
        return self._parse(text)

    def _parse(self, response: str) -> dict[str, float]:
        """Extract the 4 dimension scores from the model's response.

        Strategy mirrors terminal-bench-rl's layered parser:
          1) strip markdown fences (```yaml / ```)
          2) yaml.safe_load
          3) regex fallback per-dimension
        Raises JudgeParseError if any required dim is missing or out of [0, 1].
        """
        text = self._strip_fences(response).strip()
        scores: dict[str, float] = {}

        try:
            data = yaml.safe_load(text)
            if isinstance(data, dict):
                for dim in DIMENSIONS:
                    if dim in data and isinstance(data[dim], (int, float)):
                        scores[dim] = float(data[dim])
        except yaml.YAMLError:
            pass

        # Regex fallback for any dimension YAML didn't recover.
        for dim in DIMENSIONS:
            if dim in scores:
                continue
            m = re.search(rf"{dim}\s*[:=]\s*([0-9]*\.?[0-9]+)", response, re.IGNORECASE)
            if m:
                scores[dim] = float(m.group(1))

        missing = [d for d in DIMENSIONS if d not in scores]
        if missing:
            raise JudgeParseError(f"missing dimensions {missing} in response: {response[:500]!r}")
        for dim, v in scores.items():
            if not 0.0 <= v <= 1.0:
                raise JudgeParseError(f"{dim}={v} out of [0, 1] in: {response[:500]!r}")
        return scores

    @staticmethod
    def _strip_fences(text: str) -> str:
        text = re.sub(r"^```(?:yaml|yml)?\s*\n", "", text.strip(), flags=re.IGNORECASE)
        text = re.sub(r"\n```\s*$", "", text)
        return text

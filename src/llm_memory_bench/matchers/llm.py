from __future__ import annotations

from ..providers.base import LLMProvider
from .base import FactMatcher, MatchVerdict

JUDGE_PROMPT = """\
You are evaluating whether a stored memory captures the same information as an expected fact.

Expected fact: {expected}
Stored memory: {stored}

Does the stored memory capture the same core information as the expected fact?
Consider semantic equivalence, not exact wording. Minor differences in phrasing are acceptable.
Missing important details or adding incorrect information counts against a match.

Respond with exactly one word: MATCH, PARTIAL, or NO_MATCH"""


class LLMMatcher(FactMatcher):
    """Uses an LLM as a judge for semantic fact matching."""

    name = "llm"

    def __init__(self, provider: LLMProvider):
        self.provider = provider

    async def match(self, expected: str, stored: str) -> MatchVerdict:
        prompt = JUDGE_PROMPT.format(expected=expected, stored=stored)
        response = await self.provider.generate(
            messages=[{"role": "user", "content": prompt}],
            tools=[],
            system="You are a precise evaluator. Respond with exactly one word.",
        )
        text = response.text.strip().upper()
        if "MATCH" in text and "NO_MATCH" not in text and "PARTIAL" not in text:
            return MatchVerdict.MATCH
        elif "PARTIAL" in text:
            return MatchVerdict.PARTIAL
        return MatchVerdict.NO_MATCH

from __future__ import annotations

from ..providers.base import LLMProvider
from .base import FactMatcher, MatchVerdict
from .embedding import DEFAULT_MODEL

DEFAULT_CONFIDENT_MATCH_THRESHOLD = 0.75
DEFAULT_CONFIDENT_MISMATCH_THRESHOLD = 0.4


class HybridMatcher(FactMatcher):
    """Two-stage matcher: embedding filter + LLM for uncertain cases.

    Strategy:
    1. High confidence (similarity >= 0.75) → MATCH (skip LLM)
    2. Low confidence (similarity < 0.4) → NO_MATCH (skip LLM)
    3. Uncertain middle (0.4-0.75) → Escalate to LLM

    Reduces LLM calls by ~60-80% while fixing embedding false negatives.
    """

    name = "hybrid"

    def __init__(
        self,
        provider: LLMProvider,
        embedding_model: str = DEFAULT_MODEL,
        confident_match_threshold: float = DEFAULT_CONFIDENT_MATCH_THRESHOLD,
        confident_mismatch_threshold: float = DEFAULT_CONFIDENT_MISMATCH_THRESHOLD,
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for the hybrid matcher.\n"
                "Install it with: pip install sentence-transformers"
            ) from None

        self.embedding_model = SentenceTransformer(embedding_model)
        self.provider = provider
        self.confident_match_threshold = confident_match_threshold
        self.confident_mismatch_threshold = confident_mismatch_threshold

        # Track escalation stats
        self.total_comparisons = 0
        self.llm_escalations = 0

    async def match(self, expected: str, stored: str) -> MatchVerdict:
        self.total_comparisons += 1

        # Stage 1: Fast embedding filter
        embeddings = self.embedding_model.encode(
            [expected, stored], normalize_embeddings=True
        )
        similarity = float(embeddings[0] @ embeddings[1])

        # High confidence match - skip LLM
        if similarity >= self.confident_match_threshold:
            return MatchVerdict.MATCH

        # High confidence mismatch - skip LLM
        if similarity < self.confident_mismatch_threshold:
            return MatchVerdict.NO_MATCH

        # Stage 2: Uncertain zone - escalate to LLM
        self.llm_escalations += 1

        from .llm import JUDGE_PROMPT

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

    def escalation_rate(self) -> float:
        """Fraction of comparisons that required LLM escalation."""
        if self.total_comparisons == 0:
            return 0.0
        return self.llm_escalations / self.total_comparisons

    async def close(self) -> None:
        """Print escalation stats on cleanup."""
        if self.total_comparisons > 0:
            from rich.console import Console
            console = Console()
            rate = self.escalation_rate()
            console.print(
                f"[dim]Hybrid matcher stats: {self.llm_escalations}/{self.total_comparisons} "
                f"escalated to LLM ({rate:.1%})[/dim]"
            )

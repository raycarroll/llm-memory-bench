from __future__ import annotations

from .alpsbench import AlpsBenchMatcher
from .base import FactMatcher, MatchVerdict
from .embedding import EmbeddingMatcher
from .exact import ExactMatcher
from .hybrid import HybridMatcher
from .llm import LLMMatcher

MATCHERS: dict[str, type[FactMatcher]] = {
    "llm": LLMMatcher,
    "embedding": EmbeddingMatcher,
    "hybrid": HybridMatcher,
    "exact": ExactMatcher,
    "alpsbench": AlpsBenchMatcher,
}


def list_matchers() -> list[str]:
    return list(MATCHERS.keys())

from __future__ import annotations

from .base import FactMatcher, MatchVerdict
from .embedding import EmbeddingMatcher
from .hybrid import HybridMatcher
from .llm import LLMMatcher

MATCHERS: dict[str, type[FactMatcher]] = {
    "llm": LLMMatcher,
    "embedding": EmbeddingMatcher,
    "hybrid": HybridMatcher,
}


def list_matchers() -> list[str]:
    return list(MATCHERS.keys())

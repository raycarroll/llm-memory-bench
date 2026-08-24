from __future__ import annotations

from .base import FactMatcher, MatchVerdict
from .embedding import EmbeddingMatcher
from .llm import LLMMatcher

MATCHERS: dict[str, type[FactMatcher]] = {
    "llm": LLMMatcher,
    "embedding": EmbeddingMatcher,
}


def list_matchers() -> list[str]:
    return list(MATCHERS.keys())

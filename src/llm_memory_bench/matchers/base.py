from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum


class MatchVerdict(str, Enum):
    MATCH = "match"
    PARTIAL = "partial"
    NO_MATCH = "no_match"


class FactMatcher(ABC):
    """Determines whether a stored memory captures the same info as an expected fact."""

    name: str

    @abstractmethod
    async def match(self, expected: str, stored: str) -> MatchVerdict: ...

    async def close(self) -> None:
        """Release resources (model, connection, etc). Called once after evaluation."""

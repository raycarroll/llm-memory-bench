"""Exact matcher (AlpsBench-style).

Matches on canonical 6-tuple like AlpsBench:
- (label, value, type, preference_attitude, time_scope, emotion)

Falls back to value-only comparison for legacy string format.
"""
from __future__ import annotations

import re
from typing import Any

from .base import FactMatcher, MatchVerdict

# Regex for collapsing whitespace (from AlpsBench metrics.py)
_WS_RE = re.compile(r"\s+")


def normalize_text(text: str | None) -> str:
    """Normalize text for exact matching (AlpsBench-style).

    - Lowercase
    - Collapse all whitespace to single spaces
    - Strip leading/trailing whitespace
    """
    if text is None:
        return ""
    return _WS_RE.sub(" ", str(text).strip().lower())


def canonical_tuple(item: dict[str, Any] | str) -> tuple[str, ...]:
    """Extract AlpsBench canonical 6-tuple for matching.

    AlpsBench matches on: (label, value, type, preference_attitude, time_scope, emotion)

    For string input (legacy), returns single-element tuple with normalized value.
    For dict input, returns full 6-tuple.
    """
    # Legacy string format: only compare value
    if isinstance(item, str):
        return (normalize_text(item),)

    # New dict format: compare full 6-tuple (AlpsBench style)
    return (
        normalize_text(item.get("label")),
        normalize_text(item.get("fact") or item.get("value")),  # Support both field names
        normalize_text(item.get("type")),
        normalize_text(item.get("preference_attitude")),
        normalize_text(item.get("time_scope")),
        normalize_text(item.get("emotion")),
    )


class ExactMatcher(FactMatcher):
    """Exact tuple matching (AlpsBench-style).

    Matches AlpsBench's evaluation methodology:
    - Extract canonical 6-tuple: (label, value, type, preference_attitude, time_scope, emotion)
    - Normalize each field (lowercase, collapse whitespace)
    - Check tuple equality

    For legacy string inputs, compares normalized values only.
    """

    name = "exact"

    async def match(self, expected: str | dict, stored: str | dict) -> MatchVerdict:
        """Check if canonical tuples are exactly equal.

        AlpsBench matching: ALL 6 fields must match for MATCH.
        PARTIAL is only for legacy string format (substring match).
        """
        expected_tuple = canonical_tuple(expected)
        stored_tuple = canonical_tuple(stored)

        if expected_tuple == stored_tuple:
            return MatchVerdict.MATCH

        # For AlpsBench dicts: strict tuple matching, no partial
        # AlpsBench uses Counter intersection - only exact tuples count
        if isinstance(expected, dict) and isinstance(stored, dict):
            return MatchVerdict.NO_MATCH

        # For legacy strings: allow substring partial match
        if isinstance(expected, str) and isinstance(stored, str):
            if expected_tuple[0] and stored_tuple[0]:
                if expected_tuple[0] in stored_tuple[0] or stored_tuple[0] in expected_tuple[0]:
                    return MatchVerdict.PARTIAL

        return MatchVerdict.NO_MATCH

    async def close(self) -> None:
        """No cleanup needed for exact matching."""
        pass

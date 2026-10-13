"""AlpsBench fuzzy matcher.

Implements the pairwise similarity scoring described in:
https://github.com/ThisIsCosine/AlpsBench/blob/main/docs/evaluation_metric.md

Pair score = 0.40 * label_similarity
           + 0.40 * value_similarity
           + 0.10 * type_match
           + 0.10 * confidence_score

Pairs with score >= 0.30 are eligible for 1-to-1 matching via Hungarian algorithm.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

from .base import FactMatcher, MatchVerdict

_WS_RE = re.compile(r"\s+")


def normalize_text(text: str | None) -> str:
    """AlpsBench normalization: lowercase, strip, collapse whitespace."""
    if text is None:
        return ""
    return _WS_RE.sub(" ", str(text).strip().lower())


def label_similarity(label_a: str, label_b: str) -> float:
    """Hierarchical path similarity.

    Labels like "Preferences/Food" are treated as paths.
    Score = 2 * LCP / (len(path_a) + len(path_b))
    where LCP is the longest common prefix of path segments.
    """
    if not label_a or not label_b:
        return 0.0

    parts_a = label_a.split("/")
    parts_b = label_b.split("/")

    # Find longest common prefix
    lcp = 0
    for a, b in zip(parts_a, parts_b):
        if a == b:
            lcp += 1
        else:
            break

    total_len = len(parts_a) + len(parts_b)
    if total_len == 0:
        return 0.0

    return 2.0 * lcp / total_len


def token_jaccard(text_a: str, text_b: str) -> float:
    """Token-level Jaccard similarity."""
    if not text_a or not text_b:
        return 0.0

    tokens_a = set(text_a.split())
    tokens_b = set(text_b.split())

    if not tokens_a or not tokens_b:
        return 0.0

    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b

    return len(intersection) / len(union) if union else 0.0


def sequence_match_ratio(text_a: str, text_b: str) -> float:
    """Character-level sequence matching (like difflib)."""
    if not text_a or not text_b:
        return 0.0

    return SequenceMatcher(None, text_a, text_b).ratio()


def value_similarity(value_a: str, value_b: str) -> float:
    """Blended lexical and character similarity.

    Default formula: 0.6 * token_jaccard + 0.4 * sequence_match_ratio
    """
    jaccard = token_jaccard(value_a, value_b)
    seq_match = sequence_match_ratio(value_a, value_b)

    return 0.6 * jaccard + 0.4 * seq_match


def type_match(type_a: str, type_b: str) -> float:
    """Exact type match: 1.0 if equal, else 0.0."""
    return 1.0 if type_a == type_b else 0.0


def confidence_score(conf_a: float, conf_b: float) -> float:
    """Confidence similarity with penalty for large differences.

    No penalty when |conf_a - conf_b| <= 0.2
    Linear penalty after that until score reaches 0.
    """
    diff = abs(conf_a - conf_b)

    if diff <= 0.2:
        return 1.0

    # Linear penalty: at diff=1.0, score=0
    # score = 1 - (diff - 0.2) / 0.8
    penalty = (diff - 0.2) / 0.8
    score = max(0.0, 1.0 - penalty)

    return score


def pair_score(
    item_a: dict[str, Any],
    item_b: dict[str, Any],
) -> float:
    """Compute AlpsBench pair score between two memory items.

    Weights:
    - 0.40: label similarity
    - 0.40: value similarity
    - 0.10: type match
    - 0.10: confidence score
    """
    # Normalize fields - handle both ground truth and model output formats
    # Ground truth: {label, fact, type}
    # Model output: {label, value, type, confidence}
    label_a = normalize_text(item_a.get("label", ""))
    label_b = normalize_text(item_b.get("label", ""))

    value_a = normalize_text(item_a.get("fact") or item_a.get("value", ""))
    value_b = normalize_text(item_b.get("value") or item_b.get("fact", ""))

    # Handle both FactType enum and string
    type_a_raw = item_a.get("type", "direct")
    type_b_raw = item_b.get("type", "direct")
    type_a = normalize_text(str(type_a_raw).lower() if hasattr(type_a_raw, 'value') else str(type_a_raw))
    type_b = normalize_text(str(type_b_raw).lower() if hasattr(type_b_raw, 'value') else str(type_b_raw))

    conf_a = float(item_a.get("confidence", 0.8))
    conf_b = float(item_b.get("confidence", 0.8))

    # Compute similarities
    label_sim = label_similarity(label_a, label_b)
    value_sim = value_similarity(value_a, value_b)
    type_sim = type_match(type_a, type_b)
    conf_sim = confidence_score(conf_a, conf_b)

    # Weighted sum
    score = (
        0.40 * label_sim +
        0.40 * value_sim +
        0.10 * type_sim +
        0.10 * conf_sim
    )

    return score


def greedy_matching(
    expected_items: list[dict],
    stored_items: list[dict],
    min_score: float = 0.30,
) -> tuple[list[tuple[int, int, float]], set[int], set[int]]:
    """Greedy 1-to-1 matching.

    Returns:
        matches: list of (expected_idx, stored_idx, score)
        matched_expected: set of matched expected indices
        matched_stored: set of matched stored indices
    """
    # Compute all pair scores
    scores = []
    for i, exp_item in enumerate(expected_items):
        for j, stored_item in enumerate(stored_items):
            score = pair_score(exp_item, stored_item)
            if score >= min_score:
                scores.append((score, i, j))

    # Sort by score descending
    scores.sort(reverse=True, key=lambda x: x[0])

    # Greedy assignment
    matches = []
    matched_expected = set()
    matched_stored = set()

    for score, i, j in scores:
        if i not in matched_expected and j not in matched_stored:
            matches.append((i, j, score))
            matched_expected.add(i)
            matched_stored.add(j)

    return matches, matched_expected, matched_stored


class AlpsBenchMatcher(FactMatcher):
    """AlpsBench fuzzy matcher with pairwise similarity scoring.

    Uses weighted similarity scoring:
    - Label: hierarchical path LCP
    - Value: 0.6*Jaccard + 0.4*sequence_match
    - Type: exact match
    - Confidence: penalty for large differences

    Accepts pairs with score >= 0.30 and uses greedy 1-to-1 matching.
    """

    name = "alpsbench"

    def __init__(self, min_pair_score: float = 0.30):
        self.min_pair_score = min_pair_score
        self.pair_scores = []  # Track for debugging
        self.last_pair_score: float | None = None  # Last computed score

    async def match(self, expected: str | dict, stored: str | dict) -> MatchVerdict:
        """Match a single pair using AlpsBench similarity scoring.

        For single-pair matching, we compute the pair score and
        return MATCH if >= threshold, else NO_MATCH.

        For batch evaluation (matching multiple expected vs multiple stored),
        use match_batch() instead.
        """
        # Convert to dicts if strings
        if isinstance(expected, str):
            expected = {"fact": expected, "type": "direct", "confidence": 0.8}
        if isinstance(stored, str):
            stored = {"value": stored, "type": "direct", "confidence": 0.8}

        score = pair_score(expected, stored)
        self.pair_scores.append(score)
        self.last_pair_score = score  # Store for retrieval

        if score >= self.min_pair_score:
            if score >= 0.8:  # High confidence
                return MatchVerdict.MATCH
            else:  # Medium confidence
                return MatchVerdict.PARTIAL

        return MatchVerdict.NO_MATCH

    def get_last_pair_score(self) -> float | None:
        """Get the last computed pair score for tracking in evaluation."""
        return self.last_pair_score

    async def close(self) -> None:
        """No cleanup needed."""
        pass

from __future__ import annotations

from .base import FactMatcher, MatchVerdict

DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_MATCH_THRESHOLD = 0.75
DEFAULT_PARTIAL_THRESHOLD = 0.55


class EmbeddingMatcher(FactMatcher):
    """Cosine similarity via sentence-transformers. Runs locally, no API calls."""

    name = "embedding"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        match_threshold: float = DEFAULT_MATCH_THRESHOLD,
        partial_threshold: float = DEFAULT_PARTIAL_THRESHOLD,
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for the embedding matcher.\n"
                "Install it with: pip install sentence-transformers"
            ) from None

        self.model = SentenceTransformer(model_name)
        self.match_threshold = match_threshold
        self.partial_threshold = partial_threshold

    async def match(self, expected: str, stored: str) -> MatchVerdict:
        embeddings = self.model.encode([expected, stored], normalize_embeddings=True)
        score = float(embeddings[0] @ embeddings[1])

        if score >= self.match_threshold:
            return MatchVerdict.MATCH
        if score >= self.partial_threshold:
            return MatchVerdict.PARTIAL
        return MatchVerdict.NO_MATCH

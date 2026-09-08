# Hybrid Matcher

Two-stage fact matching: embedding filter + LLM for uncertain cases.

## Strategy

The hybrid matcher combines the speed of embedding similarity with the semantic understanding of LLM judgment:

1. **High confidence match** (similarity ≥ 0.75) → MATCH (skip LLM)
2. **High confidence mismatch** (similarity < 0.4) → NO_MATCH (skip LLM)
3. **Uncertain middle range** (0.4-0.75) → Escalate to LLM for semantic judgment

This reduces LLM calls by 60-80% while fixing embedding matcher's false negatives in the middle range.

## Why Use Hybrid?

**Embedding matcher limitations:**
- Fast and local (no API costs)
- BUT: Marks semantically equivalent facts as NO_MATCH when verbosity differs
- Example: Expected "We have the entire semester..." vs Stored "full semester..." → 0.46 similarity → NO_MATCH
- Reality: These are the same fact, just different phrasing

**LLM matcher limitations:**
- Better semantic understanding
- BUT: Slow and expensive (API calls for every comparison)
- Hundreds of comparisons per evaluation → high cost

**Hybrid solution:**
- Use embeddings for obvious cases (most comparisons)
- Only pay for LLM judgment on uncertain cases
- Best of both worlds: speed + accuracy

## Usage

```bash
llm-memory-bench evaluate \
  results/run_*.json \
  --matcher hybrid \
  --judge-provider anthropic \
  --judge-model claude-sonnet-4-20250514
```

Like the LLM matcher, hybrid requires `--judge-provider` and `--judge-model` for the escalation cases.

## Statistics

The hybrid matcher tracks and reports escalation statistics:

```
Hybrid matcher stats: 42/200 escalated to LLM (21.0%)
```

This shows:
- Total comparisons: 200
- LLM escalations: 42
- Embedding-only decisions: 158 (79%)

Lower escalation rates = more cost savings.

## Tuning Thresholds

Default thresholds:

```python
confident_match_threshold = 0.75      # Above this → MATCH without LLM
confident_mismatch_threshold = 0.4    # Below this → NO_MATCH without LLM
```

To adjust (requires code change):

```python
from llm_memory_bench.matchers.hybrid import HybridMatcher

matcher = HybridMatcher(
    provider=provider,
    confident_match_threshold=0.80,    # More conservative → more LLM calls
    confident_mismatch_threshold=0.35  # Less conservative → fewer LLM calls
)
```

**Wider gap (0.35-0.80):** More LLM escalations, higher cost, higher accuracy  
**Narrower gap (0.45-0.70):** Fewer LLM escalations, lower cost, may miss edge cases

## Performance Comparison

Based on typical evaluation workloads:

| Matcher | Speed | Cost | Accuracy | False Negatives |
|---------|-------|------|----------|-----------------|
| Embedding | Fast (local) | $0 | Good for exact matches | High (verbosity differences) |
| LLM | Slow (API) | High ($5-20/eval) | Best | Low |
| Hybrid | Medium | Low ($1-4/eval) | Near-LLM quality | Low |

**When to use each:**

- **Embedding:** Quick iterations, tight verbosity control in ground truth
- **LLM:** Final evaluation, research publications, when accuracy > cost
- **Hybrid:** Production evaluations, CI/CD pipelines, cost-aware workflows

## Implementation Details

### Two-Stage Flow

```python
async def match(self, expected: str, stored: str) -> MatchVerdict:
    # Stage 1: Compute embedding similarity (fast, local)
    embeddings = self.embedding_model.encode([expected, stored])
    similarity = embeddings[0] @ embeddings[1]
    
    # Early exit for confident cases
    if similarity >= 0.75:
        return MatchVerdict.MATCH
    if similarity < 0.4:
        return MatchVerdict.NO_MATCH
    
    # Stage 2: LLM judgment for uncertain cases
    return await self._llm_judge(expected, stored)
```

### Escalation Tracking

The matcher maintains counters:
- `total_comparisons`: All match() calls
- `llm_escalations`: Cases that reached Stage 2

These are reported via `matcher.close()` at the end of evaluation.

### Memory Model

The embedding model (all-MiniLM-L6-v2) is loaded once per matcher instance and reused across all comparisons. LLM calls are made on-demand only for uncertain cases.

## Cost Estimation

Typical evaluation scenario:
- 10 conversations
- 30 expected facts
- 50 stored tool calls
- ~300 comparisons (each expected fact vs each stored fact)

**Embedding matcher:** $0 (local computation)

**LLM matcher:**
- 300 comparisons × 2 API calls (judge prompt) = 600 calls
- ~$15-20 (depending on model)

**Hybrid matcher:**
- 300 comparisons
- ~60 escalations (20% rate)
- 60 × 2 API calls = 120 calls
- ~$3-5 (depending on model)

**Savings:** 75-80% cost reduction vs pure LLM matcher

## Known Limitations

1. **Fixed thresholds:** Not adaptive to dataset characteristics
2. **Binary escalation:** Either full LLM judgment or none (no "partial LLM" mode)
3. **No caching:** Repeated comparisons of identical pairs don't reuse results
4. **Embedding model fixed:** Can't switch to domain-specific embeddings

## Future Improvements

Possible enhancements:

1. **Adaptive thresholds:** Learn optimal thresholds from labeled data
2. **Tiered LLM:** Use fast/cheap model (Haiku) for initial escalation, full model (Sonnet) only for complex cases
3. **Comparison caching:** Store (expected, stored, verdict) triples to avoid re-judging
4. **Embedding fine-tuning:** Train on memory-specific semantic equivalence
5. **Confidence scores:** LLM returns confidence (0-1) instead of binary verdict
6. **Batch escalation:** Collect uncertain cases and judge in single multi-turn prompt

## Related

- [Embedding Matcher](../src/llm_memory_bench/matchers/embedding.py) - Fast local similarity
- [LLM Matcher](../src/llm_memory_bench/matchers/llm.py) - Semantic judgment
- [Evaluation Guide](evaluation.md) - How evaluation works

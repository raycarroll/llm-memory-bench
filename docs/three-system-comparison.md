# Three-System Comparison: BFCL vs MemoryHub vs GBrain

Benchmark run on 2026-08-31 with:
- **Model**: claude-sonnet-4-5@20250929
- **Dataset**: 10 conversations, 50 expected facts (cumulative ground truth)
- **Matcher**: hybrid (embedding filter + LLM for uncertain cases)

## Results Summary

| Metric | BFCL | MemoryHub | GBrain | Winner |
|--------|------|-----------|--------|--------|
| **Recall** | 80.0% | 55.0% | **85.0%** ★ | GBrain |
| **Precision** | 19.1% | **21.6%** ★ | 7.1% | MemoryHub |
| **F1 Score** | 30.8% | **31.0%** ★ | 13.1% | MemoryHub |
| **Tool Calls** | 211 | **138** ★ | 606 | MemoryHub |
| **Overhead** | 4.2x | **2.8x** ★ | 12.1x | MemoryHub |
| **False Positives** | 170 | **100** ★ | 557 | MemoryHub |

### Raw Numbers

| System | True Positives | False Negatives | False Positives | Total Calls |
|--------|----------------|-----------------|-----------------|-------------|
| BFCL | 40.0 | 10.0 | 170 | 211 |
| MemoryHub | 27.5 | 22.5 | 100 | 138 |
| GBrain | 42.5 | 7.5 | 557 | 606 |

## Storage Behavior Characterization

### BFCL: Aggressive Storage
**Prompt**: "Actively manage important information"

**Strategy**: Over-stores due to vague guidance
- ✓ Captured 80% of important facts (40/50)
- ✗ 81% of stored facts were noise
- ✗ 4.2x overhead (211 calls for 50 facts)

**Why**: No clear DO/DON'T lists → model errs on side of storing too much

### MemoryHub: Balanced Filtering
**Prompt**: Explicit DO/DON'T lists + hygiene rules

**Strategy**: Most efficient overall
- ~ Captured 55% of important facts (27.5/50)
- ✓ 22% precision (least noisy)
- ✓ 2.8x overhead (138 calls)
- ✓ Best F1 score (31%)

**Why**: Explicit filtering guidance → better precision but misses some facts

### GBrain: Store Everything
**Prompt**: "Brain-first protocol: write back on every mention"

**Strategy**: Extreme over-storage by design
- ✓ Captured 85% of important facts (42.5/50) - **BEST RECALL**
- ✗ 7% precision - **WORST PRECISION**
- ✗ 12.1x overhead (606 calls) - **MOST EXPENSIVE**
- ✗ 93% of stored facts were noise

**Why**: Philosophy is "storage cheap, search powerful" → store everything, filter during retrieval

## Key Findings

### 1. Prompt Specificity Inversely Correlates with Storage Volume

```
Vague (BFCL):           211 calls,  19% precision
Explicit (MemoryHub):   138 calls,  22% precision  ← Best balance
Everything (GBrain):    606 calls,   7% precision  ← Assumes search filters
```

### 2. All Systems Find Facts, Difference is in Filtering

**Recall ranges**: 55-85% (only 30 percentage point spread)

**Precision ranges**: 7-22% (3x variation)

**Conclusion**: Models can identify what's important. The prompt determines what they DO with that knowledge.

### 3. GBrain's Design Assumption

**Assumption**: "Storage is cheap, search is powerful, so store everything and let retrieval filter"

**Reality**:
- ✓ Highest recall (85%)
- ✗ Worst precision (7%)
- ✗ 12x storage overhead
- ✗ Stores 11 noise facts per 1 real fact

**Unknown**: Does GBrain's search effectively filter the 557 false positives?

Without retrieval simulation, we can't test if the design assumption holds.

## Storage Cost Analysis

For 50 expected facts:

| System | Total Calls | Overhead | Noise Calls | Efficiency |
|--------|-------------|----------|-------------|------------|
| GBrain | 606 | 1,112% | 557 (92%) | Worst |
| BFCL | 211 | 322% | 170 (81%) | Poor |
| MemoryHub | 138 | 176% | 100 (72%) | Best |

**Interpretation**:
- MemoryHub stores 1.8 noise facts per 1 real fact
- BFCL stores 4.3 noise facts per 1 real fact
- GBrain stores 13.1 noise facts per 1 real fact

## Prompt Engineering Impact

### What Makes MemoryHub Most Balanced?

**Explicit DO lists**:
- "DO write preferences, decisions, architectural choices"
- "DO write tool configuration, workflow patterns"

**Explicit DON'T lists**:
- "Skip ephemeral things like 'user asked me to read a file'"
- "Don't write code patterns (derivable from code)"

**Weight guidance**:
- 1.0 = critical policy
- 0.8-0.9 = strong preference
- 0.5-0.7 = nice-to-know
- <0.5 = skip entirely

**Result**: Clear decision boundary → better filtering, lower overhead

### What Makes BFCL Noisy?

**Vague guidance**:
- "Actively manage your memory data"
- "Keep track of important information"

**No examples** of what to skip

**Result**: Ambiguity → over-storage "just in case"

### What Makes GBrain Extremely Noisy?

**Explicit "write everything" instruction**:
- "Before answering any question... Brain first"
- "Write back. When I make a decision, mention a new person/company, or state a preference, use remember()"

**No filtering guidance** at all

**Philosophy**: Defer filtering to retrieval time

**Result**: Comprehensive capture at massive storage cost

## When Each Approach Works

### BFCL (Vague, Aggressive)
**Good for**:
- Exploratory sessions where you don't know what matters yet
- When missing information is more costly than noise
- When storage is cheap and retrieval is good

**Bad for**:
- Production systems with storage costs
- When precision matters
- When context window limits matter

### MemoryHub (Explicit, Balanced)
**Good for**:
- Production agent systems
- When you have clear policies about what to remember
- When storage efficiency matters
- When you want predictable behavior

**Bad for**:
- Open-ended discovery
- When requirements change frequently
- When you can't define clear DO/DON'T lists upfront

### GBrain (Write Everything)
**Good for**:
- Personal knowledge bases where completeness > efficiency
- When retrieval is very good at filtering
- When the user manually curates memories later
- Research/discovery phases

**Bad for**:
- Production agents (too expensive)
- Token-limited contexts
- When precision matters
- Batch processing without human curation

## Implications for Memory System Design

### Storage vs Retrieval Tradeoff

GBrain's approach tests a hypothesis:
> "Better to store everything and rely on search than to filter during storage"

**For this to work**:
1. Search must handle 12x data volume efficiently
2. Retrieval must filter 93% noise without degrading quality
3. Storage cost must be negligible
4. Context windows must be large enough to handle noise

**We can't validate** without retrieval simulation in the benchmark.

### Prompt Engineering Lessons

1. **Vague prompts → over-storage**: Models default to storing too much when uncertain

2. **Explicit examples > abstract rules**: "Skip ephemeral" is clearer than "manage important information"

3. **Weight guidance helps**: Numeric thresholds (1.0, 0.8, 0.5) give clear decision boundaries

4. **Philosophy matters**: "Write everything" vs "Filter proactively" leads to 4x difference in overhead

## For BFCL Attribution Analysis

With GBrain's 85% storage recall:

If BFCL published end-to-end accuracy is 60%:
- Storage: 85% (from this benchmark)
- Retrieval + Generation: 60% / 85% = **71%**

So storage is NOT the bottleneck for GBrain-style systems - the challenge is in retrieval filtering and generation given noisy context.

## Recommendations

### For Benchmark Evolution

1. **Add retrieval simulation**: Test if stored facts are actually retrievable and useful

2. **Measure context efficiency**: Precision matters more when context is limited

3. **Test at scale**: Does GBrain's approach break at 1000 conversations?

### For System Selection

**Choose BFCL-style if**: You want simple prompts and don't mind noise

**Choose MemoryHub-style if**: You want production-ready efficiency and predictability

**Choose GBrain-style if**: You want maximum recall and have excellent retrieval + curation

## Related Files

- **Evaluation Results**:
  - BFCL: `results/eval_vertex_claude-sonnet-4-5-20250929_bfcl_memory_kv_20260831-102729_hybrid_20260831-105305.json`
  - MemoryHub: `results/eval_vertex_claude-sonnet-4-5-20250929_memoryhub_20260831-105936_hybrid_20260831-112333.json`
  - GBrain: `results/eval_vertex_claude-sonnet-4-5-20250929_gbrain_20260831-141402_hybrid_20260831-144957.json`

- **Prompt Sources**:
  - BFCL: `prompts/bfcl_memory_kv/v4_2025-07.yaml`
  - MemoryHub: `prompts/memoryhub/v1_lazy_2026-08.yaml`
  - GBrain: `prompts/gbrain/v0.46.19_2024-12.yaml`

- **Deep Dives**:
  - [MemoryHub Prompts Analysis](memoryhub-prompts-deep-dive.md)
  - [Prompt Versioning Architecture](prompt-versioning.md)
  - [Hybrid Matcher](hybrid-matcher.md)

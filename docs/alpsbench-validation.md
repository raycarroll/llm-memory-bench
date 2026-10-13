# Validating Against AlpsBench Results

## Goal

Replicate AlpsBench's ~33% F1 to validate our implementation and evaluation methodology.

## Approach

We approximate AlpsBench's batch extraction using:
1. **Cumulative evaluation** (ignores timing, matches whole-conversation facts)
2. **Exact matcher** (normalized string equality, like AlpsBench)
3. **Their prompt** (via `alpsbench_batch` system)

**Note:** Not a perfect replication because:
- AlpsBench shows full conversation upfront (batch)
- We still do turn-by-turn (but cumulative eval ignores timing)
- Close enough for validation purposes

## Steps

### 1. Run with AlpsBench Batch System

```bash
llm-memory-bench run \
  --dataset datasets/converted/alpsbench-task1.yaml \
  --system alpsbench_batch \
  --provider openai \
  --model gpt-4o \
  --max-conversations 10
```

**What this does:**
- Uses AlpsBench's exact prompt
- Uses their taxonomy labels
- High recall focus
- Still goes turn-by-turn (limitation of our runner)

### 2. Evaluate with Exact Matching + Cumulative Mode

```bash
# For per-turn dataset, evaluate cumulatively
llm-memory-bench evaluate results/run_*_alpsbench_batch.json \
  --matcher exact \
  --dataset datasets/converted/alpsbench-task1.yaml
```

**What this does:**
- Uses exact normalized string matching (AlpsBench-style)
- Cumulative evaluation if dataset supports it
- Ignores timing, just checks: were the facts stored somewhere?

### 3. Compare to AlpsBench Baseline

**Expected results:**
- AlpsBench paper: **GPT-4.1-mini = 33.69% F1**
- Our replication target: ~30-35% F1 (accounting for minor methodology differences)

**If we get significantly different results:**
- Check matching logic
- Check normalization
- Check dataset filtering (we remove confidence < 0.5)

## Implementation Details

### Exact Matcher

Matches AlpsBench's `_canonical_memory_item`:

```python
def normalize_text(text: str) -> str:
    # Lowercase
    # Collapse whitespace to single spaces
    # Strip
    return re.sub(r"\s+", " ", text.strip().lower())

def match(expected, stored):
    return normalize_text(expected) == normalize_text(stored)
```

### AlpsBench Batch System

Uses their exact prompt from `prompts/task1_extract.txt`:
- "Extract candidate long-term user memories with high recall"
- Taxonomy labels (Preferences/Food, Personal_Background/Education, etc.)
- Confidence levels (0.95, 0.80, 0.65, 0.50)
- Evidence text requirement

### Differences from True AlpsBench

| Aspect | AlpsBench | Our Approximation |
|--------|-----------|-------------------|
| Input | Full conversation JSON upfront | Turn-by-turn streaming |
| Extraction | Single batch call | Can call at any turn |
| Evaluation | Cumulative (timing irrelevant) | Cumulative (timing irrelevant) ✓ |
| Matching | Exact normalized strings | Exact normalized strings ✓ |
| Output | JSON with memory_items | Tool calls with arguments ✓ |

**Key:** Cumulative evaluation makes timing irrelevant, so the turn-by-turn vs batch difference shouldn't affect F1 significantly.

## Validation Targets

### Full Baseline Comparison

From AlpsBench README:

| Model | Task 1 F1 | Our Target |
|-------|-----------|------------|
| Gemini-3 Flash | 51.67% | ~48-52% |
| DeepSeek Reasoner | 47.79% | ~44-48% |
| GPT-5.2 | 41.43% | ~38-42% |
| Claude-Sonnet-4.5 | 41.64% | ~38-42% |
| **GPT-4.1-mini** | **33.69%** | **~30-35%** |
| Llama-4 Maverick | 22.07% | ~20-24% |

**If we match within ±3%:** Implementation validated ✓

**If we're >5% off:** Investigation needed

## What This Validates

### If Results Match (~33% F1)

✅ **Our evaluation logic is correct**
- Matching works
- F1 calculation correct
- Dataset processing correct

✅ **The performance gap is real**
- 33% (batch) vs 11% (proactive) = task difficulty
- Not an implementation bug

### If Results Don't Match

Possible issues:
1. **Matching too strict/lenient** - adjust exact matcher
2. **Dataset filtering** - check confidence threshold
3. **Normalization** - check string processing
4. **Tool extraction** - check how we parse tool arguments

## Next: Quantify Task Difficulty

Once validated, compare:

```bash
# Batch extraction (AlpsBench-style)
llm-memory-bench run --system alpsbench_batch --max-conversations 50
llm-memory-bench evaluate results/run_*.json --matcher exact
# Expected: ~33% F1

# Proactive tool-calling (our methodology)
llm-memory-bench run --system simple --max-conversations 50
llm-memory-bench evaluate results/run_*.json --matcher embedding
# Expected: ~11% F1

# Difference: 33% - 11% = 22% = cost of proactive real-time judgment
```

This quantifies exactly how much harder proactive judgment is.

## Usage Example

```bash
# Full validation workflow
cd /Users/rcarroll/Documents/code/llm_memory_bench

# 1. Run batch extraction
llm-memory-bench run \
  --dataset datasets/converted/alpsbench-task1.yaml \
  --system alpsbench_batch \
  --provider openai \
  --model gpt-4o-mini \
  --max-conversations 10

# 2. Evaluate with exact matching
llm-memory-bench evaluate results/run_*_alpsbench_batch.json \
  --matcher exact

# 3. Check F1 score
cat results/eval_*.json | jq '.metrics.extraction_f1'

# Expected: ~0.30-0.35 (30-35%)
# If close to AlpsBench's 33.69%: VALIDATED ✓
```

## Troubleshooting

### Issue: F1 much lower than expected

**Check:**
1. Is confidence filtering applied? (we filter < 0.5)
2. Is language filtering active? (English only by default)
3. Are tool calls being parsed correctly?

**Debug:**
```bash
# Check what was actually stored
cat results/run_*.json | jq '.conversations[0].turns[] | select(.tool_calls) | .tool_calls'

# Check ground truth
cat datasets/converted/alpsbench-task1.yaml | grep "should_store:" -A 3 | head -20
```

### Issue: F1 much higher than expected

**Check:**
1. Is exact matching too lenient? (substring matches counted as full matches?)
2. Is cumulative vs per-turn evaluation correct?

**Debug:**
```bash
# Check matcher verdicts
cat results/eval_*.json | jq '.conversations[0].turns[].fact_matches[] | {expected, verdict}'
```

## References

- AlpsBench repo: https://github.com/ThisIsCosine/AlpsBench
- AlpsBench prompt: `prompts/task1_extract.txt`
- AlpsBench metrics: `src/benchmark/metrics.py` (`_multiset_f1`, `_canonical_memory_item`)

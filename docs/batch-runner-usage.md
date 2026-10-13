# Batch Runner Usage - AlpsBench Validation

## What It Does

The batch runner replicates AlpsBench's methodology:
1. **Sends entire conversation as JSON** in a single API call
2. **No tool calls** - model returns pure JSON
3. **Batch extraction** - model sees everything upfront

## Quick Start

```bash
# Run AlpsBench-style batch extraction
llm-memory-bench run \
  --dataset datasets/converted/alpsbench-task1.yaml \
  --system alpsbench_batch \
  --provider openai \
  --model gpt-4o \
  --batch \
  --max-conversations 10

# Evaluate with exact matching (AlpsBench-style)
llm-memory-bench evaluate results/run_*_alpsbench_batch.json \
  --matcher exact

# Check F1 - should be ~33% (vs AlpsBench's 33.69% for GPT-4.1-mini)
cat results/eval_*.json | jq '.metrics'
```

## Validation Goals

### Primary Goal: Validate Our Implementation

**If we get ~30-35% F1:**
- ✅ Our evaluation logic is correct
- ✅ Our matching works properly
- ✅ Our dataset processing is sound

**If we're off by >5%:**
- ❌ Investigation needed
- Check matching algorithm
- Check normalization
- Check dataset filtering

### Secondary Goal: Quantify Task Difficulty

Once validated, compare batch vs proactive:

```bash
# Batch (easier)
llm-memory-bench run --batch --system alpsbench_batch ...
# Expected: ~33% F1

# Proactive (harder)
llm-memory-bench run --system simple ...
# Expected: ~11% F1

# Difference: 22 percentage points = cost of proactive judgment
```

## How Batch Mode Works

### Input Format

```python
# Batch runner sends:
{
  "dialogue": [
    {"role": "user", "text": "I prefer Python for scripting"},
    {"role": "assistant", "text": "Great! Here's a Python script..."},
    {"role": "user", "text": "I live in Boston"},
    # ... ALL turns at once
  ]
}
```

### Output Format

```json
{
  "memory_items": [
    {
      "type": "direct",
      "label": "Preferences/Programming",
      "value": "Prefers Python for scripting",
      "confidence": 0.95,
      "evidence_text": "I prefer Python for scripting"
    },
    {
      "type": "direct", 
      "label": "Personal_Background/Location",
      "value": "Lives in Boston",
      "confidence": 0.95,
      "evidence_text": "I live in Boston"
    }
  ]
}
```

### Conversion to Tool Calls

For evaluation compatibility, we convert each memory_item to a pseudo-tool-call:

```python
# Each item becomes:
ToolCall(
  name="batch_memory_item",
  arguments={
    "type": "direct",
    "label": "Preferences/Programming",
    "value": "Prefers Python for scripting",
    "confidence": 0.95,
    "evidence_text": "..."
  }
)

# Then extract_stored_fact() returns item["value"]
```

## Comparison Table

| Aspect | AlpsBench Original | Our Batch Runner | Our Proactive |
|--------|-------------------|------------------|---------------|
| **Input** | Full conversation JSON | Full conversation JSON ✓ | Turn-by-turn |
| **API calls** | 1 per conversation | 1 per conversation ✓ | N (one per turn) |
| **Model sees** | Everything upfront | Everything upfront ✓ | Incremental history |
| **Output** | JSON (no tools) | JSON (no tools) ✓ | Tool calls |
| **Timing** | Irrelevant | Irrelevant ✓ | Critical |
| **Task** | Find facts in text | Find facts in text ✓ | Proactive recognition |

## Baseline Comparison

AlpsBench paper results (Task 1 Extraction):

| Model | AlpsBench F1 | Our Target |
|-------|--------------|------------|
| Gemini-3 Flash | 51.67% | ~48-52% |
| DeepSeek Reasoner | 47.79% | ~44-48% |
| GPT-5.2 | 41.43% | ~38-42% |
| Claude-Sonnet-4.5 | 41.64% | ~38-42% |
| Qwen3-max | 39.01% | ~36-40% |
| **GPT-4.1-mini** | **33.69%** | **~30-35%** |
| Llama-4 Maverick | 22.07% | ~20-24% |

## Example Session

```bash
# Full validation workflow
cd /Users/rcarroll/Documents/code/llm_memory_bench

# 1. Run batch extraction (10 conversations for speed)
llm-memory-bench run \
  --dataset datasets/converted/alpsbench-task1.yaml \
  --system alpsbench_batch \
  --provider openai \
  --model gpt-4o-mini \
  --batch \
  --max-conversations 10

# Output shows:
# [1/10] sess_0517169e56bb (33 turns) [batch mode]
#   done — 4 memories extracted, 45123 tokens

# 2. Evaluate with exact matching
llm-memory-bench evaluate results/run_*_alpsbench_batch.json \
  --matcher exact

# 3. Check results
cat results/eval_*.json | jq '{
  f1: .metrics.extraction_f1,
  precision: .metrics.extraction_precision,
  recall: .metrics.extraction_recall,
  total_calls: .metrics.total_tool_calls
}'

# Expected output:
# {
#   "f1": 0.32,       # Close to AlpsBench's 33.69%
#   "precision": 0.35,
#   "recall": 0.30,
#   "total_calls": 42
# }
```

## Flags

### --batch
**Type:** Flag  
**Description:** Enable batch extraction mode

**What it changes:**
- ✅ Sends full conversation as JSON (not turn-by-turn)
- ✅ No tools exposed to model
- ✅ Model returns JSON directly
- ✅ One API call per conversation (not per turn)

**Use with:**
- `--system alpsbench_batch` (uses AlpsBench prompt)
- `--matcher exact` (AlpsBench-style matching)

### Required for Validation

```bash
# These 3 flags together replicate AlpsBench:
--batch                    # Batch input/output
--system alpsbench_batch   # AlpsBench prompt
--matcher exact            # Exact string matching

# Plus standard flags:
--provider openai
--model gpt-4o-mini
--max-conversations 10
```

## Troubleshooting

### Issue: Low F1 (~10-15% instead of ~33%)

**Possible causes:**
1. Model not understanding batch format
2. JSON parsing failed
3. Matching too strict

**Debug:**
```bash
# Check what was extracted
cat results/run_*.json | jq '.conversations[0].turns[] | select(.tool_calls) | .tool_calls[0].arguments'

# Should see AlpsBench-style memory items:
# {"type": "direct", "label": "...", "value": "...", "confidence": 0.95}
```

### Issue: No memories extracted

**Check:**
```bash
# Look at raw model response (would need to add logging)
# For now, check if tool_calls exist:
cat results/run_*.json | jq '[.conversations[].turns[].tool_calls // [] | length] | add'

# Should be >0
```

**Possible causes:**
- JSON parsing failed
- Model didn't understand prompt
- Model returned markdown code blocks (should be handled, but check)

### Issue: F1 much higher than AlpsBench (~50%+)

**Check:**
- Are we using --matcher exact? (embedding matcher is more lenient)
- Is dataset filtering correct? (we filter confidence < 0.5)

## Next Steps After Validation

Once batch mode is validated (~33% F1):

### 1. Compare Proactive vs Batch

```bash
# Run both modes on same data
llm-memory-bench run --batch --system alpsbench_batch --max-conversations 50
llm-memory-bench run --system simple --max-conversations 50

# Compare F1 scores
# Batch: ~33%
# Proactive: ~11%  
# Gap: ~22% = difficulty of proactive real-time judgment
```

### 2. Test Claude vs GPT-4o

```bash
# Claude batch
llm-memory-bench run --batch --system alpsbench_batch \
  --provider anthropic --model claude-sonnet-4-20250514

# Claude proactive
llm-memory-bench run --system simple \
  --provider anthropic --model claude-sonnet-4-20250514

# Compare if Claude handles proactive better
```

### 3. Test Prompt Variations

```bash
# Try different prompts in proactive mode
llm-memory-bench run --system simple ...
llm-memory-bench run --system behavioral_relevance ...
llm-memory-bench run --system memoryhub ...

# See which prompt design works best for proactive judgment
```

## References

- AlpsBench repo: https://github.com/ThisIsCosine/AlpsBench
- AlpsBench prompt: `prompts/task1_extract.txt`
- AlpsBench metrics: `src/benchmark/metrics.py`

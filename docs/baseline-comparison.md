# Baseline Comparison: Tool-Calling vs Memory Judgment

## Overview

This benchmark measures **proactive memory judgment** — whether models know *when* to store information. To isolate this from general tool-calling ability, we compare against BFCL.

## The Judgment Hierarchy

Performance depends on three orthogonal capabilities:

```
1. Tool execution     ← BFCL (live_simple, etc.) tests this
   ↓
2. Proactive storage  ← Both BFCL Memory and llm-memory-bench test this
   ↓  
3. Retrieval          ← BFCL Memory tests this (Phase 2)
```

**Key difference:** BFCL Memory measures end-to-end (black box), llm-memory-bench measures storage quality (white box).

### 1. Tool Execution (BFCL baseline)

**What it tests:** Can the model call functions correctly when explicitly requested?

- Choosing the right function from multiple options
- Extracting arguments correctly from the request
- Knowing when NOT to call (irrelevance detection)

**How to run:**

```bash
# Install BFCL
pip install bfcl

# Generate responses for Claude Sonnet 4
bfcl generate --model claude-sonnet-4-20250514 \
  --test-category live_simple

# Evaluate (uses AST-based matching)
bfcl evaluate --model claude-sonnet-4-20250514 \
  --test-category live_simple

# View results
cat ~/.cache/bfcl/result/claude-sonnet-4-20250514/live_simple_score.json
```

**Datasets:**
- `live_simple`: Single function call (257 examples)
- `live_multiple`: Sequential calls (100 examples)  
- `live_parallel`: Parallel calls (140 examples)
- `live_irrelevance`: Should NOT call (400 examples)

**Metrics:**
- Accuracy (exact match of function + arguments via AST)
- Type correctness
- Irrelevance detection rate

### 2. Semantic Extraction (future: prompted AlpsBench)

**What it tests:** Can the model extract the right facts when explicitly asked?

**Not yet implemented.** This would involve:
- Adding explicit "Remember that..." prompts before each ground-truth fact in AlpsBench
- Testing extraction accuracy without proactive judgment
- Using the same evaluation (LLM/embedding matcher) as the main benchmark

**Rationale:** Isolates semantic understanding from judgment timing.

### 3. Proactive Storage + Retrieval (BFCL Memory)

**What it tests:** Can the model store facts during conversation, then retrieve them later?

```bash
# Phase 1: Model proactively stores during prerequisite conversations
# Phase 2: Model retrieves facts to answer questions (no dialogue history)
bfcl generate --model claude-sonnet-4-20250514 --test-category memory_kv
bfcl evaluate --model claude-sonnet-4-20250514 --test-category memory_kv
```

**Metrics:**
- Answer accuracy (end-to-end black box)
- Tests 3 storage backends: KV, vector, recursive summarization

**Key characteristics:**
- ✅ End-to-end: storage + snapshot + reload + retrieval
- ✅ Multi-scenario: student, healthcare, finance, customer, notetaker
- ❌ Ambiguous failures: can't distinguish storage vs retrieval issues

### 4. Storage Quality Isolation (llm-memory-bench)

**What it tests:** What facts does the model actually store (white-box diagnostics)?

```bash
llm-memory-bench run \
  --dataset datasets/converted/alpsbench-task1.yaml \
  --provider anthropic \
  --model claude-sonnet-4-20250514 \
  --system memoryhub

llm-memory-bench evaluate results/[run-file].json \
  --matcher embedding
```

**Metrics:**
- Extraction precision/recall/F1 (specific failure modes)
- Per-turn ground truth comparison
- Semantic matching (paraphrasing okay)

**Key characteristics:**
- ✅ White-box: inspect what was stored vs what should have been
- ✅ Noise resistance: ~90% of turns don't warrant storage
- ✅ System comparison: test different prompts/schemas
- ❌ No retrieval testing: assumes perfect recall

## Interpreting Score Gaps

### Case 1: Storage problem (low recall)
```
BFCL live_simple:  95% accuracy (tool execution)
BFCL memory:       45% answer accuracy (end-to-end)
llm-memory-bench:  48% recall, 51% precision (storage)
```
**Interpretation:** Tool execution is fine. Storage judgment is broken.

**Action:** Improve system prompts for proactive triggers. The model can call tools correctly when told to, but doesn't recognize what's worth storing.

### Case 2: Retrieval problem (good storage, bad answers)  
```
BFCL live_simple:  92% accuracy
BFCL memory:       45% answer accuracy
llm-memory-bench:  88% recall, 92% precision
```
**Interpretation:** Storage is good (high recall/precision), but retrieval is broken. The gap (88% → 45%) is the retrieval failure.

**Action:** Fix memory backend queries, improve search/indexing. Consider different storage backend (vector vs KV).

### Case 3: All strong
```
BFCL live_simple:  95% accuracy
BFCL memory:       88% answer accuracy  
llm-memory-bench:  90% recall, 87% precision
```
**Interpretation:** Strong across the board. Proactive judgment works, tool execution works, retrieval works.

**Action:** Performance near ceiling. Focus on edge cases or harder datasets.

### Case 4: Selective but accurate
```
BFCL memory:       82% answer accuracy
llm-memory-bench:  52% recall, 96% precision
```
**Interpretation:** Model stores conservatively (low recall) but what it stores is high quality (high precision). Can still answer most questions because it stored the critical facts.

**Action:** If acceptable, keep as-is. If more coverage needed, tune prompts to be more aggressive about storage.

## Why Use BFCL Instead of Reimplementing?

BFCL provides:
- **Sophisticated evaluation:** AST-based matching handles argument variations, type conversions, order-independent parallel calls
- **25+ model handlers:** Pre-built integrations for all major providers
- **Public leaderboard:** Compare your results against published baselines
- **Active maintenance:** Regularly updated with new test categories

Your benchmark's unique contribution is:
- Testing **proactive judgment** (BFCL assumes explicit function requests)
- Comparing **memory systems** (simple vs gbrain vs memoryhub)
- Dataset quality improvements (AlpsBench filtering, task2 fix)

## Reporting Results

When publishing results, include:

```markdown
### Tool-Calling Baseline (BFCL v4 live_simple)
- Accuracy: 92%
- Irrelevance detection: 98%

### Memory Extraction (AlpsBench task1, memoryhub system)
- Extraction recall: 68%
- Extraction precision: 82%
- Noise resistance: 91%

### Gap Analysis
The 24% gap (92% → 68% recall) represents the cost of **proactive judgment**.
The model can execute tools correctly when told to, but struggles to identify
what information is worth storing in natural conversation.
```

## Future Work: Prompted AlpsBench Variant

To complete the hierarchy, implement an "explicit memory request" variant:

```yaml
# Original (proactive)
- role: user
  content: "I prefer metric units for all calculations"
  ground_truth:
    should_store:
      - fact: "Prefers metric units"

# Prompted variant (reactive) 
- role: user
  content: "Remember that I prefer metric units for all calculations"
  ground_truth:
    should_store:
      - fact: "Prefers metric units"
```

This would isolate semantic extraction (what to store) from judgment (when to store).

## Metrics Comparison

**See [metrics.md](metrics.md) for comprehensive metric definitions, formulas, and interpretation guide.**

### Quick Comparison

### BFCL Metrics

BFCL has **two types of tests** with different metrics:

#### Function-Calling Tests (live_simple, live_multiple, live_parallel)

Uses **AST-based exact matching**:

| Metric | What it measures | How it's computed |
|--------|------------------|-------------------|
| **Accuracy** | Exact correctness of function calls | AST structural match: function name + all arguments match |
| **Type correctness** | Type conversion validity | Checks if parameter types can be safely converted |
| **Irrelevance detection** | Not calling when no function matches | Fraction of irrelevance tests with zero calls |

**Key characteristics:**
- ✅ Precise: catches subtle errors (wrong type, missing required param)
- ❌ No partial credit: close-but-wrong = 0% 
- ❌ Semantic blind: paraphrased arguments fail

**Example:**
```python
Expected: get_weather(location="San Francisco", unit="fahrenheit")
Model: get_weather(location="SF", unit="fahrenheit")
Result: FAIL (0%) - "SF" ≠ "San Francisco"
```

#### Memory Tests (memory_kv, memory_vector, memory_rec_sum)

Uses **answer text matching**:

| Metric | What it measures | How it's computed |
|--------|------------------|-------------------|
| **Answer accuracy** | End-to-end correctness | Does answer text contain expected value? |

**Key characteristics:**
- ✅ Tests full pipeline: storage + retrieval
- ❌ Black box: can't distinguish storage vs retrieval failures
- ❌ No partial credit: either contains value or doesn't

**Example:**
```python
Question: "What is my first name?"
Expected: ["Michael"]

Model answer: "Your first name is Michael."
Result: CORRECT (100%)

Model answer: "I don't have that information."
Result: INCORRECT (0%)
```

### llm-memory-bench Metrics

We use **semantic matching** (LLM judge or embeddings) to evaluate **storage quality** (what was actually stored):

| Metric | What it measures | How it's computed |
|--------|------------------|-------------------|
| **extraction_precision** | Quality of calls made | `true_positives / (true_positives + false_positives)` |
| **extraction_recall** | Coverage of expected facts | `true_positives / (true_positives + false_negatives)` |
| **extraction_f1** | Harmonic mean | `2 * precision * recall / (precision + recall)` |
| **noise_resistance_rate** | Ignoring irrelevant turns | `(noise_turns - noise_violations) / noise_turns` |
| **schema_validity_rate** | Tool schema compliance | `valid_calls / total_calls` |

**Matching verdicts:**
- `MATCH`: Stored content captures the same core information (perfect recall)
- `PARTIAL`: Some information captured but incomplete (counts as 0.5 TP + 0.5 FN)
- `NO_MATCH`: Missed entirely (false negative)

**Key characteristics:**
- ✅ Semantic: paraphrasing is okay if meaning preserved
- ✅ Partial credit: recognizes almost-correct answers
- ✅ Proactive focus: measures judgment, not just execution
- ❌ Judge variance: LLM matching less deterministic than AST
- ❌ Slower: semantic comparison costs tokens or compute (embeddings)

**Example:**
```python
# Expected fact
"User prefers metric units for calculations"

# Stored: MATCH (paraphrased but semantically equivalent)
"Uses metric system for all math"

# Stored: PARTIAL (incomplete - missing "calculations" context)  
"Prefers metric units"

# Stored: NO_MATCH (semantically different)
"User is from Europe"
```

### Summary: Three Different Evaluation Approaches

| Benchmark | What's Evaluated | How It's Scored | Failure Visibility |
|-----------|------------------|-----------------|-------------------|
| **BFCL Function Tests** | Tool execution | AST exact match of calls | Specific (wrong param) |
| **BFCL Memory Tests** | End-to-end answers | Text contains value | Ambiguous (storage? retrieval?) |
| **llm-memory-bench** | Storage quality | Semantic fact match | Specific (false negative) |

### Why Different Metrics?

The metric choice reflects what each benchmark tests:

**BFCL function tests check execution precision:**
- Did you call `get_weather` or `weather_forecast`? (function selection)
- Is location a string or dict? (type correctness)
- Did you include all required parameters? (argument extraction)

→ **Exact AST match** is the right metric because there's one correct answer defined by the schema.

**BFCL memory tests check end-to-end success:**
- Can you answer "What is my first name?" correctly?
- Does your text response contain the expected value?

→ **Text matching** is the right metric for user-facing quality, but can't diagnose failures.

**llm-memory-bench checks storage quality:**
- Did you recognize "I'm a data scientist" as worth storing? (implicit trigger detection)
- Did you store "Data scientist role" vs "User is a data scientist" vs "Works in data science"? (semantic extraction)
- Did you ignore "It's raining today"? (noise resistance)

→ **Semantic match** is the right metric because there are many valid ways to phrase the same fact.

### Practical Implications

**High BFCL, low memory extraction:**
```
BFCL accuracy:        92%
Extraction recall:    45%
Extraction precision: 78%
```
**Diagnosis:** Good at tool execution, poor at proactive judgment.
- The model can call tools correctly when told to
- But it misses 55% of facts worth storing (low recall)
- When it does store, it's mostly correct (high precision)

**Action:** Focus on prompt engineering to improve trigger recognition.

---

**Low BFCL, low memory extraction:**
```
BFCL accuracy:        65%
Extraction recall:    40%
Extraction precision: 52%
```
**Diagnosis:** Fundamental tool-calling issues.
- The model struggles with function calls generally
- Memory performance is capped by execution ability
- Low precision suggests schema misunderstanding

**Action:** Check tool schema formatting, consider model fine-tuning, or try a different provider.

---

**High BFCL, high memory extraction:**
```
BFCL accuracy:        95%
Extraction recall:    88%
Extraction precision: 91%
```
**Diagnosis:** Strong across the board.
- Proactive judgment is not the bottleneck
- Most failures are edge cases or ambiguous facts
- Performance is near ceiling for this dataset

**Action:** Test on harder datasets or focus on value benchmark.

## References

- **BFCL:** https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard
- **BFCL Leaderboard:** https://gorilla.cs.berkeley.edu/leaderboard.html
- **AlpsBench:** https://huggingface.co/datasets/Cosineyx/Alpsbench

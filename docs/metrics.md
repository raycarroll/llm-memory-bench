# Metrics Reference

## llm-memory-bench Metrics

All metrics measure **storage quality** — what facts were actually stored vs what should have been stored.

**Note:** Metric names are prefixed based on dataset type:
- **Per-turn datasets** (e.g., AlpsBench): `per_turn_extraction_recall`, `per_turn_extraction_precision`, etc.
- **Cumulative datasets** (e.g., BFCL Memory): `cumulative_extraction_recall`, `cumulative_extraction_precision`, etc.

See [Dataset Types](dataset-types.md) for details on the difference between per-turn and cumulative ground truth.

---

## Per-Turn Metrics (AlpsBench)

**What they test:** Proactive judgment timing — **WHEN** to store facts

**Evaluation approach:** Facts stored during turn N are matched against turn N's expected facts. Timing matters.

### Core Extraction Metrics

#### per_turn_extraction_precision
**What it measures:** Quality of storage decisions at each turn — of all tool calls made at each turn, what fraction stored the correct fact for that turn?

**Formula:**
```
precision = true_positives / (true_positives + false_positives)
```

**Range:** 0.0 to 1.0 (higher is better)

**Interpretation:**
- **1.0** = Every tool call stored a fact that should have been stored (no noise)
- **0.5** = Half of tool calls stored irrelevant information (over-storage)
- **0.0** = Every tool call was spurious (storing nothing useful)

**Example:**
```
Expected facts: ["User is a data scientist", "Prefers Python"]
Tool calls made: 
  1. add_memory("User is a data scientist") ✓ true positive
  2. add_memory("It's sunny today")         ✗ false positive
  3. add_memory("Prefers Python")           ✓ true positive

precision = 2 / (2 + 1) = 0.667
```

---

#### per_turn_extraction_recall
**What it measures:** Coverage of storage at correct timing — of all facts that should have been stored at specific turns, what fraction were captured at those turns?

**Formula:**
```
recall = true_positives / (true_positives + false_negatives)
```

**Range:** 0.0 to 1.0 (higher is better)

**Interpretation:**
- **1.0** = Captured every fact worth storing (complete coverage)
- **0.5** = Missed half the important facts (incomplete memory)
- **0.0** = Stored nothing at all (total miss)

**Example:**
```
Expected facts: ["User is a data scientist", "Prefers Python", "Works remotely"]
Tool calls made:
  1. add_memory("User is a data scientist") ✓ matches "User is a data scientist"
  2. add_memory("Prefers Python")           ✓ matches "Prefers Python"
  (nothing for "Works remotely")            ✗ false negative

recall = 2 / (2 + 1) = 0.667
```

---

#### per_turn_extraction_f1
**What it measures:** Harmonic mean of per-turn precision and recall — overall storage quality balancing both accuracy and coverage with correct timing.

**Formula:**
```
f1 = 2 * (precision * recall) / (precision + recall)
```

**Range:** 0.0 to 1.0 (higher is better)

**Interpretation:**
- **1.0** = Perfect precision AND perfect recall
- **0.7-0.9** = Strong performance, minor issues
- **0.4-0.7** = Moderate performance, significant gaps
- **<0.4** = Poor performance, major storage issues

**Why F1 matters:**
Raw accuracy can be misleading. A system that stores nothing has perfect precision (no false positives) but 0% recall. F1 penalizes this imbalance.

**Example:**
```
precision = 0.667, recall = 0.667
f1 = 2 * (0.667 * 0.667) / (0.667 + 0.667) = 0.667
```

---

### Per-Turn Noise Resistance

**Note:** Noise resistance is **only applicable to per-turn datasets**. Cumulative datasets don't have per-turn expectations, so noise resistance isn't measured.

#### per_turn_noise_resistance_rate
**What it measures:** Ability to ignore irrelevant conversation turns at the right time — of all turns with no facts to store, what fraction had zero tool calls?

**Formula:**
```
noise_resistance = (noise_turns - noise_violations) / noise_turns
```

**Range:** 0.0 to 1.0 (higher is better)

**Interpretation:**
- **1.0** = Never called tools on noise turns (perfect restraint)
- **0.9** = Called tools on 10% of noise turns (conservative)
- **0.5** = Called tools on half of noise turns (over-eager)
- **<0.3** = Calls tools constantly (poor judgment)

**Why this matters:**
AlpsBench conversations are ~90% noise. A system that stores every turn would get high recall but terrible noise resistance and precision.

**Example:**
```
Conversation has 20 turns:
  - 18 noise turns (casual chat, no facts)
    - 16 had zero tool calls ✓
    - 2 had spurious tool calls ✗
  - 2 fact-bearing turns

noise_resistance = (18 - 2) / 18 = 0.889
```

---

### Schema Validity

#### schema_validity_rate
**What it measures:** Tool schema compliance — what fraction of tool calls had valid arguments per the schema?

**Formula:**
```
schema_validity = valid_calls / total_calls
```

**Range:** 0.0 to 1.0 (higher is better)

**Interpretation:**
- **1.0** = All tool calls were schema-compliant
- **<1.0** = Some calls had missing required params, wrong types, or invalid values
- **<0.8** = Significant schema understanding issues

**Example:**
```
Tool schema requires: {"fact": string, "category": enum}
Call 1: {"fact": "...", "category": "preference"} ✓ valid
Call 2: {"fact": "..."}                           ✓ valid (category optional)
Call 3: {"category": "preference"}                ✗ invalid (missing required "fact")
Call 4: {"fact": "...", "category": "invalid"}    ✗ invalid (not in enum)

schema_validity = 2 / 4 = 0.5
```

---

## Cumulative Metrics (BFCL Memory)

**What they test:** Storage completeness — **WHAT** gets stored (timing irrelevant)

**Evaluation approach:** ALL facts stored across the entire conversation are matched against the cumulative expected facts. Order doesn't matter, only completeness.

### Core Extraction Metrics

#### cumulative_extraction_recall
**What it measures:** Completeness of storage — of all expected facts (anywhere in the conversation), what fraction were stored?

**Formula:**
```
recall = true_positives / (true_positives + false_negatives)
```

**Range:** 0.0 to 1.0 (higher is better)

**Interpretation:**
- **1.0** = Captured every fact worth storing (complete coverage)
- **0.5** = Missed half the important facts (incomplete memory)
- **0.0** = Stored nothing at all (total miss)

**Key difference from per-turn:** A fact stored at **any point** during the conversation counts as captured. With per-turn, it must be stored at a specific turn.

**Example:**
```
Expected cumulative facts: 
  ["User is a data scientist", "Prefers Python", "Works remotely"]

Conversation has 3 turns:
  Turn 1: add_memory("User is a data scientist") ✓
  Turn 2: (no tool calls)
  Turn 3: add_memory("Prefers Python")           ✓
  (nothing for "Works remotely")                  ✗

cumulative_recall = 2 / 3 = 0.667
```

---

#### cumulative_extraction_precision
**What it measures:** Quality of all storage decisions across the conversation — of all facts stored (at any point), what fraction were expected?

**Formula:**
```
precision = true_positives / (true_positives + false_positives)
```

**Range:** 0.0 to 1.0 (higher is better)

**Interpretation:**
- **1.0** = Every fact stored was expected (no spurious storage)
- **0.5** = Half of stored facts were not expected (over-storage)
- **0.0** = No stored facts were expected (all noise)

**Example:**
```
Expected cumulative facts: ["User is a data scientist", "Prefers Python"]

All tool calls across conversation:
  1. add_memory("User is a data scientist") ✓ expected
  2. add_memory("It's sunny today")         ✗ not expected
  3. add_memory("Prefers Python")           ✓ expected

cumulative_precision = 2 / 3 = 0.667
```

---

#### cumulative_extraction_f1
**What it measures:** Harmonic mean of cumulative precision and recall — overall storage completeness balancing accuracy and coverage.

**Formula:**
```
f1 = 2 * (precision * recall) / (precision + recall)
```

**Range:** 0.0 to 1.0 (higher is better)

**Interpretation:**
- **1.0** = Perfect precision AND perfect recall across the conversation
- **0.7-0.9** = Strong completeness, minor gaps
- **0.4-0.7** = Moderate completeness, significant gaps
- **<0.4** = Poor completeness, major storage issues

**Example:**
```
cumulative_precision = 0.667, cumulative_recall = 0.667
cumulative_f1 = 2 * (0.667 * 0.667) / (0.667 + 0.667) = 0.667
```

---

### Noise Resistance: Not Applicable

**Important:** Cumulative datasets do not measure noise resistance because there are no per-turn expectations. We only care that facts are stored **somewhere** in the conversation, not **when** they're stored.

If a model stores facts too eagerly (e.g., on noise turns), this is captured by `cumulative_extraction_precision` (false positives for irrelevant facts), but not by a separate noise metric.

---

### Count Metrics

These provide context for the percentage metrics above:

- **true_positives** — number of facts correctly stored
- **false_negatives** — number of facts that should have been stored but weren't
- **false_positives** — number of spurious tool calls
- **noise_turns** — number of turns with no facts to store
- **noise_violations** — number of noise turns where tools were called anyway
- **total_tool_calls** — total number of memory tool calls made

---

## Matching Verdicts

When evaluating whether a stored fact matches an expected fact, we use semantic matching with three verdicts:

### MATCH
The stored content captures the same core information as the expected fact.

**Scoring:** Counts as 1.0 true positive

**Examples:**
- Expected: "User prefers Python"
- Stored: "User's preferred language is Python" → **MATCH**
- Stored: "Prefers Python for scripting" → **MATCH**

### PARTIAL
The stored content captures some but not all of the expected information.

**Scoring:** Counts as 0.5 true positive + 0.5 false negative

**Examples:**
- Expected: "User is a data scientist working remotely"
- Stored: "User is a data scientist" → **PARTIAL** (missing "remotely")
- Stored: "Works remotely" → **PARTIAL** (missing "data scientist")

### NO_MATCH
The stored content doesn't capture the expected fact at all.

**Scoring:** Counts as 1.0 false negative

**Examples:**
- Expected: "User prefers Python"
- Stored: "User is from Seattle" → **NO_MATCH**
- Stored: (no tool call made) → **NO_MATCH**

---

## BFCL Metrics Comparison

### What BFCL Measures

BFCL has **two different test categories** with different scoring:

#### Function-Calling Tests (live_simple, live_multiple, live_parallel)

Uses **AST-based exact matching** of tool calls:

| Metric | What it measures |
|--------|------------------|
| **Accuracy** | Exact correctness: function name + all arguments match structurally |
| **Type correctness** | Can parameter types be safely converted? |
| **Irrelevance detection** | Fraction of tests where no function was called (when none should be) |

**Scoring:** Binary (100% or 0%) - even one character difference = failure

**Example:**
```
Expected: get_weather(location="San Francisco", unit="fahrenheit")
Model: get_weather(location="SF", unit="fahrenheit")
Score: FAIL (0%) - "SF" ≠ "San Francisco"
```

#### Memory Tests (memory_kv, memory_vector, memory_rec_sum)

Uses **answer text matching** (end-to-end):

| Metric | What it measures |
|--------|------------------|
| **Answer accuracy** | Does the text answer contain the expected value? |

**Scoring:** Does the model's natural language response contain the expected answer?

**Example:**
```
Question: "What is my first name?"
Expected: ["Michael"]

Model answer: "Your first name is Michael."
Score: CORRECT (contains "Michael")

Model answer: "I don't have that information."
Score: INCORRECT (missing expected value)
```

**Note:** BFCL Memory measures end-to-end success (storage + retrieval) but doesn't distinguish which failed.

### Key Differences

| Aspect | BFCL Function Tests | BFCL Memory Tests | llm-memory-bench |
|--------|---------------------|-------------------|------------------|
| **What's tested** | Tool execution | Storage + retrieval | Storage only |
| **Matching** | Exact (AST) | Text contains value | Semantic (meaning) |
| **Granularity** | Binary (0%/100%) | Binary (0%/100%) | Three-level verdicts |
| **Evaluation** | White box (tool calls) | Black box (answers) | White box (stored facts) |
| **Failure mode** | Specific (wrong param) | Ambiguous (storage? retrieval?) | Specific (false negative) |
| **False positives** | Measured (irrelevance tests) | Not measured | `extraction_precision` |
| **Coverage** | Not measured | Not measured | `extraction_recall` |

### Example: Different Evaluation Approaches

#### BFCL Function-Calling Tests
```
Expected: get_weather(location="San Francisco", unit="fahrenheit")
Model: get_weather(location="San Francisco Bay Area", unit="fahrenheit")

Score: FAIL (0%)
Reason: "San Francisco Bay Area" ≠ "San Francisco" (exact AST match required)
```

#### BFCL Memory Tests
```
Question: "What is my first name?"
Expected: ["Michael"]

Model answer: "Based on our conversation, your first name is Michael."
Score: CORRECT (100%)
Reason: Answer text contains "Michael"

Model answer: "Your name is Mike."
Score: INCORRECT (0%)
Reason: "Mike" ≠ "Michael" (exact value match required)
```

#### llm-memory-bench
```
Expected fact: "User's name is Michael"
Stored: "Customer name is Michael"

Score: MATCH (100%)
Reason: Semantic equivalence - same meaning, different wording acceptable
```

---

## Metric Relationships

### Precision/Recall Tradeoff

Systems can optimize for different points on the curve:

```
High precision, low recall (conservative):
  - Stores only obvious facts
  - Few false positives
  - Misses subtle facts
  - Example: precision=0.95, recall=0.35

Balanced:
  - Good judgment across types
  - Example: precision=0.80, recall=0.70

Low precision, high recall (aggressive):
  - Stores liberally
  - Catches most facts
  - Includes noise
  - Example: precision=0.55, recall=0.90
```

### F1 as the Tiebreaker

When comparing systems, F1 score provides a single number:

```
System A: precision=0.95, recall=0.35 → f1=0.51
System B: precision=0.80, recall=0.70 → f1=0.75

System B is better overall (higher F1)
```

### Noise Resistance vs Recall

These often trade off:

```
More aggressive prompting:
  → Higher recall (catches more facts)
  → Lower noise resistance (more false positives)

More conservative prompting:
  → Lower recall (misses subtle facts)
  → Higher noise resistance (fewer false positives)
```

---

## Interpretation Guide

### Strong Performance
```
extraction_precision:    0.80 - 1.00
extraction_recall:       0.70 - 1.00
extraction_f1:           0.75 - 1.00
noise_resistance_rate:   0.85 - 1.00
schema_validity_rate:    0.95 - 1.00
```
**Diagnosis:** System works well, minor tuning only

---

### Moderate Performance
```
extraction_precision:    0.60 - 0.80
extraction_recall:       0.50 - 0.70
extraction_f1:           0.55 - 0.75
noise_resistance_rate:   0.70 - 0.85
schema_validity_rate:    0.85 - 0.95
```
**Diagnosis:** Functional but needs improvement in prompt or schema design

---

### Poor Performance
```
extraction_precision:    < 0.60
extraction_recall:       < 0.50
extraction_f1:           < 0.55
noise_resistance_rate:   < 0.70
schema_validity_rate:    < 0.85
```
**Diagnosis:** Fundamental issues - check prompt, model capability, or tool schema

---

### Specific Diagnoses

**High precision, low recall:**
- **Problem:** Too conservative, missing facts
- **Action:** Add more proactive guidance to prompt, provide examples

**Low precision, high recall:**
- **Problem:** Over-eager, storing noise
- **Action:** Add filtering guidance, emphasize "important facts only"

**Low noise resistance:**
- **Problem:** Calling tools on casual conversation
- **Action:** Add explicit "don't store" examples for greetings, small talk

**Low schema validity:**
- **Problem:** Model doesn't understand tool parameters
- **Action:** Simplify schema, add parameter descriptions, check provider formatting

---

## Cross-Benchmark Diagnosis

Combine BFCL and llm-memory-bench metrics for full picture:

### Scenario 1: Storage Problem
```
BFCL accuracy:            92% (tool execution works)
llm-memory-bench recall:  35% (missing 65% of facts)
```
**Diagnosis:** Proactive judgment broken, not tool execution  
**Action:** Improve prompts for memory-specific guidance

### Scenario 2: Retrieval Problem
```
BFCL memory accuracy:     45% (end-to-end fails)
llm-memory-bench recall:  88% (storage works)
```
**Diagnosis:** Storage is good, retrieval is broken  
**Action:** Fix memory backend queries, not prompts

### Scenario 3: All Good
```
BFCL accuracy:            95%
BFCL memory accuracy:     85%
llm-memory-bench recall:  82%
llm-memory-bench f1:      0.85
```
**Diagnosis:** Strong performance across the board  
**Action:** Performance near ceiling, focus on edge cases

---

## Per-Fact-Type Breakdown

The benchmark also reports metrics by fact type:

```json
{
  "metrics_by_fact_type": {
    "direct": {
      "recall": 0.75,
      "tp": 450,
      "fn": 150
    },
    "indirect": {
      "recall": 0.58,
      "tp": 280,
      "fn": 200
    }
  }
}
```

**Direct facts:** Explicitly stated ("I prefer Python")  
**Indirect facts:** Implied or contextual ("I always use Python for scripting" → prefers Python)

Lower recall on indirect facts is expected — they're harder to recognize.

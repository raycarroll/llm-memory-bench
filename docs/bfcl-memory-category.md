# BFCL Memory Category Analysis

## Overview

BFCL v4 includes a **memory-based agentic category** with 155 test cases across 5 scenarios. It tests **reactive retrieval** — whether models can recall facts from long conversations when explicitly asked.

**Blog post:** https://gorilla.cs.berkeley.edu/blogs/16_bfcl_v4_memory.html

## Test Structure

### Format

1. **Prerequisite conversation** (provided before the test)
   - Long multi-turn dialogue (8-10 user messages)
   - User naturally shares personal information
   - No explicit "remember this" prompts
   
2. **Test question** (the actual test)
   - Single user message asking for specific information
   - E.g., "What is my first name?"
   
3. **Expected answer** (ground truth)
   - Specific fact(s) from prerequisite conversation
   - E.g., `["Michael"]`

### Example

**Prerequisite conversation (excerpt):**
```
User: "Hey there! ... My name is Michael, and this is my first 
      time interacting with your company..."
User: "I'm 35 years old, live in Seattle, and am pretty serious 
      about both my work and my hobbies..."
User: "I work as a freelance graphic designer..."
[... 7 more messages about preferences, needs, etc.]
```

**Test question:**
```
User: "What is my first name?"
```

**Expected answer:**
```json
{
  "ground_truth": ["Michael"],
  "source": "My name is Michael, and this is my first time..."
}
```

**Scoring:**
Model's text answer is checked for the expected value. If the response contains "Michael", it's correct.

Example responses:
- "Your first name is Michael." → ✓ CORRECT
- "Based on our earlier conversation, it's Michael." → ✓ CORRECT  
- "I don't recall that information." → ✗ INCORRECT

## Scenarios (155 total tests)

| Scenario | Count | Description |
|----------|-------|-------------|
| `student` | 50 | College student discussing courses, projects, deadlines |
| `customer` | 30 | First-time buyer inquiring about products |
| `healthcare` | 25 | Patient discussing medical history, conditions |
| `finance` | 25 | Investment manager describing deals, strategies |
| `notetaker` | 25 | Personal assistant recalling tasks, schedules |

## Memory Backends Tested

BFCL tests three storage paradigms (run on the same test cases):

- **`memory_kv`** — Key-value store (e.g., Redis, dict)
- **`memory_vector`** — Vector database (e.g., Pinecone, Chroma)
- **`memory_rec_sum`** — Recursive summarization (hierarchical compression)

The test cases themselves don't specify which backend to use — this is a runtime configuration.

## Evaluation Method

**BFCL Memory uses answer text matching:**

```
Question: "What is my favorite course?"
Expected: ["Advanced Algorithms"]

Evaluation:
1. Model generates text response (no dialogue history)
2. Check if response contains "Advanced Algorithms"
3. Score: CORRECT or INCORRECT (binary)

Final metric: % of questions answered correctly
```

**Not evaluated:**
- Which memory API calls were made during storage
- How facts were structured in the backend
- Whether facts were over/under-stored

**Only evaluated:**
- Can the model produce the correct answer text?

## What BFCL Memory Tests

✅ **Proactive storage** — models must decide what to store during prerequisite conversations (no explicit "remember" prompts)  
✅ **Reactive retrieval** — answering explicit questions about stored information  
✅ **Long-context recall** — finding facts buried in multi-turn conversations  
✅ **Specific fact extraction** — returning exact answers ("Michael", not "the user's name")  
✅ **Multi-scenario generalization** — different domains (healthcare, finance, etc.)  
✅ **End-to-end pipeline** — storage → snapshot → reload → retrieval

## What It Does NOT Test

❌ **Storage quality isolation** — failures are ambiguous (didn't store? stored wrong? can't retrieve?)  
❌ **Per-fact ground truth** — no labeled "should store" annotations per turn  
❌ **Storage metrics** — only measures final answer accuracy, not what was stored  
❌ **Noise resistance** — prerequisite conversations are information-rich, minimal noise

## Comparison to llm-memory-bench

| Aspect | BFCL Memory | llm-memory-bench |
|--------|-------------|------------------|
| **Focus** | End-to-end accuracy (black box) | Storage quality (white box) |
| **Phases** | Storage + retrieval | Storage only |
| **Evaluation** | Does answer text contain expected value? | Does stored fact match semantically? |
| **Matching** | Text contains substring | Semantic equivalence |
| **Metrics** | Answer accuracy (%) | Precision, recall, F1 |
| **Failure mode** | Ambiguous (storage? retrieval? both?) | Specific (false positive/negative) |
| **Ground truth** | Expected answer values | Per-turn fact annotations |
| **Noise** | Information-rich conversations | ~90% noise turns (AlpsBench) |
| **Diagnostic** | User-facing quality | System debugging |
| **Partial credit** | No (binary correct/incorrect) | Yes (MATCH/PARTIAL/NO_MATCH) |

## Diagnostic Complementarity

Both benchmarks test proactive storage, but measure it differently:

**BFCL Memory (black box):**
- Metric: Can you answer questions from conversation?
- Failure: Ambiguous — didn't store? stored incorrectly? can't retrieve?
- Value: User-facing quality measure

**llm-memory-bench (white box):**
- Metric: What facts were actually stored?
- Failure: Specific — false negative (didn't store) vs false positive (stored noise)
- Value: System debugging and diagnosis

### Combined Diagnostic Power

```
Scenario 1: Storage problem
  BFCL: 45% answer accuracy
  Bench: 48% recall, 51% precision
  → Diagnosis: Storage judgment is broken
  → Action: Fix proactive triggers, improve prompts

Scenario 2: Retrieval problem  
  BFCL: 45% answer accuracy
  Bench: 88% recall, 92% precision
  → Diagnosis: Storage is good, retrieval is broken
  → Action: Fix memory backend queries, improve search

Scenario 3: Selective but accurate
  BFCL: 88% answer accuracy
  Bench: 48% recall, 95% precision
  → Diagnosis: Stores selectively, retrieves well
  → Action: Storage is conservative but high-quality
```

### Integration Approach

**Using BFCL prerequisite conversations for storage testing:**

We extract BFCL prerequisite conversations and reverse-engineer ground truth from the questions:

```bash
# Convert BFCL memory data
llm-memory-bench convert --source bfcl-memory \
  --bfcl-repo /path/to/gorilla \
  --output datasets/converted

# Run storage quality benchmark
llm-memory-bench run \
  --dataset datasets/converted/bfcl_memory_student.yaml \
  --system memoryhub

llm-memory-bench evaluate results/[run-file].json
```

**Ground truth derivation:**
- BFCL question: "What is my first name?"
- BFCL answer: ["Michael"]
- Answer source: "My name is Michael, and this is..."
- → Turn containing that text gets: `should_store: ["User's name is Michael"]`

**Two-phase diagnostic workflow:**

```bash
# Phase 1: Test storage quality (our benchmark)
llm-memory-bench run --dataset bfcl_memory_student.yaml --system memoryhub
# → Get: 85% recall, 78% precision (white-box storage metrics)

# Phase 2: Test end-to-end (BFCL's benchmark)  
bfcl generate --model claude-sonnet-4 --test-category memory_kv
bfcl evaluate --model claude-sonnet-4 --test-category memory_kv
# → Get: 72% answer accuracy (black-box retrieval metrics)

# Diagnosis:
# Gap (85% → 72%) suggests retrieval issues, not storage
```

**Benefits:**
- 5 domain-specific scenarios (student, healthcare, finance, customer, notetaker)
- High-quality conversations (human-authored, realistic)
- Linked to BFCL questions for validation
- Complementary to AlpsBench (different domains, denser information)

## Data Availability

**BFCL memory data:**
- `bfcl_eval/data/BFCL_v4_memory.json` — 155 test questions
- `bfcl_eval/data/memory_prereq_conversation/` — 5 scenario prerequisite dialogues
  - `memory_customer.json`
  - `memory_healthcare.json`
  - `memory_finance.json`
  - `memory_student.json`
  - `memory_notetaker.json`
- `bfcl_eval/data/possible_answer/BFCL_v4_memory.json` — Expected answers

**License:** Apache 2.0 (same as BFCL)

## Recommendation

**Use BFCL memory category for:**
- ✅ Retrieval baselines (can the model recall what was stored?)
- ✅ Long-context memory testing (facts buried in conversation)
- ✅ Cross-backend comparison (KV vs vector vs summarization)

**Use llm-memory-bench for:**
- ✅ Storage judgment baselines (what gets stored unprompted?)
- ✅ Noise resistance testing (ignoring irrelevant turns)
- ✅ Memory system prompt comparison (simple vs gbrain vs memoryhub)

**Combine them for:**
- ✅ End-to-end memory system evaluation (storage + retrieval)
- ✅ Gap analysis (stored but not retrieved vs not stored at all)
- ✅ Multi-phase benchmarking (decision → execution → recall)

## Next Steps (Future Work)

1. **Adapt BFCL prereq conversations** as llm-memory-bench scenarios
   - Run proactive extraction test on the conversations
   - Compare what gets stored vs what's needed for BFCL questions
   
2. **Two-phase benchmark**
   - Phase 1: Proactive storage (llm-memory-bench)
   - Phase 2: Retrieval accuracy (BFCL memory)
   - Metric: % of BFCL questions answerable from stored facts

3. **Backend comparison**
   - Test memory systems with different storage paradigms
   - See if KV/vector/summarization affects proactive judgment

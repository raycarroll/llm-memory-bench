# BFCL Memory KV System - Usage Guide

## Overview

The `bfcl_memory_kv` system implements BFCL v4's memory category with Key-Value backend and domain-specific scenarios.

## Quick Start

### CLI Usage

```bash
# Run with student scenario
llm-memory-bench run \
  --dataset datasets/converted/bfcl_memory_student.yaml \
  --system bfcl_memory_kv \
  --scenario student \
  --provider anthropic \
  --model claude-sonnet-4-20250514

# Run with healthcare scenario
llm-memory-bench run \
  --dataset datasets/converted/bfcl_memory_healthcare.yaml \
  --system bfcl_memory_kv \
  --scenario healthcare
```

### Available Scenarios

| Scenario | Persona | Use Case |
|----------|---------|----------|
| `student` | Academic-support assistant | College student coursework, deadlines, projects |
| `customer` | E-commerce support | Product inquiries, order tracking, preferences |
| `finance` | Executive assistant | Investment decisions, financial goals, deals |
| `healthcare` | Patient care assistant | Medical history, treatments, appointments |
| `notetaker` | Personal organization | Tasks, deadlines, reminders |

### Error Handling

**Missing scenario:**
```bash
llm-memory-bench run --dataset data.yaml --system bfcl_memory_kv
# ERROR: BFCL systems require --scenario parameter.
#        Available: ['student', 'customer', 'finance', 'healthcare', 'notetaker']
```

**Invalid scenario:**
```bash
llm-memory-bench run --dataset data.yaml --system bfcl_memory_kv --scenario invalid
# ERROR: Invalid scenario: invalid.
#        Must be one of: ['student', 'customer', 'finance', 'healthcare', 'notetaker']
```

**Scenario on non-BFCL system (currently allowed):**
```bash
llm-memory-bench run --dataset data.yaml --system simple --scenario student
# Works but scenario is ignored (simple system doesn't use it)
```

## System Components

### 1. System Prompt

**Format:**
```
{scenario_persona}

{memory_instruction}
```

**Example (student scenario):**
```
You are an academic-support assistant for college student. Remember key 
personal and academic details discussed across sessions, and draw on them 
to answer questions or give guidance.

You have access to an advanced memory system, consisting of two memory types 
'Core Memory' and 'Archival Memory'...
```

### 2. Tool Schema

**Core Memory Tools** (limited size, always in context):
- `core_memory_add(key: str, value: str)` - Add key-value pair
- `core_memory_remove(key: str)` - Remove by key
- `core_memory_replace(key: str, value: str)` - Update value

**Archival Memory Tools** (larger capacity, search-based):
- `archival_memory_add(key: str, value: str)` - Add to long-term storage
- `archival_memory_key_search(query: str, k: int = 5)` - BM25+ search

**Constraints:**
- Keys must be `snake_case` (no spaces)
- Core memory: max 7 entries, 300 chars per value
- Archival memory: max 50 entries, 2000 chars per value

### 3. Fact Extraction

Tool calls are converted to facts for evaluation:

```python
# Tool call
core_memory_add(key="student_name", value="Michael")

# Extracted fact
"student_name: Michael"
```

## Comparison Examples

### Same Dataset, Different Systems

```bash
# Basic memory-aware prompt
llm-memory-bench run --dataset bfcl_memory_student.yaml --system simple

# BFCL with student scenario (domain-specific)
llm-memory-bench run --dataset bfcl_memory_student.yaml --system bfcl_memory_kv --scenario student

# MemoryHub (sophisticated generic prompts)
llm-memory-bench run --dataset bfcl_memory_student.yaml --system memoryhub
```

**Expected pattern:**
- `simple`: ~65% recall (basic memory guidance)
- `bfcl_memory_kv --scenario student`: ~65% recall (domain-specific persona)
- `memoryhub`: ~72% recall (sophisticated generic prompts)

### Same System, Different Scenarios

```bash
# Aligned scenario
llm-memory-bench run --dataset bfcl_memory_student.yaml --system bfcl_memory_kv --scenario student
# → Expected: 65% recall

# Misaligned scenario
llm-memory-bench run --dataset bfcl_memory_student.yaml --system bfcl_memory_kv --scenario healthcare
# → Expected: 58% recall (-7% penalty for wrong domain context)
```

## Evaluation

```bash
# Run benchmark
llm-memory-bench run \
  --dataset datasets/converted/bfcl_memory_student.yaml \
  --system bfcl_memory_kv \
  --scenario student \
  --output results/bfcl_kv_student.json

# Evaluate with embedding matcher
llm-memory-bench evaluate results/bfcl_kv_student.json \
  --matcher embedding \
  --output results/eval_bfcl_kv_student.json

# View results
cat results/eval_bfcl_kv_student.json | jq '.metrics'
```

**Result tracking:**

The scenario is tracked in both run and evaluation results:

```json
{
  "config": {
    "system": "bfcl_memory_kv",
    "scenario": "student",
    "model": "claude-sonnet-4-20250514"
  },
  "memory_system": {
    "system": "bfcl_memory_kv",
    "scenario": "student"
  },
  "metrics": {
    "extraction_recall": 0.68,
    "extraction_precision": 0.82
  }
}
```

## Implementation Details

**File:** `src/llm_memory_bench/systems/bfcl_memory_kv.py`

**Key design decisions:**

1. **Scenario validation in `__init__`** - Fails fast if scenario missing/invalid
2. **Persona + instruction composition** - Clean separation of domain and system prompts
3. **Version info includes scenario** - Results track complete system configuration
4. **Fact extraction format** - `"key: value"` for evaluation matching

**Code structure:**
```python
class BFCLMemoryKVSystem(MemorySystem):
    def __init__(self, scenario: str | None = None):
        # Validate scenario required
        if not scenario:
            raise ValueError("BFCL systems require --scenario")
        self.scenario = scenario

    def system_prompt(self) -> str:
        # Combine persona + memory instruction
        persona = BFCL_SCENARIOS[self.scenario]
        return f"{persona}\n\n{MEMORY_INSTRUCTION_CORE_ARCHIVAL}"

    def tool_definitions(self) -> list[dict]:
        # BFCL KV tool schema
        return [...]

    def version_info(self) -> dict:
        # Track scenario in results
        return {"system": self.name, "scenario": self.scenario}
```

## Next Steps

1. **Convert BFCL memory datasets** - Use the `bfcl-memory` converter
2. **Run baseline comparisons** - Compare `bfcl_memory_kv` vs existing systems
3. **Test scenario alignment** - Compare aligned vs misaligned scenarios
4. **Implement other backends** (optional) - Add `bfcl_memory_vector` and `bfcl_memory_rec_sum`

## References

- **Design Doc:** [docs/bfcl-prompts.md](bfcl-prompts.md)
- **Prompt Comparison:** [docs/prompt-comparison.md](prompt-comparison.md)
- **BFCL Source:** https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard

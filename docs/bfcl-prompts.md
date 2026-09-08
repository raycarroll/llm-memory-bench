# BFCL Memory Prompts: Composition and Integration

## Key Insight

**BFCL scenarios are NOT generic prompt modifiers — they are integral to the BFCL system definition.**

```
Valid Systems (Each is a complete unit):
┌─────────────────────────────────────────────────────┐
│ System: simple                                       │
│ ├─ Prompt: "store important information"            │
│ └─ Tools: add_memory(fact, category)                │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ System: memoryhub                                    │
│ ├─ Prompt: MemoryHub instructions (detailed)        │
│ └─ Tools: memory(action, ...)                       │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ System: bfcl_memory_kv --scenario student           │
│ ├─ Persona: "academic-support assistant..."         │
│ ├─ Prompt: "actively manage memory... Core/Archival"│
│ └─ Tools: core_memory_add, archival_memory_search   │
└─────────────────────────────────────────────────────┘

Invalid Hybrid (DO NOT DO THIS):
┌─────────────────────────────────────────────────────┐
│ System: memoryhub --scenario student  ❌             │
│ ├─ Persona: "academic-support assistant..." (BFCL)  │
│ ├─ Prompt: MemoryHub instructions (memoryhub)       │
│ └─ Tools: memory(action, ...)  (memoryhub)          │
│                                                      │
│ This mixes BFCL persona with memoryhub prompts/tools│
│ → Not a valid comparison to either system           │
└─────────────────────────────────────────────────────┘
```

**Why:** Mixing BFCL personas with other systems' prompts creates untested hybrids that don't represent either benchmark accurately.

## Overview

BFCL Memory tests use **domain-specific system prompts** composed of three parts. The prompts are NOT just the one-liners in `MEMORY_AGENT_SETTINGS` — those are scenario personas that get combined with memory system instructions and tool schemas to form complete BFCL systems.

## BFCL Backends: Prompt and Tool Differences

The three BFCL backends differ in **both prompts and tool schemas**.

### Backend Comparison

| Aspect | memory_kv | memory_vector | memory_rec_sum |
|--------|-----------|---------------|----------------|
| **Prompt Template** | CORE_ARCHIVAL | CORE_ARCHIVAL | UNIFIED |
| **Memory Model** | Two-tier (Core + Archival) | Two-tier (Core + Archival) | Single unified |
| **Storage Interface** | Key-value pairs | Text entries with IDs | Append-only text |
| **Core Tools** | `core_memory_add(key, value)` | `core_memory_add(text)` | `memory_append(text)` |
| **Archival Tools** | `archival_memory_add(key, value)` | `archival_memory_add(text)` | None (unified) |
| **Context Format** | JSON dict | JSON array of objects | Plain text string |

## BFCL Prompt Structure

### Full System Prompt Composition

```
[SYSTEM PROMPT] =
    [SCENARIO PERSONA]
    +
    [MEMORY BACKEND INSTRUCTION]  ← Differs between KV/Vector and RecSum
    +
    [CURRENT MEMORY CONTENT]      ← Format differs per backend
```

### 1. Scenario Persona (MEMORY_AGENT_SETTINGS)

These are **role descriptions** that set the domain context:

```python
MEMORY_AGENT_SETTINGS = {
    "student": "You are an academic-support assistant for college student. Remember key personal and academic details discussed across sessions, and draw on them to answer questions or give guidance.",
    
    "customer": "You are a general customer support assistant for an e-commerce platform. Your task is to understand and remember information that can be used to provide information about user inquiries, preferences, and offer consistent, helpful assistance over multiple interactions.",
    
    "finance": "You are a high-level executive assistant supporting a senior finance professional. Retain and synthesize both personal and professional information including facts, goals, prior decisions, and family life across sessions to provide strategic, context-rich guidance and continuity.",
    
    "healthcare": "You are a healthcare assistant supporting a patient across appointments. Retain essential medical history, treatment plans, and personal preferences to offer coherent, context-aware guidance and reminders.",
    
    "notetaker": "You are a personal organization assistant. Capture key information from conversations, like tasks, deadlines, and preferences, and use it to give reliable reminders and answers in future sessions.",
}
```

**Key characteristics:**
- Domain-specific context (student support vs healthcare vs finance)
- Implicit guidance about what to remember
- No explicit tool instructions
- Sets expectations for what "important information" means in this context

### 2. Memory Backend Instruction

**Two different prompt templates** depending on backend:

#### MEMORY_BACKEND_INSTRUCTION_CORE_ARCHIVAL

**Used by:** `memory_kv`, `memory_vector`

```
{scenario_setting}

You have access to an advanced memory system, consisting of two memory types 
'Core Memory' and 'Archival Memory'. Both type of memory is persistent across 
multiple conversations with the user, and can be accessed in a later interactions. 
You should actively manage your memory data to keep track of important information, 
ensure that it is up-to-date and easy to retrieve to provide personalized responses 
to the user later.

The Core memory is limited in size, but always visible to you in context. The 
Archival Memory has a much larger capacity, but will be held outside of your 
immediate context due to its size.

Here is the content of your Core Memory from previous interactions:
{memory_content}
```

**Key details:**
- Explicitly mentions "two memory types"
- Explains Core (limited, in-context) vs Archival (large, out-of-context)
- Shows only Core memory in context (Archival must be searched)

#### MEMORY_BACKEND_INSTRUCTION_UNIFIED

**Used by:** `memory_rec_sum`

```
{scenario_setting}

You have access to an advanced memory system, which is persistent across multiple 
conversations with the user, and can be accessed in a later interactions. You 
should actively manage your memory data to keep track of important information, 
ensure that it is up-to-date and easy to retrieve to provide personalized responses 
to the user later.

Here is the content of your memory system from previous interactions:
{memory_content}
```

**Key details:**
- Says "an advanced memory system" (singular, not two types)
- No mention of Core vs Archival distinction
- Simpler prompt (no capacity explanations)

**Prompt difference impact:**
The Core+Archival prompt tells models they have two storage tiers with different characteristics, while the Unified prompt presents a single memory space.

### 3. Current Memory Content

Dynamically generated from the memory backend state:

```json
{
  "user_name": "Michael",
  "age": "35",
  "location": "Seattle",
  "occupation": "freelance graphic designer"
}
```

For empty memory (first turn):
```
There is no content in the core memory at this point.
```

## Memory Tool Schemas: Backend Differences

### memory_kv (Key-Value Store)

**Interface:** Structured key-value pairs

**Core Memory Tools:**
```python
core_memory_add(key: str, value: str)
    """Add a key-value pair to the short-term memory. Keys must be snake_case."""

core_memory_remove(key: str)
    """Remove a key-value pair from the short-term memory."""

core_memory_replace(key: str, value: str)
    """Replace the value of an existing key."""

core_memory_retrieve(key: str)
    """Retrieve the value for a specific key."""

core_memory_key_search(query: str, k: int = 5)
    """Search for keys matching the query (BM25+ similarity)."""
```

**Archival Memory Tools:**
```python
archival_memory_add(key: str, value: str)
    """Add a key-value pair to the long-term archival memory."""

archival_memory_remove(key: str)
    """Remove a key from archival memory."""

archival_memory_key_search(query: str, k: int = 5)
    """Search archival memory keys (BM25+ similarity)."""
```

**Constraints:**
- Keys must be `snake_case` (no spaces)
- Core memory: max 7 entries, 300 chars per value
- Archival memory: max 50 entries, 2000 chars per value

**Context format:**
```json
{
  "student_name": "Michael",
  "age": "35",
  "occupation": "graphic designer"
}
```

---

### memory_vector (Vector Database)

**Interface:** Text entries with auto-generated IDs, vector similarity search

**Core Memory Tools:**
```python
core_memory_add(text: str) -> dict[str, int]
    """Add a new entry to core memory. Returns the assigned ID."""

core_memory_remove(vec_id: int)
    """Remove an entry by ID."""

core_memory_update(vec_id: int, new_text: str)
    """Update an existing entry."""

core_memory_retrieve(query: str, top_k: int = 5)
    """Retrieve entries via vector similarity search."""
```

**Archival Memory Tools:**
```python
archival_memory_add(text: str) -> dict[str, int]
    """Add a new entry to archival memory. Returns ID."""

archival_memory_remove(vec_id: int)
    """Remove an entry by ID."""

archival_memory_retrieve(query: str, top_k: int = 5)
    """Retrieve entries via vector similarity search."""
```

**Constraints:**
- Core memory: max 7 entries, 300 chars per entry
- Archival memory: max 50 entries, 2000 chars per entry
- Uses vector embeddings for retrieval (not keyword-based)

**Context format:**
```json
[
  {"id": 0, "text": "Student name is Michael"},
  {"id": 1, "text": "Age is 35"},
  {"id": 2, "text": "Works as graphic designer"}
]
```

---

### memory_rec_sum (Recursive Summarization)

**Interface:** Single append-only memory, no explicit search

**Memory Tools:**
```python
memory_append(text: str)
    """Append new text to the end of the memory."""

memory_update(text: str)
    """Update/overwrite the entire memory content."""

memory_replace(old_text: str, new_text: str)
    """Replace a specific substring in the memory."""

memory_retrieve()
    """Retrieve the entire memory content."""

memory_clear()
    """Clear all memory."""
```

**Constraints:**
- No size limit specified (managed via summarization)
- No separate Core/Archival tiers
- No search tools (memory is always fully in-context via summarization)

**Context format:**
```
Student name is Michael. Age 35. Works as a freelance graphic designer 
in Seattle. Prefers metric units. Interested in advanced algorithms course.
```

---

## Key Differences Summary

| Aspect | memory_kv | memory_vector | memory_rec_sum |
|--------|-----------|---------------|----------------|
| **Storage abstraction** | Key-value dictionary | Text documents with IDs | Continuous text |
| **Organization** | Model chooses keys | Auto-assigned IDs | Model organizes text |
| **Retrieval** | Key lookup or BM25 search | Vector similarity | Full retrieval only |
| **Update model** | Replace by key | Replace by ID | Text manipulation |
| **Core/Archival** | Yes (2 tiers) | Yes (2 tiers) | No (unified) |
| **Context format** | JSON dict | JSON array | Plain text |
| **Cognitive load** | Naming keys | Managing IDs | Structuring text |

## How This Compares to Our Benchmark

### Our Existing Systems (No Scenarios)

| System | Persona | Memory Instructions | Tools | Scenario Support |
|--------|---------|---------------------|-------|------------------|
| **simple** | None | Basic: "When you learn something important, store it" | `add_memory(fact, category)` | No |
| **memoryhub** | None | Detailed: When to store, when not to, examples | `memory(action, ...)` with 6+ actions | No |
| **claude_code** | None | Claude Code's actual auto-memory instructions | `memory(...)` | No |

**Characteristics:**
- Domain-agnostic (no persona)
- Designed to generalize across contexts
- One prompt works for all datasets

### BFCL Systems (Scenarios Required)

| System | Persona | Memory Instructions | Tools | Scenarios |
|--------|---------|---------------------|-------|-----------|
| **bfcl_memory_kv** | **Domain-specific** via --scenario | "actively manage your memory data" (Core + Archival) | `core_memory_add/remove/update`, `archival_memory_add/search` | student, customer, finance, healthcare, notetaker |
| **bfcl_memory_vector** | Same personas | Same | Vector search tools | Same 5 |
| **bfcl_memory_rec_sum** | Same personas | Simplified (unified memory) | Recursive summary tools | Same 5 |

**Characteristics:**
- Domain-specific (persona sets context)
- Designed for scenario-specific deployments
- Tests if prompt specificity helps memory judgment

### Key Architectural Difference

**Our systems:** One prompt → all domains
```
simple → AlpsBench (general conversations)
simple → BFCL student data (academic context)
simple → BFCL healthcare data (medical context)
// Same prompt everywhere
```

**BFCL systems:** Different persona per domain
```
bfcl_memory_kv --scenario student → BFCL student data
bfcl_memory_kv --scenario healthcare → BFCL healthcare data
// Different personas, same core instructions
```

### What This Tests

**Question:** Does domain-specific context improve memory judgment?

**Experiment:**
```bash
# Generic system on domain-specific data
llm-memory-bench run --dataset bfcl_memory_student.yaml --system simple
# → Baseline: 28% recall

# Domain-specific BFCL system on matching data
llm-memory-bench run --dataset bfcl_memory_student.yaml --system bfcl_memory_kv --scenario student
# → Test: Does "academic-support assistant" persona improve recall?

# Domain-specific BFCL system on mismatched data
llm-memory-bench run --dataset bfcl_memory_student.yaml --system bfcl_memory_kv --scenario healthcare
# → Control: Does wrong persona hurt?
```

**Hypothesis:** BFCL's domain personas improve recall by setting expectations about what information is "important" in that context.

## Integration Design: The Right Way

**CRITICAL:** Scenarios are parameters of the BFCL system, NOT generic modifiers.

### ❌ WRONG: Mixing BFCL personas with other systems

```bash
# This creates an invalid hybrid
llm-memory-bench run --dataset bfcl_memory_student.yaml --system memoryhub --scenario student
# → memoryhub instructions + BFCL student persona = INVALID
```

**Why this is wrong:**
- Invalidates benchmarking (not comparing real systems)
- Creates untested hybrid prompts
- Can't attribute results to either system

### ✅ CORRECT: Scenario as BFCL system parameter

BFCL scenarios are part of the BFCL memory system definition.

## Implementation Options

### Option 1: Separate System Classes (Explicit)

Create one system class per BFCL scenario:

```python
# src/llm_memory_bench/systems/bfcl_memory_kv_student.py
class BFCLMemoryKVStudentSystem(MemorySystem):
    def system_prompt(self) -> str:
        persona = "You are an academic-support assistant for college student..."
        memory_instruction = "You have access to an advanced memory system..."
        return f"{persona}\n\n{memory_instruction}"
    
    def tool_schema(self) -> list[dict]:
        return [
            {
                "name": "core_memory_add",
                "description": "Add a key-value pair to the short-term memory...",
                "parameters": {...}
            },
            # ... other BFCL KV tools
        ]
```

**File structure:**
```
systems/
  ├── bfcl_memory_kv_student.py
  ├── bfcl_memory_kv_customer.py
  ├── bfcl_memory_kv_healthcare.py
  ├── bfcl_memory_kv_finance.py
  ├── bfcl_memory_kv_notetaker.py
  ├── memoryhub.py  # Separate, no scenarios
  └── simple.py     # Separate, no scenarios
```

**Usage:**
```bash
llm-memory-bench run --dataset bfcl_memory_student.yaml --system bfcl_memory_kv_student
llm-memory-bench run --dataset bfcl_memory_student.yaml --system memoryhub  # Different system entirely
```

**Benefits:**
- ✅ Crystal clear: each name is a complete, valid system
- ✅ No mixing possible
- ✅ Easy to audit what prompt was actually used
- ✅ Results show exact system tested

**Drawbacks:**
- ❌ Code duplication across scenario classes (share tool schema)
- ❌ 15 system files (3 backends × 5 scenarios)

### Option 2: Parameterized BFCL System (Compact)

One BFCL system class per backend, scenario as parameter:

```python
# src/llm_memory_bench/systems/bfcl_memory_kv.py
BFCL_SCENARIOS = {
    "student": "You are an academic-support assistant for college student...",
    "customer": "You are a general customer support assistant...",
    # ... all 5
}

class BFCLMemoryKVSystem(MemorySystem):
    def __init__(self, config: dict):
        self.scenario = config.get("scenario", "student")
        if self.scenario not in BFCL_SCENARIOS:
            raise ValueError(f"Invalid BFCL scenario: {self.scenario}")
    
    def system_prompt(self) -> str:
        persona = BFCL_SCENARIOS[self.scenario]
        memory_instruction = self._get_bfcl_memory_instruction()
        return f"{persona}\n\n{memory_instruction}"
    
    def tool_schema(self) -> list[dict]:
        return self._get_bfcl_kv_tools()
```

**Usage:**
```bash
# Scenario required for BFCL systems
llm-memory-bench run --dataset bfcl_memory_student.yaml --system bfcl_memory_kv --scenario student

# Other systems don't accept --scenario (error if provided)
llm-memory-bench run --dataset alpsbench-task1.yaml --system memoryhub  # No --scenario
```

**Benefits:**
- ✅ Less code duplication (3 system files instead of 15)
- ✅ Scenario validation at system level
- ✅ Clear separation (only BFCL systems accept scenario)

**Drawbacks:**
- ❌ Scenario parameter required for BFCL systems
- ❌ Results need both system + scenario to be meaningful

## Which Backend to Implement First?

### Recommendation: Start with memory_kv

**Reasons:**
1. **Simplest tool schema** - Key-value is familiar, widely used
2. **Most structured** - Explicit keys make inspection easier
3. **Debugging friendly** - Can see exactly what's stored under each key
4. **Direct comparison** - Similar to our existing `simple` system (both use add/remove)
5. **Same prompt as vector** - Both use CORE_ARCHIVAL instruction (only tools differ)

**Implementation priority:**
1. ✅ **memory_kv** - Start here (5 scenarios = 1 system class)
2. ⏸️ **memory_vector** - Add later if testing retrieval mechanisms
3. ⏸️ **memory_rec_sum** - Add later if testing unstructured storage

### Backend Selection for Research Questions

| Research Question | Backends Needed | Why |
|------------------|-----------------|-----|
| **Do domain personas improve storage judgment?** | memory_kv only | Any backend works; KV is simplest |
| **How does BFCL compare to memoryhub?** | memory_kv only | Simplest apples-to-apples comparison |
| **Does two-tier memory (Core+Archival) help?** | memory_kv + memory_rec_sum | KV has tiers, rec_sum is unified |
| **KV vs vector vs summarization - which is best?** | All 3 backends | Direct backend comparison |
| **Does retrieval mechanism matter for storage?** | memory_kv + memory_vector | Same prompt, different retrieval |

**For initial benchmarking:** Implement only `bfcl_memory_kv` with all 5 scenarios. This gives us:
- 5 complete BFCL systems (one per scenario)
- Comparison to our existing systems (simple, memoryhub)
- Test of domain-specific persona impact

Add other backends only if specific research questions require them.

## Recommended Approach

**Use Option 2 (Parameterized BFCL System) with validation:**

### 1. System Implementation

```python
# src/llm_memory_bench/systems/bfcl_memory_kv.py
BFCL_SCENARIOS = {
    "student": "You are an academic-support assistant for college student...",
    "customer": "You are a general customer support assistant for an e-commerce platform...",
    "finance": "You are a high-level executive assistant supporting a senior finance professional...",
    "healthcare": "You are a healthcare assistant supporting a patient across appointments...",
    "notetaker": "You are a personal organization assistant...",
}

class BFCLMemoryKVSystem(MemorySystem):
    """BFCL Memory with Key-Value backend."""
    
    def __init__(self, config: dict):
        # Scenario is REQUIRED for BFCL systems
        self.scenario = config.get("scenario")
        if not self.scenario:
            raise ValueError("BFCL systems require --scenario parameter")
        if self.scenario not in BFCL_SCENARIOS:
            raise ValueError(f"Invalid scenario: {self.scenario}. Must be one of: {list(BFCL_SCENARIOS.keys())}")
    
    def system_prompt(self) -> str:
        persona = BFCL_SCENARIOS[self.scenario]
        memory_instruction = """
You have access to an advanced memory system, consisting of two memory types 
'Core Memory' and 'Archival Memory'. Both type of memory is persistent across 
multiple conversations with the user, and can be accessed in a later interactions. 
You should actively manage your memory data to keep track of important information, 
ensure that it is up-to-date and easy to retrieve to provide personalized responses 
to the user later.

The Core memory is limited in size, but always visible to you in context. The 
Archival Memory has a much larger capacity, but will be held outside of your 
immediate context due to its size.
"""
        return f"{persona}\n\n{memory_instruction}"
    
    def tool_schema(self) -> list[dict]:
        return [
            {
                "name": "core_memory_add",
                "description": "Add a key-value pair to the short-term memory. Make sure to use meaningful keys for easy retrieval later.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": "The key under which the value is stored. Keys must be snake_case and cannot contain spaces."
                        },
                        "value": {
                            "type": "string",
                            "description": "The value to store in the short-term memory."
                        }
                    },
                    "required": ["key", "value"]
                }
            },
            # ... other BFCL KV tools
        ]
```

### 2. CLI Usage

```bash
# BFCL systems require --scenario
llm-memory-bench run \
  --dataset datasets/converted/bfcl_memory_student.yaml \
  --system bfcl_memory_kv \
  --scenario student

# Error if scenario missing
llm-memory-bench run --dataset bfcl_memory_student.yaml --system bfcl_memory_kv
# → ERROR: BFCL systems require --scenario parameter

# Error if scenario invalid
llm-memory-bench run --dataset bfcl_memory_student.yaml --system bfcl_memory_kv --scenario invalid
# → ERROR: Invalid scenario: invalid. Must be one of: ['student', 'customer', ...]

# Non-BFCL systems don't accept --scenario
llm-memory-bench run --dataset alpsbench-task1.yaml --system memoryhub
# → OK: No scenario needed

llm-memory-bench run --dataset alpsbench-task1.yaml --system memoryhub --scenario student
# → ERROR: System 'memoryhub' does not support --scenario parameter
```

### 3. Dataset Metadata (Optional Hints)

```yaml
# datasets/converted/bfcl_memory_student.yaml
metadata:
  name: "BFCL Memory - Student Support"
  source: "BFCL v4 memory category"
  recommended_system: "bfcl_memory_kv"      # Hint for which system to use
  recommended_scenario: "student"            # Hint for which scenario
  description: "College student academic support conversations"
```

**CLI can use hints:**
```bash
# Auto-detects from metadata
llm-memory-bench run --dataset bfcl_memory_student.yaml --system bfcl_memory_kv
# → Uses scenario=student from metadata

# Override if needed
llm-memory-bench run --dataset bfcl_memory_student.yaml --system bfcl_memory_kv --scenario customer
# → Warning: Dataset recommends scenario=student, using customer
```

### 4. Results Tracking

```json
{
  "config": {
    "system": "bfcl_memory_kv",
    "scenario": "student",
    "dataset": "bfcl_memory_student.yaml"
  },
  "metrics": {
    "extraction_recall": 0.72,
    "extraction_precision": 0.85
  }
}
```

## Expected Results Pattern

### Cross-System Comparison (Same Dataset)

Test different complete systems on the same BFCL student dataset:

```
Dataset: bfcl_memory_student.yaml

System: simple (basic memory guidance)
  → 65% recall

System: bfcl_memory_kv --scenario student (BFCL KV + student persona)
  → 65% recall

System: memoryhub (MemoryHub's own prompts, no BFCL persona)
  → 72% recall
```

**Comparison validity:**
- ✅ simple vs memoryhub → compares basic vs sophisticated prompts
- ✅ bfcl_memory_kv vs simple → compares domain-specific vs generic
- ✅ bfcl_memory_kv vs memoryhub → compares complete systems
- ❌ bfcl_memory_kv + memoryhub prompts → INVALID HYBRID

### Scenario Alignment Impact (BFCL Systems Only)

Test BFCL system with different scenarios on matching/mismatching datasets:

```
System: bfcl_memory_kv

Student dataset + --scenario student:     72% recall (aligned)
Student dataset + --scenario healthcare:  58% recall (misaligned, -14%)
Student dataset + --scenario finance:     62% recall (misaligned, -10%)

Healthcare dataset + --scenario healthcare: 75% recall (aligned)
Healthcare dataset + --scenario student:    61% recall (misaligned, -14%)
```

**Hypothesis:** BFCL personas set domain-specific expectations about what's "important."

### System × Dataset Matrix

Each system on each dataset (BFCL systems use aligned scenario):

| Dataset | simple | bfcl_memory_kv | memoryhub |
|---------|--------|----------------|-----------|
| **alpsbench-task1** | 65% | N/A* | 72% |
| **bfcl_memory_student** | 65% | 65%** | 72% |
| **bfcl_memory_healthcare** | 65% | 68%** | 74% |
| **bfcl_memory_finance** | 65% | 62%** | 70% |

\* BFCL systems require scenario, AlpsBench doesn't have domain context  
\** Using aligned scenario for each dataset

**Expected patterns:**
- memoryhub outperforms across datasets (more sophisticated prompts)
- BFCL systems competitive with simple on domain-specific data
- Domain-specific scenarios may help or hurt depending on alignment

## Implementation Checklist

### Core BFCL System Implementation

- [ ] Create `src/llm_memory_bench/systems/bfcl_memory_kv.py`
  - [ ] Add `BFCL_SCENARIOS` dict with 5 personas
  - [ ] Implement `__init__` with scenario validation
  - [ ] Implement `system_prompt()` combining persona + memory instruction
  - [ ] Implement `tool_schema()` with BFCL KV tools (core_memory_add, etc.)

- [ ] Create `src/llm_memory_bench/systems/bfcl_memory_vector.py`
  - [ ] Share `BFCL_SCENARIOS` (move to shared constants)
  - [ ] Implement vector-based tool schema

- [ ] Create `src/llm_memory_bench/systems/bfcl_memory_rec_sum.py`
  - [ ] Use `MEMORY_BACKEND_INSTRUCTION_UNIFIED` template
  - [ ] Implement recursive summarization tool schema

### CLI and Configuration

- [ ] Update CLI to accept `--scenario` parameter
  - [ ] Only allow for systems that declare scenario support
  - [ ] Error if scenario provided to non-BFCL system
  - [ ] Error if scenario missing for BFCL system

- [ ] Update dataset metadata schema
  - [ ] Add optional `recommended_system` field
  - [ ] Add optional `recommended_scenario` field
  - [ ] Update BFCL dataset conversions to include hints

- [ ] Update `run` command
  - [ ] Auto-detect scenario from dataset metadata if not provided
  - [ ] Warn if explicit scenario differs from recommended
  - [ ] Pass scenario to system via config dict

### Results and Validation

- [ ] Update results serialization
  - [ ] Include scenario in config block
  - [ ] Track system + scenario as composite key

- [ ] Update `evaluate` command
  - [ ] Show scenario in output header
  - [ ] Group results by system + scenario combination

- [ ] Add validation
  - [ ] Prevent mixing BFCL scenarios with non-BFCL systems
  - [ ] Clear error messages for invalid configurations

### Documentation

- [ ] Update README with BFCL system examples
- [ ] Add scenario parameter docs to CLI help
- [ ] Document valid system + scenario combinations
- [ ] Add example comparison: bfcl_memory_kv vs memoryhub

## Summary: Systems vs Scenarios

### What is a "System"?

A complete system = persona + memory instructions + tool schema

**Our existing systems:**
- `simple` = basic memory-aware prompt + simple add_memory tool
- `memoryhub` = MemoryHub prompts + MemoryHub tool set
- `claude_code` = Claude Code auto-memory prompts + memory tool

**New BFCL systems:**
- `bfcl_memory_kv` = BFCL persona (via --scenario) + BFCL KV instructions + BFCL KV tools
- `bfcl_memory_vector` = BFCL persona (via --scenario) + BFCL vector instructions + BFCL vector tools
- `bfcl_memory_rec_sum` = BFCL persona (via --scenario) + BFCL summarization instructions + BFCL summary tools

### What is a "Scenario"?

A parameter of BFCL systems that selects the domain persona. Not applicable to other systems.

**Valid scenarios (BFCL only):**
- `student` - Academic support assistant
- `customer` - E-commerce support
- `finance` - Executive assistant
- `healthcare` - Patient care assistant
- `notetaker` - Personal organization

### Valid Comparisons

```bash
# Comparing complete systems on same dataset
llm-memory-bench run --dataset bfcl_memory_student.yaml --system simple
llm-memory-bench run --dataset bfcl_memory_student.yaml --system memoryhub
llm-memory-bench run --dataset bfcl_memory_student.yaml --system bfcl_memory_kv --scenario student

# Comparing BFCL scenarios on same dataset
llm-memory-bench run --dataset bfcl_memory_student.yaml --system bfcl_memory_kv --scenario student
llm-memory-bench run --dataset bfcl_memory_student.yaml --system bfcl_memory_kv --scenario customer

# Comparing BFCL backends with same scenario
llm-memory-bench run --dataset bfcl_memory_student.yaml --system bfcl_memory_kv --scenario student
llm-memory-bench run --dataset bfcl_memory_student.yaml --system bfcl_memory_vector --scenario student
llm-memory-bench run --dataset bfcl_memory_student.yaml --system bfcl_memory_rec_sum --scenario student
```

### Invalid Comparisons

```bash
# ❌ Applying BFCL scenario to non-BFCL system
llm-memory-bench run --dataset bfcl_memory_student.yaml --system memoryhub --scenario student
# → ERROR: System 'memoryhub' does not support --scenario

# ❌ BFCL system without scenario
llm-memory-bench run --dataset bfcl_memory_student.yaml --system bfcl_memory_kv
# → ERROR: BFCL systems require --scenario parameter (unless auto-detected from metadata)
```

## References

- **BFCL Source:** `/tmp/bfcl-check/berkeley-function-call-leaderboard/bfcl_eval/constants/default_prompts.py`
- **Memory System Implementation:** `bfcl_eval/model_handler/utils.py` (line 610-650)
- **KV Memory Backend:** `bfcl_eval/eval_checker/multi_turn_eval/func_source_code/memory_kv.py`

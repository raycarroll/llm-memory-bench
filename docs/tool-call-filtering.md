# Tool Call Filtering in Evaluation

## Problem

The extraction benchmark tests **proactive storage judgment**, not retrieval behavior. Search/read/query tool calls should not affect metrics.

Previously, the evaluator counted ALL tool calls, including retrieval:
- ❌ Search calls on noise turns counted as false positives
- ❌ Search calls inflated total_tool_calls metric
- ❌ Search calls appeared in evaluation output

## Solution

The evaluator now filters tool calls early via `extract_stored_fact()`:

```python
# Only count tool calls that return non-empty content
tool_calls = [
    tc for tc in all_tool_calls
    if memory_system.extract_stored_fact(tc)
]
```

Each memory system's `extract_stored_fact()` returns empty string for retrieval actions.

## System Behaviors

### MemoryHub
```python
if action in ("write", "update"):
    return content
# search, list, read, similar, relationships, status → empty string
```

### BFCL Memory KV
```python
if tool_call.name in ("core_memory_add", "core_memory_replace", "archival_memory_add"):
    return value
# archival_memory_key_search, core_memory_remove → empty string
```

### GBrain
All tools are storage (no retrieval tools exposed):
- `remember` → returns fact
- `put_page` → returns content
- `capture` → returns content

### Claude Code
Only storage tool:
- `save_memory` → returns body

### Simple
Only storage tool:
- `add_memory` → returns fact

### OpenClaw
Only file writes to memory paths:
- `write(MEMORY.md)` → returns content
- `write(src/main.py)` → empty string (not a memory file)
- `edit(MEMORY.md)` → returns newText

### Behavioral Relevance
Only storage tool:
- `store_fact` → returns fact

## Testing

```bash
# After the fix, search calls won't appear in evaluation output
llm-memory-bench evaluate results/run_*.json --matcher embedding

# Check that noise turns with only search calls show 0 false positives
cat results/eval_*.json | jq '.conversations[0].turn_evaluations[] | select(.is_noise_turn == true)'
```

## Impact on Metrics

**Before fix:**
- total_tool_calls = storage + retrieval calls
- noise_violations = any tool call on noise turn (including search)
- false_positives = noise violations (inflated by search)

**After fix:**
- total_tool_calls = storage calls only
- noise_violations = storage calls on noise turns
- false_positives = noise violations (accurate)

Search/retrieval calls are completely ignored for evaluation.

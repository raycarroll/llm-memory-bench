# Sweep Systems

Memory system variants that combine turn-by-turn extraction with post-conversation sweep for higher recall.

## Overview

Sweep systems execute in two phases:

1. **Streaming Phase**: Normal turn-by-turn conversation with memory tool usage (realistic)
2. **Sweep Phase**: Post-conversation extraction pass with high-recall prompt (comprehensive)

This provides:
- Realistic streaming behavior during conversation
- High recall coverage from retrospective analysis
- Fair comparison with batch-only systems

## Available Sweep Systems

### memoryhub_sweep

MemoryHub with final extraction sweep.

```bash
llm-memory-bench run \
  --dataset alpsbench \
  --model gpt-4.1-mini \
  --system memoryhub_sweep
```

### simple_sweep

Simple memory system with final extraction sweep.

```bash
llm-memory-bench run \
  --dataset alpsbench \
  --model gpt-4.1-mini \
  --system simple_sweep
```

### gbrain_sweep

GBrain with final extraction sweep.

```bash
llm-memory-bench run \
  --dataset alpsbench \
  --model gpt-4.1-mini \
  --system gbrain_sweep
```

## Execution Flow

```
Conversation: User → Agent (turn 1)
              User → Agent (turn 2)
              ...
              User → Agent (turn N)
              
↓ Streaming phase complete

Final Sweep: LLM reviews ALL turns with extraction prompt
             Extracts any missed memories
             
↓ Sweep complete

Results: Turn-by-turn tool calls + sweep tool calls
```

## Comparison

### Streaming Only (realistic)
```bash
llm-memory-bench run --system memoryhub
# Result: ~3-5% recall (conservative, real-time decisions)
```

### Streaming + Sweep (realistic + comprehensive)
```bash
llm-memory-bench run --system memoryhub_sweep
# Result: ~35-45% recall (realistic behavior + high coverage)
```

### Batch Only (academic baseline)
```bash
llm-memory-bench run --system alpsbench_batch
# Result: ~45-50% recall (maximum, unrealistic for live agents)
```

## When to Use

**Use streaming only (`memoryhub`, `simple`):**
- Testing production agent behavior
- Measuring real-time extraction quality
- Evaluating agent responsiveness

**Use streaming + sweep (`memoryhub_sweep`, `simple_sweep`):**
- Fair comparison with batch systems
- Maximum recall while maintaining realistic behavior
- Hybrid evaluation (behavior + coverage)

**Use batch only (`alpsbench_batch`):**
- Academic benchmark comparison
- Maximum theoretical recall
- Post-conversation analysis use cases

## Implementation Details

Sweep systems inherit from base systems:

```python
# systems/memoryhub_sweep.py
from .memoryhub import MemoryHubMemorySystem

class MemoryHubSweepSystem(MemoryHubMemorySystem):
    name = "memoryhub_sweep"
    
    def requires_final_sweep(self) -> bool:
        return True
    
    def final_sweep_prompt(self) -> str:
        return "Extract ALL memories with HIGH RECALL..."
```

The runner automatically detects `requires_final_sweep()` and executes the sweep phase.

## Results Structure

Sweep results include an additional "sweep" turn:

```json
{
  "conversation_id": "conv_123",
  "turn_results": [
    {"turn_index": 0, "role": "user", "tool_calls": []},
    {"turn_index": 1, "role": "assistant", "tool_calls": [...]},
    ...
    {"turn_index": 15, "role": "sweep", "tool_calls": [...]}
  ]
}
```

The sweep turn contains all memories extracted during the final pass.

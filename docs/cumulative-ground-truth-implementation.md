# Cumulative Ground Truth Implementation

## Status: ✅ COMPLETE - Schema ✓ | Evaluation ✓ | Metrics ✓ | Converter ✓ | Docs ✓

## Overview

Adding support for cumulative ground truth to enable BFCL memory dataset usage alongside per-turn datasets like AlpsBench.

## Completed: Schema Changes ✓

### Dataset Types

```python
class GroundTruthType(str, Enum):
    PER_TURN = "per_turn"      # AlpsBench-style
    CUMULATIVE = "cumulative"   # BFCL-style
```

### Dataset Model

```python
class Dataset(BaseModel):
    conversations: list[Conversation]
    ground_truth_type: GroundTruthType = GroundTruthType.PER_TURN  # Default
```

### Conversation Model

```python
class Conversation(BaseModel):
    id: str
    turns: list[Turn]
    cumulative_ground_truth: GroundTruth | None = None  # Only for cumulative datasets
```

### Validation ✓

**Enforces mutual exclusivity:**
- Per-turn datasets: `cumulative_ground_truth` must be None
- Cumulative datasets: `cumulative_ground_truth` must be present, per-turn ground truth must be empty

**Error messages:**
```
ValueError: Dataset type is 'per_turn' but conversation 'conv_1' has cumulative_ground_truth.
ValueError: Dataset type is 'cumulative' but conversation 'conv_1' has no cumulative_ground_truth.
ValueError: Dataset type is 'cumulative' but conversation 'conv_1' turn 0 has per-turn ground_truth.
```

---

## TODO: Evaluation Logic

### 1. Update Evaluator Entry Point

**File:** `src/llm_memory_bench/evaluator.py`

```python
async def evaluate_run(
    dataset: Dataset,
    run_result: RunResult,
    matcher: FactMatcher | None = None,
    judge_config: RunConfig | None = None,
) -> EvaluationResult:
    # ... existing setup ...
    
    # Route based on dataset type
    for conv_result in run_result.conversation_results:
        conversation = conv_map.get(conv_result.conversation_id)
        if not conversation:
            continue
        
        if dataset.ground_truth_type == GroundTruthType.PER_TURN:
            conv_eval = await evaluate_conversation_per_turn(
                conversation, conv_result, memory_system, matcher
            )
        else:
            conv_eval = await evaluate_conversation_cumulative(
                conversation, conv_result, memory_system, matcher
            )
        
        eval_result.conversation_evaluations.append(conv_eval)
    
    return eval_result
```

### 2. Rename Existing Function

```python
# Rename: evaluate_conversation → evaluate_conversation_per_turn
async def evaluate_conversation_per_turn(
    conversation: Conversation,
    conv_result: ConversationResult,
    memory_system: MemorySystem,
    matcher: FactMatcher,
) -> ConversationEvaluation:
    # ... existing per-turn logic ...
```

### 3. Add Cumulative Evaluator

**File:** `src/llm_memory_bench/evaluator.py`

```python
async def evaluate_conversation_cumulative(
    conversation: Conversation,
    conv_result: ConversationResult,
    memory_system: MemorySystem,
    matcher: FactMatcher,
) -> ConversationEvaluation:
    """Evaluate cumulative ground truth - timing doesn't matter, only completeness."""
    
    # Expected facts (from cumulative ground truth)
    if not conversation.cumulative_ground_truth:
        raise ValueError(f"Cumulative evaluation requires cumulative_ground_truth")
    
    expected_facts = conversation.cumulative_ground_truth.should_store
    
    # Collect ALL facts stored across entire conversation
    all_stored_facts: list[str] = []
    for turn_result in conv_result.turn_results:
        for tool_call in turn_result.tool_calls:
            fact = memory_system.extract_stored_fact(tool_call)
            if fact:
                all_stored_facts.append(fact)
    
    # Match all stored vs all expected (order doesn't matter)
    tp_count = 0
    partial_count = 0
    matched_indices: set[int] = set()
    
    for expected in expected_facts:
        best_verdict = MatchVerdict.NO_MATCH
        best_idx = None
        
        # Find best match among all stored facts
        for idx, stored in enumerate(all_stored_facts):
            if idx in matched_indices:
                continue  # Already matched to another expected fact
            
            verdict = await matcher.match(expected.fact, stored)
            if verdict.value > best_verdict.value:
                best_verdict = verdict
                best_idx = idx
        
        if best_verdict == MatchVerdict.MATCH:
            tp_count += 1
            if best_idx is not None:
                matched_indices.add(best_idx)
        elif best_verdict == MatchVerdict.PARTIAL:
            partial_count += 1
            if best_idx is not None:
                matched_indices.add(best_idx)
    
    # Calculate metrics
    fp_count = len(all_stored_facts) - len(matched_indices)  # Unmatched stored facts
    fn_count = len(expected_facts) - tp_count - (partial_count * 0.5)  # Unmatched expected facts
    
    # Adjust TP for partials
    adjusted_tp = tp_count + (partial_count * 0.5)
    
    return ConversationEvaluation(
        conversation_id=conversation.id,
        true_positives=adjusted_tp,
        false_positives=fp_count,
        false_negatives=fn_count,
        total_tool_calls=len(all_stored_facts),
        noise_turns=0,  # Not applicable for cumulative
        noise_violations=0,  # Not applicable for cumulative
    )
```

---

## TODO: Metric Naming

### Separate Metric Names by Type

**File:** `src/llm_memory_bench/evaluator.py`

```python
def compute_metrics(
    eval_result: EvaluationResult,
    dataset_type: GroundTruthType,
) -> dict:
    """Compute metrics with names that reflect the evaluation type."""
    
    # ... calculate base values ...
    
    if dataset_type == GroundTruthType.PER_TURN:
        return {
            "per_turn_extraction_recall": recall,
            "per_turn_extraction_precision": precision,
            "per_turn_extraction_f1": f1,
            "per_turn_noise_resistance_rate": noise_resistance,
            "schema_validity_rate": schema_validity,
            # ... other metrics
        }
    else:  # CUMULATIVE
        return {
            "cumulative_extraction_recall": recall,
            "cumulative_extraction_precision": precision,
            "cumulative_extraction_f1": f1,
            "schema_validity_rate": schema_validity,
            # Note: noise_resistance not applicable for cumulative
            # ... other metrics
        }
```

### Metric Interpretation

**Per-turn metrics:**
- `per_turn_extraction_recall`: Of facts that should be stored at specific turns, how many were?
- `per_turn_extraction_precision`: Of facts stored at each turn, how many were correct for that turn?
- `per_turn_noise_resistance_rate`: How often did model correctly NOT store on noise turns?

**Cumulative metrics:**
- `cumulative_extraction_recall`: Of all expected facts, how many were stored (regardless of when)?
- `cumulative_extraction_precision`: Of all facts stored, how many were expected?
- `noise_resistance_rate`: Not applicable (no per-turn expectations)

### Output Format

```json
{
  "config": {
    "dataset": "bfcl_memory_student.yaml",
    "ground_truth_type": "cumulative"
  },
  "metrics": {
    "cumulative_extraction_recall": 0.82,
    "cumulative_extraction_precision": 0.76,
    "cumulative_extraction_f1": 0.79,
    "schema_validity_rate": 0.98
  }
}
```

---

## TODO: BFCL Converter Update

**File:** `src/llm_memory_bench/adapters/bfcl_memory.py`

### Changes Needed

1. **Remove per-turn fact mapping logic** - Don't try to map facts to specific turns
2. **Create cumulative ground truth** - All questions become cumulative expected facts
3. **Set dataset type** - Mark as cumulative

```python
def convert_bfcl_memory(...):
    # ... load BFCL data ...
    
    for scenario in scenarios:
        # Filter questions for this scenario
        scenario_questions = [q for q in all_questions if q.get("scenario") == scenario]
        scenario_answers = [
            a for a in all_answers
            if any(q["id"] == a["id"] for q in scenario_questions)
        ]
        
        conversations = []
        
        for entry in prereq_entries:
            # Extract turns (no per-turn ground truth)
            turns = []
            for turn_group in entry.get("question", []):
                if isinstance(turn_group, list):
                    for turn in turn_group:
                        turns.append({
                            "role": turn.get("role", "user"),
                            "content": turn.get("content", ""),
                            # NO ground_truth here
                        })
            
            # Derive cumulative ground truth from all questions
            cumulative_facts = []
            for question in scenario_questions:
                answer = answer_map.get(question["id"])
                if not answer:
                    continue
                
                ground_truth = answer.get("ground_truth", [])
                if not ground_truth:
                    continue
                
                question_content = question["question"][0][0]["content"]
                fact_description = _derive_fact_description(
                    question_content,
                    ground_truth,
                    answer.get("source", "")
                )
                
                cumulative_facts.append({
                    "fact": fact_description,
                    "type": "direct",
                    "bfcl_question_id": question["id"],
                })
            
            conversation = {
                "id": entry_id,
                "source": f"bfcl-memory-{scenario}",
                "cumulative_ground_truth": {
                    "should_store": cumulative_facts,
                    "should_not_store": [],
                },
                "turns": turns,
            }
            
            conversations.append(conversation)
        
        # Write with type marker
        output_data = {
            "ground_truth_type": "cumulative",
            "conversations": conversations,
        }
        
        with open(output_file, "w") as f:
            yaml.dump(output_data, f, ...)
```

---

## TODO: Documentation

### 1. Create Dataset Types Guide

**File:** `docs/dataset-types.md`

```markdown
# Dataset Types

## Per-Turn Ground Truth

**Used by:** AlpsBench

**Structure:**
```yaml
ground_truth_type: per_turn
conversations:
  - turns:
      - content: "I'm a data scientist"
        ground_truth:
          should_store:
            - fact: "User is a data scientist"
```

**Tests:** Proactive judgment timing - WHEN to store facts

**Metrics:**
- `per_turn_extraction_recall`
- `per_turn_extraction_precision`
- `per_turn_noise_resistance_rate`

---

## Cumulative Ground Truth

**Used by:** BFCL Memory

**Structure:**
```yaml
ground_truth_type: cumulative
conversations:
  - cumulative_ground_truth:
      should_store:
        - fact: "User's dog is Archie"
        - fact: "User is a CS student"
    turns:
      - content: "..." # Archie mentioned somewhere
      - content: "..." # CS major mentioned somewhere
```

**Tests:** Storage completeness - WHAT gets stored (timing irrelevant)

**Metrics:**
- `cumulative_extraction_recall`
- `cumulative_extraction_precision`
- `cumulative_extraction_f1`

**Note:** `noise_resistance_rate` not applicable (no per-turn expectations)
```

### 2. Update Main README

Add section explaining dataset types and when to use each.

### 3. Update Metrics Documentation

**File:** `docs/metrics.md`

Add separate sections for per-turn vs cumulative metric definitions.

---

## Implementation Checklist

- [x] Add `GroundTruthType` enum
- [x] Add `ground_truth_type` to Dataset
- [x] Add `cumulative_ground_truth` to Conversation
- [x] Add validation to prevent mixing
- [x] Update `Dataset.summary()` to handle both types
- [x] Split evaluation logic (per-turn vs cumulative)
- [x] Implement `evaluate_conversation_cumulative()`
- [x] Update metric computation with type-specific names
- [x] Update BFCL converter to use cumulative ground truth
- [x] Create `docs/dataset-types.md`
- [x] Test validation errors trigger correctly
- [x] Update `docs/metrics.md` with cumulative metrics
- [ ] Update main README
- [ ] Test with BFCL data conversion (requires BFCL data - not available in environment)
- [ ] Verify AlpsBench still works (backward compatibility - requires AlpsBench datasets)

---

## Testing Plan

### 1. Validation Tests

```python
# Should pass
dataset_per_turn = Dataset(
    ground_truth_type=GroundTruthType.PER_TURN,
    conversations=[
        Conversation(
            id="test",
            turns=[Turn(content="...", ground_truth=GroundTruth(...))]
        )
    ]
)

# Should fail - mixing types
dataset_mixed = Dataset(
    ground_truth_type=GroundTruthType.PER_TURN,
    conversations=[
        Conversation(
            id="test",
            cumulative_ground_truth=GroundTruth(...),  # ERROR!
            turns=[...]
        )
    ]
)
```

### 2. Evaluation Tests

```python
# Cumulative: Should match regardless of when stored
conversation = Conversation(
    id="test",
    cumulative_ground_truth=GroundTruth(should_store=[
        ExpectedFact(fact="User is a data scientist"),
        ExpectedFact(fact="User lives in Seattle"),
    ]),
    turns=[
        Turn(content="I'm a data scientist in Seattle")
    ]
)

conv_result = ConversationResult(
    conversation_id="test",
    turn_results=[
        TurnResult(
            tool_calls=[
                ToolCall(name="add_memory", arguments={"fact": "Data scientist"}),
                ToolCall(name="add_memory", arguments={"fact": "Lives in Seattle"}),
            ]
        )
    ]
)

# Should get high recall/precision (both facts stored)
```

### 3. BFCL Conversion Test

```bash
llm-memory-bench convert --source bfcl-memory --bfcl-repo /path/to/gorilla

# Check output:
# - ground_truth_type: cumulative
# - cumulative_ground_truth present
# - No per-turn ground_truth
# - Validation passes
```

### 4. Backward Compatibility Test

```bash
# Existing AlpsBench datasets should still work
llm-memory-bench run --dataset datasets/converted/alpsbench-task1.yaml --system simple

# Should use per-turn evaluation automatically
```

---

## Migration Guide

### For Existing AlpsBench Datasets

No changes needed! Datasets without `ground_truth_type` default to `per_turn`.

### For New BFCL Datasets

Use the converter:
```bash
llm-memory-bench convert --source bfcl-memory --bfcl-repo /path/to/gorilla
```

Output will have `ground_truth_type: cumulative` automatically.

### For Custom Datasets

Choose based on your evaluation goal:

**Use per_turn when:**
- Testing proactive judgment timing
- You care WHEN facts are stored
- You have noise turns that should trigger no storage

**Use cumulative when:**
- Testing storage completeness
- Timing doesn't matter
- You only care WHAT gets stored eventually

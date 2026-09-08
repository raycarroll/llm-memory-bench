# Dataset Types

LLM Memory Bench supports two types of ground truth, each testing different aspects of memory storage:

## Per-Turn Ground Truth

**Used by:** AlpsBench

**What it tests:** Proactive judgment timing - **WHEN** to store facts

**Key characteristic:** Ground truth is specified for each turn. The system must store the right facts at the right time.

### Structure

```yaml
ground_truth_type: per_turn  # Optional - this is the default
conversations:
  - id: conv_1
    turns:
      - role: user
        content: "I'm a data scientist working on LLM evaluation"
        ground_truth:
          should_store:
            - fact: "User is a data scientist"
              type: direct
            - fact: "User works on LLM evaluation"
              type: direct
      
      - role: user
        content: "The weather is nice today"
        ground_truth:
          should_store: []  # Noise turn - nothing to store
```

### Evaluation

- **Per-turn matching:** Facts stored during turn N are matched against turn N's expected facts
- **Timing matters:** Storing a fact too early or too late counts as wrong
- **Noise resistance:** Measures whether the system correctly avoids storing on noise turns

### Metrics

- `per_turn_extraction_recall` - Of facts that should be stored at specific turns, how many were?
- `per_turn_extraction_precision` - Of facts stored at each turn, how many were correct for that turn?
- `per_turn_extraction_f1` - Harmonic mean of precision and recall
- `per_turn_noise_resistance_rate` - How often did model correctly NOT store on noise turns?
- `schema_validity_rate` - Percentage of tool calls that match the schema

---

## Cumulative Ground Truth

**Used by:** BFCL Memory

**What it tests:** Storage completeness - **WHAT** gets stored (timing irrelevant)

**Key characteristic:** Ground truth is specified once per conversation. All expected facts can be stored at any point during the conversation.

### Structure

```yaml
ground_truth_type: cumulative
conversations:
  - id: conv_1
    cumulative_ground_truth:
      should_store:
        - fact: "User's dog is named Archie"
          type: direct
        - fact: "User is a CS student"
          type: direct
    turns:
      - role: user
        content: "I'm studying computer science"
        ground_truth:
          should_store: []  # Empty for cumulative datasets
      
      - role: user
        content: "My golden retriever Archie loves the park"
        ground_truth:
          should_store: []  # Empty for cumulative datasets
```

### Evaluation

- **Conversation-level matching:** ALL facts stored across the entire conversation are matched against the cumulative expected facts
- **Order-independent:** Doesn't matter when facts are stored, only that they are stored somewhere
- **No noise turns:** Noise resistance is not applicable (no per-turn expectations)

### Metrics

- `cumulative_extraction_recall` - Of all expected facts, how many were stored (regardless of when)?
- `cumulative_extraction_precision` - Of all facts stored, how many were expected?
- `cumulative_extraction_f1` - Harmonic mean of precision and recall
- `schema_validity_rate` - Percentage of tool calls that match the schema

**Note:** `noise_resistance_rate` is **not applicable** for cumulative datasets.

---

## Choosing a Dataset Type

### Use Per-Turn When:

- Testing proactive judgment timing
- You care **when** facts are stored
- You have noise turns that should trigger no storage
- You want to test the model's ability to distinguish signal from noise

**Example use case:** Testing whether an agent can identify and store only relevant information from mixed conversations

### Use Cumulative When:

- Testing storage completeness
- Timing doesn't matter (e.g., batch processing scenarios)
- You only care **what** gets stored eventually
- The conversation is structured such that facts can appear anywhere

**Example use case:** Testing whether an agent can extract all key information from a conversation log, regardless of when it stores each fact

---

## Validation Rules

Datasets are validated on load to ensure consistency:

### Per-Turn Datasets

- ✅ Each turn can have `ground_truth.should_store` with facts
- ❌ Must NOT have `cumulative_ground_truth` at conversation level

### Cumulative Datasets

- ✅ Must have `cumulative_ground_truth` at conversation level
- ❌ Per-turn `ground_truth.should_store` must be empty

**The types are mutually exclusive** - a dataset cannot mix per-turn and cumulative ground truth.

### Error Examples

```python
# ERROR: Cumulative type but no cumulative ground truth
Dataset(
    ground_truth_type="cumulative",
    conversations=[
        Conversation(id="test", turns=[...])  # Missing cumulative_ground_truth
    ]
)
# ValueError: Dataset type is 'cumulative' but conversation 'test' has no cumulative_ground_truth

# ERROR: Cumulative type but has per-turn ground truth
Dataset(
    ground_truth_type="cumulative",
    conversations=[
        Conversation(
            id="test",
            cumulative_ground_truth=GroundTruth(...),
            turns=[
                Turn(
                    content="...",
                    ground_truth=GroundTruth(should_store=[...])  # Not allowed!
                )
            ]
        )
    ]
)
# ValueError: Dataset type is 'cumulative' but conversation 'test' turn 0 has per-turn ground_truth
```

---

## Creating Datasets

### Per-Turn (Manual)

```yaml
ground_truth_type: per_turn  # Optional - default
conversations:
  - id: example_1
    turns:
      - role: user
        content: "Your conversation here"
        ground_truth:
          should_store:
            - fact: "Extract this fact"
              type: direct
```

### Cumulative (Manual)

```yaml
ground_truth_type: cumulative
conversations:
  - id: example_1
    cumulative_ground_truth:
      should_store:
        - fact: "Extract this fact somewhere in the conversation"
          type: direct
    turns:
      - role: user
        content: "Your conversation here"
        ground_truth:
          should_store: []  # Must be empty
```

### Using Converters

```bash
# AlpsBench → per-turn ground truth
llm-memory-bench convert --source alpsbench-memory --alpsbench-file task1.csv

# BFCL Memory → cumulative ground truth
llm-memory-bench convert --source bfcl-memory --bfcl-repo /path/to/gorilla
```

---

## Backward Compatibility

Existing datasets without `ground_truth_type` default to `per_turn` for backward compatibility. All AlpsBench datasets continue to work without modification.

# Experiment Grouping

Group benchmark runs for reproducibility and comparison.

## Config Example

```yaml
experiment:
  group_id: memoryhub-eager-baseline
  runs: 5  # Run this experiment 5 times
  metadata:
    memory_system: memoryhub
    rules_version: eager-2026-11
    description: MemoryHub with eager loading
```

**Single run** (default):
```yaml
experiment:
  group_id: memoryhub-test
  # runs: 1 is implicit
```

**Multiple runs** for measuring variance:
```yaml
experiment:
  group_id: memoryhub-baseline
  runs: 10  # Run 10 times with identical config
```

## File Naming

Results are saved with experiment group prefix and run number:
- **Without group**: `results/bench_20261104-103851_claude-code_remote-http-v1.json`
- **With group (single run)**: `results/bench_memoryhub-eager-baseline_20261104-103851_claude-code_remote-http-v1.json`
- **With group (multiple runs)**: 
  - `results/bench_memoryhub-eager-baseline_20261104-103851_run1_claude-code_remote-http-v1.json`
  - `results/bench_memoryhub-eager-baseline_20261104-104205_run2_claude-code_remote-http-v1.json`
  - `results/bench_memoryhub-eager-baseline_20261104-104520_run3_claude-code_remote-http-v1.json`

## Use Cases

### 1. Reproducibility (multiple runs)

Measure non-determinism by running the same experiment multiple times:

```yaml
experiment:
  group_id: memoryhub-baseline
  runs: 5  # Automatically run 5 times
  metadata:
    description: Baseline with 5 runs to measure variance
```

One `llm-memory-bench bench` command generates 5 result files with `run1`, `run2`, ..., `run5` suffixes.

### 2. A/B Comparison

Compare different configurations:

```yaml
# Config A: eager loading
experiment:
  group_id: memoryhub-eager
  metadata:
    loading_pattern: eager

# Config B: lazy loading  
experiment:
  group_id: memoryhub-lazy
  metadata:
    loading_pattern: lazy
```

### 3. Rules/System Comparison

Compare different memory systems or rule sets:

```yaml
# Experiment 1: default generated rules
experiment:
  group_id: memoryhub-default-rules
  metadata:
    rules: generated-eager-2026-11
  
# Experiment 2: BEN comprehensive rules
experiment:
  group_id: memoryhub-ben-rules
  metadata:
    rules: BEN-comprehensive-207lines
```

## Results Structure

Each result file includes the experiment metadata:

```json
{
  "environment": {
    "experiment": {
      "group_id": "memoryhub-eager-baseline",
      "metadata": {
        "memory_system": "memoryhub",
        "rules_version": "eager-2026-11",
        "description": "MemoryHub with eager loading",
        "run_number": "2"  // Added for multi-run experiments
      },
      "timestamp": "2026-11-04T10:00:00Z"
    },
    "host": {...},
    "integration": {...}
  },
  "conversations": [...]
}
```


## Future: Comparison CLI

Planned commands for aggregating and comparing experiments:

```bash
# List all experiment groups
llm-memory-bench experiments list

# Show stats for all runs in a group
llm-memory-bench experiments show memoryhub-eager-baseline

# Compare two experiment groups side-by-side
llm-memory-bench experiments compare memoryhub-eager memoryhub-lazy

# Aggregate statistics across multiple runs
llm-memory-bench experiments aggregate memoryhub-baseline-run*
```

## Analyzing Results

**Find all runs in an experiment group:**
```bash
ls results/bench_memoryhub-eager-baseline_*.json
```

**Calculate statistics across multiple runs:**
```bash
# Count stored facts per run
for f in results/bench_memoryhub-baseline_*_run*_*.json; do
  echo "$f: $(jq '.conversations[0].stored | length' $f)"
done

# Mean and std dev
jq -s 'map(.conversations[0].stored | length) | {mean: (add/length), values: .}' \
  results/bench_memoryhub-baseline_*_run*_*.json
```

**Compare two experiment groups:**
```bash
# Eager vs Lazy loading
echo "Eager:" && jq -s 'map(.conversations[0].stored | length) | add/length' results/bench_memoryhub-eager_*.json
echo "Lazy:" && jq -s 'map(.conversations[0].stored | length) | add/length' results/bench_memoryhub-lazy_*.json
```

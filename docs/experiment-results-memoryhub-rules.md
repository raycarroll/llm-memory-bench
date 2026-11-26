# Experiment Results: MemoryHub Rules Comparison

**Date:** November 4, 2026  
**Experiment Group:** `memoryhub-eager-baseline`  
**Runs per Experiment:** 3 (with full isolation)  
**Dataset:** BFCL Memory Student (first conversation)

## Objective

Compare the impact of MemoryHub rule comprehensiveness on fact extraction performance.

## Experiments

### Experiment 1: Default Rules
- **Time:** 11:22-11:30 AM
- **Rules File:** `memoryhub-loading.md` (59 lines, default)
- **Description:** Minimal MemoryHub eager loading pattern with basic memory hygiene
- **Key Characteristics:**
  - Brief "DO write" guidance (preferences, decisions, architectural choices)
  - No explicit "MUST save" instructions
  - No comprehensive fact categories

### Experiment 2: BEN Comprehensive Rules
- **Time:** 13:00-13:11 PM
- **Rules File:** `memoryhub-loading_BEN.md` (207 lines, comprehensive)
- **Description:** Detailed rules with explicit fact categories and mandatory capture
- **Key Characteristics:**
  - Explicit "MUST save when the user states or confirms" section
  - Comprehensive fact categories (preferences, background, constraints, goals, environment, decisions)
  - Mandatory capture pass before finishing
  - Detailed scope selection guidance

## Results

### Experiment 1: Default Rules

| Run | Facts Stored | Tool Calls | Behavior |
|-----|--------------|------------|----------|
| 1   | 1            | 0          | Only smoke test stored |
| 2   | 1            | 0          | Only smoke test stored |
| 3   | 1            | 0          | Only smoke test stored |

**Statistics:**
- Mean: **1.0 ± 0.0 facts**
- Range: 1-1 facts
- Coefficient of Variation: **0.0%** (perfectly consistent)
- **Outcome:** Complete extraction failure

**Analysis:**
- Claude ran all 10 conversation turns
- Responded conversationally (e.g., "Hey there! That's awesome that you're in your fourth year!")
- **Made zero tool calls** to store memories during actual conversation
- Only the smoke test memory was stored

### Experiment 2: BEN Comprehensive Rules

| Run | Facts Stored | Tool Calls | Behavior |
|-----|--------------|------------|----------|
| 1   | 11           | 12         | Extracted biographical facts |
| 2   | 34           | 44         | Highly aggressive extraction |
| 3   | 25           | 27         | Moderate extraction |

**Statistics:**
- Mean: **23.3 ± 11.6 facts**
- Range: 11-34 facts
- Coefficient of Variation: **49.7%** (high variance)
- **Outcome:** Successful extraction with significant non-determinism

**Sample Facts Extracted (Run 1):**
1. "Overall perspective on senior year: having a blast..."
2. "Biggest challenge this semester is balancing group projects..."
3. "Weekly class schedule: M/W jam-packed..."
4. "Taking Philosophy of Mind elective..."
5. "Capstone Project course - sink-or-swim part..."

**Analysis:**
- Claude actively called memory tools throughout conversation
- Stored user-stated biographical, schedule, and challenge information
- High variance (3x difference between runs) demonstrates LLM non-determinism

## Visualization

![MemoryHub Rules Comparison](images/memoryhub_experiment_comparison.png)

**Left panel:** Mean facts stored with standard deviation error bars  
**Right panel:** Individual run variance showing LLM non-determinism

## Key Findings

### 1. Rule Comprehensiveness Impact
**BEN rules extracted 23.3x more facts than default rules**

- Default rules: 1.0 facts (smoke test only)
- BEN rules: 23.3 facts (actual extraction)
- Increase: +2,233%

### 2. Default Rules Failure Mode
Default rules provided insufficient guidance for extraction:
- ✅ Claude had access to memory tools (smoke test worked)
- ✅ Claude responded to all 10 conversation turns
- ❌ Claude chose conversational responses over tool use
- ❌ Zero memories stored from user-stated facts

**Root cause:** Minimal "when to save" guidance in 59-line default rules

### 3. LLM Non-Determinism
**High variance even with identical configuration:**

- Experiment 1: CV = 0% (consistently failed)
- Experiment 2: CV = 49.7% (11-34 facts range)

**Implication:** Single-run benchmarks are unreliable. The same conversation with the same rules produced:
- Run 1: 11 facts
- Run 2: 34 facts (3.1x more)
- Run 3: 25 facts

### 4. Statistical Significance
**What "Mean: 23.3 ± 11.6 facts" means:**

- Average: 23.3 facts per run
- Standard deviation: ±11.6 facts
- **68% confidence interval:** 11.7 to 34.9 facts
- Running this again would likely produce 12-35 facts

This variance is purely from **LLM non-determinism** - same input, different extraction decisions.

## Conclusions

1. **Comprehensive rules are essential**
   - Default MemoryHub rules (59 lines) → near-zero extraction
   - BEN comprehensive rules (207 lines) → 23x improvement
   - Need explicit "MUST save" categories and examples

2. **Multiple runs are mandatory**
   - Single runs unreliable due to 50% CV
   - Recommend N≥5 runs for stable measurements
   - Report mean AND standard deviation

3. **Rule design matters more than tool availability**
   - Same MCP tools, same model, same dataset
   - Only difference: rule comprehensiveness
   - 2,233% performance difference

4. **Benchmark design implications**
   - Use BEN-style comprehensive rules for evaluation
   - Run multiple iterations to measure variance
   - Track both extraction rate AND consistency
   - Consider variance as a quality metric (lower is better)

## Recommendations

### For Benchmarking
- ✅ Use comprehensive rules (BEN-style) with explicit categories
- ✅ Run N≥5 iterations per configuration
- ✅ Report mean ± standard deviation
- ✅ Include variance (CV%) as a quality metric
- ⚠️ Single-run results are unreliable

### For Production Deployments
- Comprehensive rules increase extraction but also variance
- Consider trade-off: completeness vs. consistency
- May need additional validation/deduplication
- Scope selection (user vs project) needs clear guidance

## Files

### Result Files
```
results/bench_memoryhub-eager-baseline_20261104-112232_run1_claude-code_remote-http-v1.json
results/bench_memoryhub-eager-baseline_20261104-112642_run2_claude-code_remote-http-v1.json
results/bench_memoryhub-eager-baseline_20261104-113037_run3_claude-code_remote-http-v1.json
results/bench_memoryhub-eager-baseline_20261104-130003_run1_claude-code_remote-http-v1.json
results/bench_memoryhub-eager-baseline_20261104-130516_run2_claude-code_remote-http-v1.json
results/bench_memoryhub-eager-baseline_20261104-131135_run3_claude-code_remote-http-v1.json
```

### Configuration
- **Config:** `configs/claude-memoryhub-remote.yaml`
- **Dataset:** `datasets/converted/bfcl_memory_student.yaml`
- **Integration Profile:** `integrations/profiles/memoryhub/remote-http-v1/profile.yaml`
- **Default Rules:** `integrations/profiles/memoryhub/remote-http-v1/assets/.claude/rules/memoryhub-loading.md` (59 lines)
- **BEN Rules:** `integrations/profiles/memoryhub/remote-http-v1/assets/.claude/rules/memoryhub-loading_BEN.md` (207 lines)

## Reproducibility

To reproduce these experiments:

```bash
# Experiment 1: Default Rules (already active)
llm-memory-bench bench configs/claude-memoryhub-remote.yaml

# Experiment 2: BEN Rules
cp integrations/profiles/memoryhub/remote-http-v1/assets/.claude/rules/memoryhub-loading_BEN.md \
   integrations/profiles/memoryhub/remote-http-v1/assets/.claude/rules/memoryhub-loading.md
llm-memory-bench bench configs/claude-memoryhub-remote.yaml
```

Experiment configuration in `configs/claude-memoryhub-remote.yaml`:
```yaml
experiment:
  group_id: memoryhub-eager-baseline
  runs: 3
  metadata:
    memory_system: memoryhub
    rules_version: eager-2026-11
```

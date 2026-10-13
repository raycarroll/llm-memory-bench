# Value Benchmarking - Quick Start

**Goal:** Show quantitative evidence that memory improves task outcomes

**Status:** ✅ Scenario generation working | ⏳ Evaluation pipeline next

---

## What You Have Now

### Generated Scenarios (50 total)

```bash
scenarios/quick-demo/alpsbench-value-targeted.yaml
```

**Distribution:**
- 36 preference scenarios ("prefers X" → recommendations)
- 6 constraint scenarios (budget/time limits → suggestions)
- 4 location scenarios ("lives in X" → local recommendations)
- 3 dietary scenarios (vegetarian/allergies → food)
- 1 skill scenario (proficiency level → code/explanations)

### Example Scenario Structure

```yaml
id: sess_0124ac96a301_constraint_0
memory_bank:
  - fact: "Prefers maximum difficulty (10/10) in custom game scenarios"
    category: direct
task_prompt: "Recommend a laptop for work"
outcomes:
  - id: budget_constraint
    statement: "The recommendation stays within the user's budget constraint."
  - id: specific_models
    statement: "The response names specific models within the budget."
  - id: no_redundant_question
    statement: "The response does not ask for the user's budget again."
value_category: constraint
```

---

## Next Steps to Show Value

### 1. Run Paired Trials (baseline vs memory)

```bash
llm-memory-bench value-run \
  --scenarios scenarios/quick-demo/alpsbench-value-targeted.yaml \
  --system simple \
  --mode prompt \
  --max-turns 5 \
  --max-scenarios 10 \
  --provider anthropic \
  --model claude-sonnet-4-20250514
```

**This will:**
- Run each scenario twice: once WITHOUT memory (baseline), once WITH
- Record tokens used, turns taken, final responses
- Stop after 10 scenarios for quick testing

**Expected outputs:**
```
results/value_TIMESTAMP_anthropic_claude-sonnet-4-20250514_simple.json
```

### 2. Evaluate Quality

```bash
llm-memory-bench value-evaluate \
  results/value_*.json \
  --judge-model claude-sonnet-4-20250514
```

**This will:**
- Use LLM judge to score response quality against rubrics
- Compute metrics: quality_delta, token_efficiency, turn_compression
- Show which scenarios memory helped vs hurt

### 3. Generate Report

```bash
llm-memory-bench value-compare results/value_eval_*.json
```

**Shows:**
- Aggregate metrics across all scenarios
- Per-category breakdown (preference vs constraint vs location)
- Cost analysis (effective token savings after memory overhead)

---

## What Metrics You'll Get

### Quality
- **quality_delta**: How much better (+) or worse (-) with memory
- **scenarios_quality_improved**: Count where memory helped
- **scenarios_quality_degraded**: Count where memory hurt

### Efficiency
- **token_efficiency_ratio**: baseline_tokens / memory_tokens (>1.0 = savings)
- **turn_compression**: baseline_turns / memory_turns (>1.0 = fewer turns)
- **effective_token_savings**: Net savings after memory overhead

### ROI
- **value_per_token**: quality_delta / memory_overhead_tokens

---

## Expected Results (Hypothesis)

Based on scenario design:

| Category | Expected Quality Delta | Expected Token Savings | Why |
|----------|----------------------|----------------------|-----|
| **dietary** | +0.3 to +0.5 | 30-50% | Baseline must ask OR guess wrong |
| **location** | +0.2 to +0.4 | 20-40% | Baseline gives generic chains vs local spots |
| **preference** | +0.1 to +0.3 | 10-30% | Baseline hedges, memory enables specificity |
| **constraint** | +0.2 to +0.4 | 20-40% | Baseline asks about budget/time |
| **skill** | +0.3 to +0.5 | 20-40% | Baseline asks skill level OR over/under-explains |

**Overall expectation:** 60-80% of scenarios show positive quality_delta, 50-70% show net token savings

---

## Failure Modes to Watch

When quality_delta < 0 (memory hurts):

1. **Irrelevant memory applied** - "prefers synth percussion" → laptop recommendation
2. **Over-confident hallucination** - Agent invents details beyond memory
3. **Stale preference** - Memory contradicts current request
4. **Wrong retrieval** - Noise memory used instead of signal

Track these via the failure analysis in evaluator (TODO: implement)

---

## Commands Summary

```bash
# Generate scenarios (done)
llm-memory-bench value-generate-quick \
  --max-scenarios 50 \
  --output scenarios/quick-demo

# Run trials
llm-memory-bench value-run \
  --scenarios scenarios/quick-demo/alpsbench-value-targeted.yaml \
  --max-scenarios 10

# Evaluate (TODO: implement value-evaluate command)
# For now, outputs go to results/ for manual inspection

# Scale up
llm-memory-bench value-run \
  --scenarios scenarios/quick-demo/alpsbench-value-targeted.yaml \
  --max-scenarios 50  # All scenarios
```

---

## File Locations

```
llm_memory_bench/
├── scenarios/
│   └── quick-demo/
│       └── alpsbench-value-targeted.yaml  # ✅ Generated
│
├── results/
│   └── value_*.json                       # ⏳ Run trials to create
│
├── src/llm_memory_bench/value/
│   ├── generator_targeted.py              # ✅ Template-based generation
│   ├── runner.py                          # ✅ Paired trial execution
│   ├── evaluator.py                       # ⏳ LLM judge (exists, needs testing)
│   └── metrics.py                         # ✅ Value metric computation
│
└── docs/
    └── value-quick-start.md               # This file
```

---

## Research Questions Answered

✅ **"What scenarios demonstrate value?"**  
→ Preferences, constraints, location, dietary, skill level

✅ **"What data do we need?"**  
→ AlpsBench Task 2 memories + template tasks

✅ **"How to show value quantitatively?"**  
→ Quality delta, token efficiency, turn compression

⏳ **"Does memory actually help?"**  
→ Run trials to find out

---

## Next Immediate Actions

1. **Test the pipeline** on 10 scenarios:
   ```bash
   llm-memory-bench value-run \
     --scenarios scenarios/quick-demo/alpsbench-value-targeted.yaml \
     --max-scenarios 10 \
     --max-turns 3
   ```

2. **Inspect results** manually before full run

3. **Fix any issues** in runner.py or evaluator.py

4. **Scale to 50 scenarios** once validated

5. **Add value-evaluate command** if missing (check CLI)

6. **Generate first numbers** to share

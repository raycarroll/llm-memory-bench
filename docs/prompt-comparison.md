# Prompt comparison

Use the simulated-tool extraction benchmark to compare memory prompts while
holding the dataset, provider, and model constant. The `system` name selects a
prompt and tool contract from `src/llm_memory_bench/systems/`; it does not run a
live memory backend.

## Run the comparison

Convert a small dataset first if needed:

```bash
llm-memory-bench convert \
  --source alpsbench \
  --output datasets/converted/prompt-comparison \
  --max-conversations 10
```

Run the same model against several systems:

```bash
DATASET=datasets/converted/prompt-comparison/alpsbench-task1.yaml

for system in simple memoryhub behavioral_relevance; do
  llm-memory-bench run \
    --dataset "$DATASET" \
    --system "$system" \
    --provider openai \
    --model gpt-4o \
    --output "results/prompt-comparison/$system/run.json"
done
```

Evaluate each run with the same matcher:

```bash
for system in simple memoryhub behavioral_relevance; do
  llm-memory-bench evaluate \
    "results/prompt-comparison/$system/run.json" \
    --dataset "$DATASET" \
    --matcher embedding \
    --output "results/prompt-comparison/$system/eval_embedding.json"
done

llm-memory-bench compare \
  results/prompt-comparison/*/eval_embedding.json
```

Use `--matcher exact` for a strict comparison, `--matcher embedding` for a
local semantic comparison, or `--matcher hybrid`/`llm` when you have configured
a judge provider. The `embedding` matcher requires the optional `embedding`
dependency.

## Interpretation

- Keep provider, model, dataset, and limits fixed when comparing systems.
- Compare precision, recall, F1, noise resistance, and schema validity together;
  raw tool-call count is not a quality metric.
- Treat results as empirical. Prompt performance depends on the model, dataset,
  matcher, and memory-system schema; do not assume fixed recall percentages.
- Use `value-run` when the question is whether stored memories improve a later
  task. Extraction prompt comparisons do not measure downstream task value.

See [metrics.md](metrics.md) for metric definitions and
[baseline-comparison.md](baseline-comparison.md) for the BFCL comparison.

# Value benchmark quick start

The value benchmark runs paired baseline and memory arms against the same task
scenarios. It measures whether memory changes task quality, turns, and token
use. It is separate from extraction, which measures whether the agent stores
the right facts.

## Fastest local run

Use the checked-in goal scenario for a smoke test:

```bash
pip install -e .
export OPENAI_API_KEY="..."

llm-memory-bench value-run \
  --scenarios scenarios/quick-demo-goal.yaml \
  --system memoryhub \
  --provider openai \
  --model gpt-4o \
  --max-scenarios 1 \
  --output results/value-smoke.json
```

The command runs the paired experiment and evaluates outcomes during the run by
default. Inspect the result directly or generate an HTML trace:

```bash
llm-memory-bench value-visualize results/value-smoke.json
```

There is no separate `value-evaluate` or `value-compare` command. Use
`value-visualize` for one result and the JSON output for downstream analysis.

## Choose the memory setup

`--memory-setup-mode` controls the initial memory state:

- `none` — default; both arms start without declared memories.
- `seeded` — the memory arm receives the scenario's declared `memory_bank`;
  the baseline remains cold.
- `capture_first` — the baseline writes memories first, then the memory arm
  receives the captured snapshot.

For a declared-memory comparison:

```bash
llm-memory-bench value-run \
  --scenarios scenarios/quick-demo-goal.yaml \
  --system memoryhub \
  --provider openai \
  --model gpt-4o \
  --memory-setup-mode seeded \
  --max-scenarios 10 \
  --output results/value-seeded.json
```

The `--mode` option controls how the simulated adapter exposes memory:

- `prompt` — inject memory into the prompt.
- `tool` — expose a recall tool.
- `both` — use prompt injection and a recall tool.

## Generate scenarios

Generate LLM-authored scenarios from AlpsBench:

```bash
llm-memory-bench value-generate \
  --source alpsbench \
  --output scenarios/generated \
  --max-scenarios 50 \
  --provider openai \
  --model gpt-4o
```

For a fast template-based demo without an LLM generation call:

```bash
llm-memory-bench value-generate-quick \
  --output scenarios/quick-demo \
  --max-scenarios 10
```

Pass the generated YAML directory or file to `value-run` with
`--scenarios`.

## Real agent integration

To test a real Claude Code + memory integration in a container, use
`container-value-run` instead of `value-run`:

```bash
llm-memory-bench container-value-run \
  --config configs/claude-memoryhub-remote.yaml \
  --scenarios scenarios/test-value.yaml \
  --max-scenarios 1 \
  --user-sim-provider openai \
  --judge-provider openai \
  --output results/container-value-smoke.json
```

This requires Docker or Podman and the credentials referenced by the selected
container config. See [container-integration-profiles.md](container-integration-profiles.md)
for isolation and profile limitations.

## Other value sources

- **τ²-bench:** use `tau2-value-run` or `tau2-container-run`; see
  [tau2-value.md](tau2-value.md).
- **CL-Bench:** run `clbench-setup`, then `clbench-value-run`; see
  [clbench.md](clbench.md).

The Streamlit UI exposes the same value sources and provides task selectors and
result panels. See [ui-quickstart.md](ui-quickstart.md).

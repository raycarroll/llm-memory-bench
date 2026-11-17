# τ² Value Runs

`tau2-value-run` uses the real τ²-bench Python runner for task execution and
scoring. This repository owns only the paired experiment arms and telemetry.

τ² requires Python 3.12 or 3.13. Install it separately from the upstream
repository at the pinned revision used by this adapter:

```bash
pip install git+https://github.com/sierra-research/tau2-bench.git@c06b07dd7c4d4ec61ba6ff9bcdf0b15dd7ef36d4
```

Keep the upstream checkout available for its official data files and point
τ² at that checkout before running:

```bash
export TAU2_DATA_DIR=/path/to/tau2-bench/data
```

Create a memory plan keyed by official τ² task IDs:

```yaml
domain: airline
source_revision: c06b07dd7c4d4ec61ba6ff9bcdf0b15dd7ef36d4
tasks:
  - task_id: "3"
    memory_bank:
      - fact: "The user prefers exact numeric answers for baggage allowances."
        category: preference
```

Run a cold-start capture/replay experiment:

```bash
llm-memory-bench tau2-value-run \
  --domain airline \
  --task-ids 3 \
  --memory-plan scenarios/tau2-derived/airline-memory-plan.yaml \
  --memory-setup-mode capture_first \
  --agent-model openai/gpt-4.1 \
  --user-model openai/gpt-4.1 \
  --output results/tau2-value-airline.json
```

Each arm gets a fresh τ² environment and the same seed. In `capture_first` mode,
both arms start without preloaded memories. The baseline has the selected
memory system's tools available and captures writes made during its
conversation. The replay arm then starts with a snapshot of those captured
memories and can continue capturing new writes. The YAML memory bank is
retained as task provenance only; it is not injected into either arm.

Use `--memory-setup-mode seeded` when the memory plan contains deliberate
agent-visible prior memories. In that mode the baseline remains cold and the
memory arm receives the plan's `memory_bank`; there is no baseline handoff.
The official τ² task remains the hidden user-simulator context in both modes.

τ² produces the authoritative reward; the harness records the paired reward
delta, tool calls, LLM calls, latency, token usage, and the baseline-to-replay
memory handoff. When the provider returns LiteLLM usage, the result records
exact prompt/completion tokens; otherwise it falls back to a clearly labelled
character-count estimate (`characters / 4`).

The memory plan is deliberately separate from τ²'s task data. This makes the
memory intervention explicit and prevents the harness from rewriting the
official task, policy, database, or evaluator.

## Live MemoryHub

By default, τ² uses the deterministic harness memory adapter. To run against a
real MemoryHub instance, install the optional SDK and select live mode:

```bash
pip install -e '.[memoryhub]'
export MEMORYHUB_URL=https://memoryhub.example/mcp/
export MEMORYHUB_API_KEY=...

llm-memory-bench tau2-value-run \
  --domain airline \
  --task-ids 3 \
  --memory-plan scenarios/tau2-derived/airline-memory-plan.yaml \
  --memory-system memoryhub \
  --memoryhub-live
```

The runner creates a unique project scope for the baseline and another for
replay. Baseline writes go only to the baseline project; the captured snapshot
is written into the replay project before replay starts. Projects are deleted
after each task by default. Use `--no-memoryhub-cleanup` when inspecting the
provider state is necessary. Credentials are read from `MEMORYHUB_*` variables
and are never written to the result file.

## Container run

For the simplest execution path, build and run the self-contained τ² image:

```bash
llm-memory-bench tau2-container-run \
  --domain airline \
  --task-ids 3 \
  --memory-plan scenarios/tau2-derived/airline-memory-plan.yaml \
  --output results/tau2-value-airline.json
```

The command builds `llm-memory-bench-tau2:c06b07dd7c4d-tau2-live-memoryhub-v1` on first use. The image
contains Python 3.13, the harness, pinned τ², and the matching official τ²
data, including the optional MemoryHub SDK. Docker Desktop or a running Podman
machine is required. API credentials are passed through from the host
environment; they are not baked into the image:

```bash
export OPENAI_API_KEY=...
export MEMORYHUB_URL=https://memoryhub.example/mcp/
export MEMORYHUB_API_KEY=...
```

Add `--memory-system memoryhub --memoryhub-live` to the container command to
enable the live path. The MemoryHub URL must be reachable from inside the
container; for a host-local service, use the container-reachable host name
(for example `http://host.docker.internal:18080/mcp/` on Docker Desktop).

After the first build, use `--no-build` for a run-only invocation. Set
`LLM_MEMORY_BENCH_CONTAINER_RUNTIME=podman` to force Podman instead of Docker.

# Benchmark UI

The optional Streamlit UI defines runs using the same configuration model as
the application service layer. It presents extraction and value as the two
experiment families, with simulated-tool or real-system/containerised runtimes where supported.
τ²-bench is selectable as a value scenario source. UI-launched results are
written to `results/ui/`.

Install the UI dependency and start it from the repository root:

```bash
pip install -e '.[ui]'
llm-memory-bench-ui
```

If `llm-memory-bench-ui` was installed with `pipx`, install the optional UI
dependency into that same pipx environment:

```bash
pipx inject llm-memory-bench 'streamlit>=1.40.0'
llm-memory-bench-ui
```

The UI is intentionally a thin control layer. Benchmark execution, memory
systems, providers, and evaluators remain in the existing package so the CLI
and a future EvalHub adapter can use the same `RunSpec` and `RunService`.

Extraction runs follow the CLI artifact contract: **Run benchmark** writes the
run data only, while the **EVAL** pane evaluates a saved run and writes a
separate evaluation artifact under `results/ui/evaluations/`.

The extraction scenario panel lists YAML datasets found under `datasets/` and
also supports a custom dataset path.

The value benchmark dataset menu includes a curated τ²-derived starter set:

```text
scenarios/tau2-derived/airline-retail.yaml
```

It contains ten curated goal scenarios covering airline and retail task patterns.
It is τ²-derived rather than an official τ² dataset reproduction. The
same dataset can also be run without the UI by selecting it as the value
dataset path and using the existing value benchmark command.

## τ² Value experiments

Select `τ² Value` under `Benchmark`. The UI runs the real pinned τ² adapter in
the dedicated container image, then renders the paired baseline and memory
prompt arms. Use the τ² task-picker modal to choose tasks using readable labels
rather than raw IDs and inspect the selected task's purpose, reason for call,
user instructions, known information, and unknown information. Configure:

- τ² domain and tasks
- a memory plan, such as `scenarios/tau2-derived/airline-memory-plan.yaml`
- agent and user-simulator models
- memory system, seed, and maximum steps

Set the provider key in the environment before starting the UI. The first τ²
run builds the image if it is not present:

```bash
export OPENAI_API_KEY="your-openai-key"
llm-memory-bench-ui
```

The result view shows the memory facts, reward and cost telemetry for both
arms, the effective system prompts, reward details, and the chronological
native τ² conversation including tool calls. Each run writes timestamped
artifacts under `results/ui/`: the UI wrapper JSON, raw wrapper JSON, log, and
native τ² JSON.

## Claude Code + remote MemoryHub

Select `Claude container` under `Execution`, then choose a container config such
as `configs/claude-memoryhub-remote.yaml`. The UI runs a paired comparison:

- `Claude baseline`: a Claude Code container without the MemoryHub integration.
- `Claude + MemoryHub`: a separate Claude Code container with scenario memories
  seeded into the configured remote MemoryHub project.

Set credentials in the environment before starting the UI. Credential values
are not included in the run specification or result JSON:

```bash
export MEMORYHUB_URL="https://your-memoryhub-endpoint"
export MEMORYHUB_API_KEY="your-api-key"
export OPENAI_API_KEY="your-openai-key"
llm-memory-bench-ui
```

The runner prefers a reachable Podman engine and falls back to Docker only if
Podman is unavailable. If neither is running, start one before launching the
UI. For Podman:

```bash
podman machine start
```

To force a particular runtime, set:

```bash
export LLM_MEMORY_BENCH_CONTAINER_RUNTIME=podman  # or docker
```

The runner uses only commands supported by both CLIs (`run`, `exec`, `cp`,
`inspect`, and `rm`).

### Inspecting containers

The runner keeps the container's main process alive as a log-forwarding
process. Output from setup, agent execution, and queries is forwarded to the
container stdout stream, so it appears in Docker or Podman Desktop's log pane.
By default containers are removed after the run. To retain them for inspection,
set this in the selected container config:

```yaml
container:
  keep_alive: true
```

The container is then left running after the benchmark. Use the printed
container name with `podman exec -it <name> bash` or remove it manually after
inspection.

# Benchmark UI

The optional Streamlit UI defines runs using the same configuration model as
the application service layer. It supports local value and extraction runs,
plus Claude Code container value runs using an integration profile. UI-launched
results are written to `results/ui/`.

Install the UI dependency and start it from the repository root:

```bash
pip install -e '.[ui]'
llm-memory-bench-ui
```

The UI is intentionally a thin control layer. Benchmark execution, memory
systems, providers, and evaluators remain in the existing package so the CLI
and a future EvalHub adapter can use the same `RunSpec` and `RunService`.

The value benchmark dataset menu includes a curated τ²-derived starter set:

```text
scenarios/tau2/airline-retail.yaml
```

It contains ten goal scenarios covering airline and retail task patterns. The
same dataset can also be run without the UI by selecting it as the value
dataset path and using the existing value benchmark command.

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

# Benchmark UI

The optional Streamlit UI defines runs using the same configuration model as
the application service layer. It presents the current experiment families and
selects the runtime and configuration options that apply to each one.
UI-launched results are written to `results/ui/`.

| UI experiment | Runtime | Main setup |
|---|---|---|
| Extraction | Simulated-tool or real container | Dataset, provider, model, memory system |
| Value | Simulated-tool or real container | Scenario file, memory mode, evaluator |
| τ² Value | Pinned container | Official task IDs and memory plan |
| CL-Bench Value | External worker or container | `clbench-setup` checkout/runtime |
| OpenShift demo | Direct API | OpenShift namespaces, MCP endpoints, MemoryHub |

The UI is a control layer, not a separate benchmark implementation. It submits
the same `RunSpec` used by the application service and persists the resulting
artifacts under `results/ui/`.

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

## OpenShift operational demo

Select **OpenShift demo** under **Benchmark**. The demo runs paired memory and
baseline agents against the configured `scenarios/openshift/*.yaml` scenario.
The memory side can search MemoryHub; the baseline cannot. Both sides use
separate Kubernetes MCP endpoints and namespaces.

Before starting the run, deploy the resources described by:

- `deploy/kubernetes-mcp-sides.yaml` — side-specific MCP servers and RBAC.
- `deploy/openshift-demo.yaml` — verifier RBAC, UI Deployment, Service, and Route.
- `scenarios/openshift/network-policy.yaml` — the scenario contract and fixture.

The verifier uses the pod's ServiceAccount credentials in an in-cluster
deployment. For local execution, set `OPENSHIFT_CONTEXT` and `KUBECONFIG` to a
context that can access both demo namespaces. Set `MEMORYHUB_MCP_URL` and
`MEMORYHUB_API_KEY` for the memory side. The UI's **Extraction preparation run**
option performs a write-enabled preparation run, verifies recovery, then resets
the fixture before the paired comparison.

This is a live-cluster experiment; it is not a substitute for the simulated
extraction/value commands and should not be run against a shared production
namespace.

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

## CL-Bench Value experiments

Select **CL-Bench Value** under **Benchmark**. CL-Bench is optional and is not
installed by the normal package setup. Run the one-time setup from the
repository root first:

```bash
llm-memory-bench clbench-setup
```

The UI expects the checkout at `.cl-bench` by default and uses its `.venv/bin/python`
unless another interpreter is supplied. The first adapter slice supports the
`database_exploration` task. Choose the memory setup mode deliberately:
`none`, `seeded`, or `capture_first`. See [clbench.md](clbench.md) for the
native lifecycle and its current limitations.

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

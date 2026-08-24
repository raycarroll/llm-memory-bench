# llm-memory-bench

Benchmark LLM **proactive tool-calling judgment** — the ability to decide *when* to act without being explicitly asked.

Existing tool-calling benchmarks ([BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html), [TaskBench](https://openreview.net/pdf?id=ZUbraGNpAq)) test structural complexity: can the model call the right function with the right arguments? But they test **reactive** tool use — the user asks a question, the model picks a tool. The hard part of real-world tool use is often **proactive judgment**: recognizing that something worth acting on just happened, with no explicit trigger.

Memory extraction is a natural testbed for this. In a conversation, the user never says "save this to memory." They mention they're a data scientist, or that they prefer Python, or that there's a deploy freeze on Thursday. The model must recognize these as worth persisting, decide to call the memory tool unprompted, and do so at the right moment — all while ignoring the ~90% of turns that are noise.

This project uses conversations from [AlpsBench](https://huggingface.co/datasets/Cosineyx/Alpsbench) with human-verified ground-truth annotations, and tests LLMs against tool schemas and prompts inspired by real memory systems.

**Two benchmarks** (what you're measuring):

- **Extraction** — can the LLM identify implicit triggers in noisy conversation and call the memory tool proactively?
- **Value** — do stored memories actually improve downstream task performance?

**Two execution modes** (how you run them):

- **API-level** (`run`, `value-run`) — tests raw LLM capability for both benchmarks. Intercepts tool calls at the API layer. Fast iteration, provider-agnostic, no containers.
- **Containerised** (`bench`) — tests the full stack for the extraction benchmark. Runs conversations through real coding agents (e.g. Claude Code) with real memory systems (e.g. gbrain) inside Docker.

See [docs/architecture.md](docs/architecture.md) for diagrams and detailed design.

## Install

```bash
pip install -e .
```

Requires Python 3.11+. The containerised benchmark also requires Docker.

## Quick start

### API-level benchmark

```bash
# 1. Convert the AlpsBench dataset
llm-memory-bench convert --source alpsbench

# 2. Run the extraction benchmark
llm-memory-bench run \
  --dataset datasets/converted/alpsbench-task1.yaml \
  --system claude_code \
  --provider vertex \
  --model claude-sonnet-4@20250514

# 3. Compare runs
llm-memory-bench compare results/run-claude-code.json results/run-gbrain.json
```

### Containerised benchmark

```bash
# Run through real Claude Code + gbrain in Docker
llm-memory-bench bench configs/claude-gbrain.yaml
```

## Memory systems

Each memory system bundles a prompt, tool schemas, and extraction logic inspired by real memory systems. Different prompt designs frame the judgment differently — flat facts vs typed categories vs hierarchical pages — letting you test whether prompt design affects proactive tool-calling quality.

```bash
llm-memory-bench list-systems
```

| System | Description |
|--------|-------------|
| `simple` | Baseline single-tool system: `add_memory(fact, category)` |
| `claude_code` | Claude Code's auto-memory: typed memories (user/feedback/project/reference) with structured `save_memory` tool |
| `gbrain` | GBrain MCP knowledge-brain: `put_page` with slug-organized markdown pages + `capture` for quick one-liners |
| `memoryhub` | MemoryHub unified `memory(action=...)` dispatcher with scoped writes, weighted memories, and content type classification |

## Benchmarks

### Extraction benchmark

Tests **proactive tool-calling judgment**: can the LLM identify implicit triggers in noisy conversation and call the memory tool unprompted, at the right time, with the right content? Measures precision, recall, F1, noise resistance, and schema validity.

```bash
llm-memory-bench run \
  --dataset datasets/converted/alpsbench-task1.yaml \
  --system <simple|claude_code|gbrain> \
  --provider <anthropic|vertex|openai|litellm> \
  --model <model-id> \
  --output results/run.json
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | *(required)* | Path to converted dataset YAML |
| `--system` | `simple` | Memory system to benchmark |
| `--provider` | `anthropic` | LLM provider |
| `--model` | `claude-sonnet-4-20250514` | Model to evaluate |
| `--judge-provider` | same as `--provider` | Provider for the evaluation judge |
| `--judge-model` | same as `--model` | Model for the evaluation judge |
| `--max-conversations` | all | Limit conversations for quick tests |
| `--output` | `results/run.json` | Output path |
| `--api-key-env` | auto | Env var name for the API key |

**Metrics:**

- **extraction_precision** — of all tool calls made, how many matched expected facts
- **extraction_recall** — of all expected facts, how many were captured by tool calls
- **extraction_f1** — harmonic mean of precision and recall
- **noise_resistance_rate** — fraction of noise turns where no tool was called
- **schema_validity_rate** — fraction of tool calls with valid arguments per the schema

### Value benchmark

Tests whether stored memories actually improve task performance. Generates scenarios from conversations, then runs paired trials — one with memory and one without — to measure the delta.

```bash
# Generate scenarios (one-time, uses an LLM)
llm-memory-bench value-generate \
  --source alpsbench \
  --output scenarios \
  --max-scenarios 50

# Run paired trials
llm-memory-bench value-run \
  --scenarios scenarios \
  --system gbrain \
  --mode prompt \
  --max-turns 10 \
  --output results/value-run.json
```

**Options for `value-run`:**

| Flag | Default | Description |
|------|---------|-------------|
| `--scenarios` | *(required)* | Path to generated scenarios |
| `--system` | `simple` | Memory system to use |
| `--mode` | `prompt` | How memories are injected: `prompt` (in system prompt), `tool` (via recall tool), or `both` |
| `--max-turns` | `10` | Max conversation turns per trial |
| `--user-sim-provider` | same as `--provider` | Provider for the simulated user |
| `--user-sim-model` | same as `--model` | Model for the simulated user |
| `--max-scenarios` | all | Limit scenarios for quick tests |

### Containerised mode

Runs the same benchmarks through the full stack: a real coding agent (Claude Code) with a real memory system (gbrain, Claude Code auto-memory) installed inside Docker. Instead of intercepting tool calls at the API layer, the agent uses the memory system for real and we query what was stored afterward.

```bash
llm-memory-bench bench configs/claude-gbrain.yaml
```

This builds a Docker image with the host agent pinned to a specific version, installs the memory system, feeds conversations through the agent, then queries what was stored.

**Config format** (`configs/claude-gbrain.yaml`):

```yaml
host:
  name: claude-code
  version: "1.0.33"

system:
  name: gbrain
  version: "0.46.19.0"
  env:
    GBRAIN_SURFACE: starter

model: claude-sonnet-4@20250514

dataset: datasets/converted/task1.jsonl

max_conversations: 10
```

**Project layout for containerised benchmarks:**

```
hosts/
  claude-code/
    Dockerfile        # Base image: node + Claude Code CLI + Agent SDK
    entrypoint.sh     # Runs system installer on container start
    driver.py         # Feeds conversations via Claude Agent SDK

systems/
  gbrain/
    install.sh        # npm install + claude mcp add
    cleanup.sh        # Reset stored data between conversations
    query.sh          # Dump what was stored as JSON
  claude-code-memory/
    install.sh        # Configure auto-memory
    cleanup.sh
    query.sh

configs/
  claude-gbrain.yaml
  claude-memory.yaml
```

## Comparing runs

```bash
llm-memory-bench compare results/run1.json results/run2.json results/run3.json
```

Prints a side-by-side table of all metrics across runs, labeled by provider/model/system.

## Dataset

Uses [AlpsBench](https://huggingface.co/datasets/Cosineyx/Alpsbench) — conversations with human-verified memory items linked to source utterances.

- **Task 1 (Extraction):** Given a conversation, identify which facts to store
- **Task 2 (Updating):** Given a conversation and existing memories, identify new or updated facts

```bash
# Full dataset (English only by default)
llm-memory-bench convert --source alpsbench

# Small subset for testing
llm-memory-bench convert --source alpsbench --max-conversations 10

# Include all languages
llm-memory-bench convert --source alpsbench --all-languages
```

## Providers

| Provider | `--provider` value | Auth | Notes |
|----------|-------------------|------|-------|
| Anthropic | `anthropic` | `ANTHROPIC_API_KEY` | Claude models via direct API |
| Vertex AI | `vertex` | `ANTHROPIC_VERTEX_PROJECT_ID`, `GOOGLE_CLOUD_REGION` | Claude models via Vertex AI |
| OpenAI | `openai` | `OPENAI_API_KEY` | GPT models |
| LiteLLM | `litellm` | varies | Gemini, Mistral, Ollama, Bedrock, etc. |

## Adding a memory system

Create a new file in `src/llm_memory_bench/systems/` implementing the `MemorySystem` base class:

```python
from llm_memory_bench.systems.base import MemorySystem
from llm_memory_bench.providers.base import ToolCall

class MyMemorySystem(MemorySystem):
    name = "my_system"
    description = "Description shown in list-systems"

    def system_prompt(self) -> str:
        """Return the actual system prompt your memory system uses."""
        return "..."

    def tool_definitions(self) -> list[dict]:
        """Return the actual tool schemas your system exposes."""
        return [{"name": "...", "description": "...", "input_schema": {...}}]

    def extract_stored_fact(self, tool_call: ToolCall) -> str:
        """Extract the semantic content from a tool call for evaluation."""
        return tool_call.arguments.get("content", "")

    def format_tool_result(self, tool_call: ToolCall) -> dict:
        """Return what your system would respond with."""
        return {"status": "ok"}
```

Then register it in `src/llm_memory_bench/systems/__init__.py`.

## Example: comparing systems across models

```bash
# Claude via Vertex with each memory system
for sys in simple claude_code gbrain; do
  llm-memory-bench run \
    --dataset datasets/converted/alpsbench-task1.yaml \
    --system $sys \
    --provider vertex \
    --model claude-sonnet-4@20250514 \
    --output results/sonnet-${sys}.json
done

# GPT-4o with each memory system
for sys in simple claude_code gbrain; do
  llm-memory-bench run \
    --dataset datasets/converted/alpsbench-task1.yaml \
    --system $sys \
    --provider openai \
    --model gpt-4o \
    --output results/gpt4o-${sys}.json
done

# Compare everything
llm-memory-bench compare results/*.json
```

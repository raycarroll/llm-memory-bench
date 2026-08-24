# Architecture

## Overview

llm-memory-bench measures LLM **proactive tool-calling judgment** — the ability to decide *when* to act without being explicitly asked — using memory extraction as the task domain.

### Why memory extraction?

Existing tool-calling benchmarks test **reactive** tool use: the user asks a question, the model picks the right tool. [BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html) categorizes by call topology (simple → parallel → multi-turn). [TaskBench](https://openreview.net/pdf?id=ZUbraGNpAq) scales by graph structure (node → chain → DAG). These measure structural complexity — can the model navigate increasingly complex call patterns?

But in real-world agent deployments, the harder problem is often **proactive judgment**: recognizing that something worth acting on just happened, with no explicit trigger. ["To Call or Not to Call" (2026)](https://arxiv.org/html/2605.00737v1) separates the *judgment* of whether to call from the *mechanics* of how to call, finding that even top models misjudge when to invoke tools ~34% of the time. But their framework tests reactive decisions (user asked a question → should I search?).

Memory extraction tests a harder variant: the model must identify **implicit triggers** in noisy conversation. The user never says "save this." They mention they're a data scientist, or that deploys are frozen after Thursday. The model must recognize these as worth persisting — proactive judgment with a low signal-to-noise ratio (~90% of turns are noise).

```
  Reactive tool use              Proactive tool use
  (BFCL, TaskBench)              (llm-memory-bench)
  ─────────────────────────      ─────────────────────────
  "What's 2+2?"                  "I mostly code in Python"
       → calculator              → should I save this?
                                   (no one asked me to)

  Explicit trigger               Implicit trigger
  Obvious tool choice            Judgment about salience
  Measured by existing           Not measured by existing
  benchmarks                     benchmarks
```

### Benchmarks and execution modes

There are two orthogonal axes: **what** you're measuring (benchmark) and **how** you run it (execution mode).

**Benchmarks** (what):

| Benchmark | Question it answers |
|-----------|-------------------|
| **Extraction** | Can the LLM identify implicit triggers and call the memory tool proactively, with the right content? |
| **Value** | Do stored memories actually improve downstream task performance (quality, tokens, turns)? |

**Execution modes** (how):

| Mode | Applies to | How it works | Trade-off |
|------|-----------|-------------|-----------|
| **API-level** (`run`, `value-run`) | Extraction, Value | Intercepts tool calls at the provider API layer | Fast, provider-agnostic, tests raw LLM capability in isolation |
| **Containerised** (`bench`) | Extraction only | Runs a real coding agent + real memory system inside Docker | Slow, tests the full stack (agent + system + prompt interactions) |

The API-level mode is useful for comparing models and prompt designs quickly. The containerised mode answers "does it actually work end-to-end" — the agent uses the memory system for real, with its own prompt stack, and we query what was stored afterward.

## How this differs from AlpsBench

This project uses [AlpsBench](https://huggingface.co/datasets/Cosineyx/Alpsbench) as its data source — the annotated conversations and ground-truth memory items are excellent. But the benchmark protocol is fundamentally different.

**AlpsBench** treats memory as a **standalone NLP task**. It gives the LLM a conversation and says "output structured JSON memory items." The model directly produces `{"memory_id": "m1", "type": "direct", "value": "..."}` objects. It evaluates the model's raw capability to extract, update, retrieve, and utilize memories across four tasks.

**llm-memory-bench** tests LLMs **as they actually work inside deployed memory systems**. The differences:

```
  AlpsBench                           llm-memory-bench
  ─────────────────────────────────   ─────────────────────────────────
  "Here's a conversation.             "Here's a conversation, turn by
   Output memory items as JSON."       turn. You have a save_memory tool.
                                       Use it when appropriate."

  Uniform evaluation prompt            Real system prompts (Claude Code,
                                       GBrain, simple) — the prompt IS
                                       the variable

  Model outputs structured JSON        Model makes tool calls mid-
  after seeing full conversation       conversation; calls are
                                       intercepted and recorded

  Evaluates: can this model            Evaluates: does this model
  extract memories?                    correctly use THIS system's tools?

  Single-query utilization             Multi-turn paired trials
  (Task 4)                            (baseline vs memory-augmented)
                                       measuring token/turn/quality delta
```

| Dimension | AlpsBench | llm-memory-bench |
|-----------|-----------|------------------|
| **Evaluation format** | Direct JSON output | Tool call interception |
| **Prompt** | Uniform benchmark prompt | Real system prompts |
| **Tool schemas** | N/A (structured output) | Real MCP tool definitions per system |
| **Timing** | Post-hoc (full conversation given) | Turn-by-turn (must decide in real time) |
| **Noise resistance** | Not measured | First-class metric |
| **System comparison** | N/A | Compare Claude Code vs GBrain vs simple |
| **Value measurement** | Single query (Task 4) | Multi-turn agent-user conversations |
| **What it answers** | "Can this LLM extract memories?" | "Can this LLM proactively judge when to act, unprompted, in noisy conversation?" |

## System diagram

```
                                llm-memory-bench
  ┌───────────────────────────────────────────────────────────────────────┐
  │                                                                       │
  │                         ┌──────────────────────┐                      │
  │                         │        CLI           │                      │
  │                         └───┬──────────────┬───┘                      │
  │                  run/value-run              bench                      │
  │                             │              │                          │
  │              ┌──────────────▼──┐    ┌──────▼──────────────────┐       │
  │              │  API-level path │    │  Containerised path     │       │
  │              │                 │    │                         │       │
  │              │  ┌───────────┐  │    │  ┌───────────────────┐  │       │
  │              │  │  Runner   │  │    │  │ Docker container  │  │       │
  │              │  │           │  │    │  │                   │  │       │
  │              │  │ intercept │  │    │  │ Claude Code CLI   │  │       │
  │              │  │ tool calls│  │    │  │ + memory system   │  │       │
  │              │  └─────┬─────┘  │    │  │ + Agent SDK driver│  │       │
  │              │        │        │    │  └────────┬──────────┘  │       │
  │              │  ┌─────▼─────┐  │    │           │             │       │
  │              │  │ Provider  │  │    │    query.sh → stored    │       │
  │              │  │ API call  │  │    │                         │       │
  │              │  └───────────┘  │    └─────────────────────────┘       │
  │              │  anthropic      │                                      │
  │              │  vertex         │                                      │
  │              │  openai         │                                      │
  │              │  litellm        │                                      │
  │              └─────────────────┘                                      │
  │                                                                       │
  │  ┌──────────────────────┐     ┌──────────────────────┐               │
  │  │   Memory Systems     │     │   Evaluator          │               │
  │  │  simple              │     │   LLM-as-judge       │               │
  │  │  claude_code         │     │   (semantic matching) │               │
  │  │  gbrain              │     └──────────────────────┘               │
  │  └──────────────────────┘                                             │
  └───────────────────────────────────────────────────────────────────────┘
```

---

## Shared foundations

These components are used across benchmark modes.

### Memory systems

Each memory system is a self-contained unit that bundles everything the LLM needs to use it. This is the central design decision — rather than testing with synthetic prompts, each system faithfully reproduces a real memory system's interface.

```
  MemorySystem (ABC)
  ├── system_prompt()           # The actual instructions given to the LLM
  ├── tool_definitions()        # The actual tool schemas (MCP format)
  ├── extract_stored_fact()     # How to read what was stored from a tool call
  ├── format_tool_result()      # Simulated success response
  ├── inject_memories()         # Prepend known facts to prompt (value benchmark)
  ├── recall_tool_definitions() # Define recall_memory tool (value benchmark)
  └── format_recall_result()    # Format recall response (value benchmark)
```

| System | Tools | Prompt style | Extraction |
|--------|-------|-------------|------------|
| `simple` | `add_memory(fact, category)` | What to save / not save | `args["fact"]` |
| `claude_code` | `save_memory(name, description, type, body)` | 4 memory types with structured guidance | `args["body"]` |
| `gbrain` | `put_page(slug, body, tags)` + `capture(text)` | Hierarchical slugs, markdown pages | `args["body"]` or `args["text"]` |
| `memoryhub` | `memory(action="write", content=...)` | Unified dispatcher with scoped writes, weighted memories, content types | `args["content"]` when `action=="write"` |

### What gets passed to the LLM (API mode)

Each API call to the LLM receives three things from the memory system:

1. **System prompt** — the memory system's full instructions (e.g. Claude Code's ~60-line prompt explaining four memory types, when to save, what not to save)
2. **Tool definitions** — complete tool schemas with names, descriptions, all parameters (types, enums, required fields)
3. **Conversation messages** — the accumulated turns so far, including prior tool call/result pairs fed back as simulated successes

```
  provider.generate(
    system   = memory_system.system_prompt(),     # real system instructions
    tools    = memory_system.tool_definitions(),  # real tool schemas
    messages = [                                  # conversation so far
      {"role": "user", "content": "I'm a data scientist..."},
      {"role": "assistant", "content": "..."},       # or tool_use block
      {"role": "user", "content": [tool_result]},    # simulated success
      {"role": "user", "content": "I mostly use Python..."},
      ...
    ]
  )
```

The LLM sees the full context a real memory system would provide — the prompt IS the variable being tested.

### LLM providers

Abstracts the differences between LLM APIs behind a two-method interface:

```
  LLMProvider (ABC)
  ├── generate(messages, tools, system) -> ProviderResponse
  └── build_tool_result_messages(response, format_tool_result) -> list[dict]
```

`build_tool_result_messages` exists because Anthropic and OpenAI have incompatible tool result formats:

```
  Anthropic:                          OpenAI:
  ┌──────────────────────┐            ┌──────────────────────┐
  │ role: assistant      │            │ role: assistant      │
  │ content:             │            │ tool_calls:          │
  │   - type: tool_use   │            │   - id: call_abc     │
  │     id: toolu_abc    │            │     function:        │
  │     name: add_memory │            │       name: add_mem  │
  │     input: {...}     │            │       arguments: ... │
  ├──────────────────────┤            ├──────────────────────┤
  │ role: user           │            │ role: tool           │
  │ content:             │            │ tool_call_id: abc    │
  │   - type: tool_result│            │ content: {...}       │
  │     tool_use_id: abc │            └──────────────────────┘
  │     content: {...}   │
  └──────────────────────┘
```

### Dataset: AlpsBench

The benchmark uses [AlpsBench](https://huggingface.co/datasets/Cosineyx/Alpsbench), which provides real conversations with human-verified memory annotations.

**Task 1 (Extraction):** Given a full conversation, identify which facts to store. Each memory item is linked to a source utterance via `evidence.utterance_index`.

**Task 2 (Updating):** Given a conversation and a pre-existing memory bank, identify new or updated facts. The adapter compares input memory IDs vs reference memory IDs to find what changed.

```
  AlpsBench HuggingFace repo
  dataset/
  ├── dev/
  │   ├── task1/
  │   │   ├── model_input.jsonl      # Conversations
  │   │   └── reference_output.jsonl # Gold memory items
  │   └── task2/
  │       ├── model_input.jsonl      # Conversations + existing memories
  │       └── reference_output.jsonl # Updated memory set
  └── validation/
      └── (same structure)

  Input schema (Task 1):
  {
    "benchmark_id": "...",
    "session_id": "...",
    "input": {
      "sessions": [{
        "turns": [
          {"utterance_index": 0, "role": "user", "text": "..."},
          {"utterance_index": 1, "role": "assistant", "text": "..."}
        ]
      }]
    }
  }

  Reference schema (Task 1):
  {
    "gold": {
      "memory_items": [{
        "memory_id": "m1",
        "type": "direct",
        "value": "User prefers Python",
        "evidence": {"utterance_index": 4, "text": "I mostly code in Python"}
      }]
    }
  }
```

The adapter converts this into the benchmark's internal format:

```yaml
conversations:
  - id: sess_001
    source: alpsbench-task1
    turns:
      - role: user
        content: "I mostly code in Python and R."
        ground_truth:
          should_store:
            - fact: "User is a data scientist working in Python and R"
              type: direct
              source_id: m1
      - role: assistant
        content: "That's great! How can I help?"
        ground_truth:
          should_store: []    # noise turn — no tool call expected
```

---

## Extraction benchmark

Tests **proactive tool-calling judgment**: can the LLM identify implicit triggers in a noisy conversation and decide to call the memory tool unprompted, at the right time, with the right content?

This is distinct from reactive tool-calling benchmarks in three ways:

1. **No explicit trigger** — the user never asks the model to save anything. The model must recognize salience on its own.
2. **High noise ratio** — ~90% of turns contain nothing worth storing. The model must resist calling the tool on noise.
3. **Prompt-dependent judgment** — different memory system prompts frame what's "worth saving" differently (flat facts vs typed categories vs hierarchical pages). The prompt shapes the decision threshold.

The benchmark varies two axes: **model** (which LLM) and **system prompt** (what instructions frame the judgment). If results cluster by model, the bottleneck is raw LLM capability. If results vary by prompt within the same model, prompt design matters for proactive judgment.

### Core concept: MCP tool interception

Memory systems work by giving the LLM a tool (like `add_memory` or `put_page`) and instructions for when to use it. We measure how well the LLM follows those instructions by:

1. Presenting the LLM with a conversation containing known facts
2. Giving it the real memory system's prompt and tool schemas
3. Intercepting tool calls instead of executing them
4. Comparing intercepted calls against ground-truth annotations

```
  Conversation       Memory System         LLM Provider
  ┌──────────┐      ┌─────────────┐      ┌─────────────┐
  │ Turn 1   │      │ Prompt +    │      │             │
  │ Turn 2   │─────▶│ Tool Defs   │─────▶│  Generate   │
  │ Turn 3   │      └─────────────┘      │             │
  │ ...      │                           └──────┬──────┘
  └──────────┘                                  │
                                                │ tool_calls
                                                ▼
                                         ┌─────────────┐
                                         │ Intercepted  │
                                         │ & Recorded   │──▶ TurnResult
                                         │              │
                                         │ Simulated    │
                                         │ Success ◀────│──── format_tool_result()
                                         └─────────────┘
                                                │
                                                │ fed back as tool result
                                                ▼
                                         ┌─────────────┐
                                         │ Next Turn    │
                                         └─────────────┘
```

### Data flow

```
  AlpsBench (HuggingFace)
        │
        │ llm-memory-bench convert
        ▼
  ┌──────────────────┐
  │ YAML Dataset     │
  │                  │
  │ conversations:   │
  │  - id: sess_001  │
  │    turns:        │
  │    - role: user  │
  │      content: .. │
  │      ground_truth│
  │        should_store:        │
  │        - fact: "..."        │
  │          type: direct       │
  └──────────────────┘
        │
        │ llm-memory-bench run --system claude_code
        ▼
  ┌──────────────────┐     ┌─────────────────┐
  │ Runner           │────▶│ LLM Provider    │
  │                  │     │ (API calls)     │
  │ For each turn:   │◀────│                 │
  │  send messages   │     └─────────────────┘
  │  intercept tools │
  │  record results  │
  └──────────────────┘
        │
        │ RunResult (tool calls per turn)
        ▼
  ┌──────────────────┐     ┌─────────────────┐
  │ Evaluator        │────▶│ Judge LLM       │
  │                  │     │ (semantic match) │
  │ For each fact:   │◀────│                 │
  │  find best match │     └─────────────────┘
  │  among tool calls│
  │  verdict: MATCH  │
  │  / NO_MATCH      │
  └──────────────────┘
        │
        ▼
  ┌──────────────────┐
  │ results/run.json │
  │                  │
  │ precision: 0.82  │
  │ recall:    0.71  │
  │ f1:        0.76  │
  │ noise_resistance │
  │ schema_validity  │
  └──────────────────┘
```

### Evaluation: LLM-as-judge

Fact matching uses semantic comparison rather than string matching. The judge LLM receives each expected fact and each stored memory, and returns `MATCH`, `PARTIAL`, or `NO_MATCH`.

```
  Expected facts          Tool calls made
  from ground truth       by the LLM
  ┌────────────────┐      ┌────────────────┐
  │ "Prefers Python│      │ add_memory(    │
  │  over Java"    │─────▶│  "User likes   │──▶ Judge LLM ──▶ MATCH
  │                │      │   Python")     │
  ├────────────────┤      ├────────────────┤
  │ "Lives in      │      │                │
  │  Berlin"       │─────▶│ (no match)     │──▶ NO_MATCH (false negative)
  │                │      │                │
  │                │      ├────────────────┤
  │                │      │ add_memory(    │
  │                │      │  "Says hello") │──▶ false positive (noise turn)
  └────────────────┘      └────────────────┘
```

### Metrics

```
  precision         = TP / (TP + FP)
  recall            = TP / (TP + FN)
  f1                = harmonic mean
  noise_resistance  = 1 - (noise_violations / noise_turns)
  schema_validity   = valid_calls / total_calls
```

### Key modules

- `runner.py` — replays conversations turn-by-turn, intercepts tool calls, feeds back simulated results
- `evaluator.py` — runs LLM-as-judge semantic matching between intercepted calls and ground truth
- `dataset.py` — data model (Conversation, Turn, GroundTruth)
- `adapters/alpsbench.py` — converts AlpsBench Task 1 to the internal format

---

## Value benchmark

Tests whether stored memories actually improve task performance. This is a fundamentally different question from extraction: it doesn't care *how* things get stored — it measures whether having memories makes the LLM *better* at a downstream task.

### Concept: paired trials

For each scenario, we run the same task twice — once with no prior context (baseline) and once with memories injected — then compare quality, token usage, and turns needed.

Memories can be injected in three modes:
- **`prompt`** — prepended to the system prompt
- **`tool`** — available via a `recall_memory` tool the LLM can call
- **`both`** — combined

### Data flow

```
  AlpsBench Task 2 (HuggingFace)
        │
        │ llm-memory-bench value-generate
        │ (LLM synthesizes task + rubric from memory list)
        ▼
  ┌────────────────────┐
  │ Scenario YAML      │
  │                    │
  │ scenarios:         │
  │ - memory_bank:     │
  │   - fact: "..."    │
  │   task_prompt: ... │
  │   persona_prompt:  │
  │   quality_rubric:  │
  └────────────────────┘
        │
        │ llm-memory-bench value-run --mode prompt
        ▼
  ┌────────────────────────────────────────────────┐
  │  For each scenario, run paired trials:         │
  │                                                │
  │  BASELINE (no memory)     MEMORY (augmented)   │
  │  ┌──────────────────┐    ┌──────────────────┐  │
  │  │ Agent ◀──▶ User  │    │ Agent ◀──▶ User  │  │
  │  │  Sim             │    │  Sim             │  │
  │  │                  │    │                  │  │
  │  │ No prior context │    │ Memories in      │  │
  │  │ Must ask to learn│    │ prompt or via    │  │
  │  │                  │    │ recall tool      │  │
  │  └──────────────────┘    └──────────────────┘  │
  │         │                       │              │
  │         ▼                       ▼              │
  │  final_response          final_response        │
  │  tokens_used             tokens_used           │
  │  turns_taken             turns_taken           │
  └────────────────────────────────────────────────┘
        │
        │ LLM judge scores both responses against rubric
        ▼
  ┌──────────────────────┐
  │ results/value.json   │
  │                      │
  │ quality_delta: +0.15 │
  │ turn_compression: 2x │
  │ token_efficiency: 3x │
  │ value_per_token: ... │
  └──────────────────────┘
```

### Metrics

- **quality_delta** — how much better is the memory-augmented response (judged against rubric)
- **turn_compression** — ratio of baseline turns to memory-augmented turns
- **token_efficiency** — ratio of baseline tokens to memory-augmented tokens
- **value_per_token** — quality improvement per token of memory overhead

### Key modules

- `value/runner.py` — runs paired trials (baseline vs memory-augmented)
- `value/evaluator.py` — LLM-as-judge quality scoring against the rubric
- `value/metrics.py` — token efficiency, turn compression, quality delta
- `value/dataset.py` — ValueScenario, ValueDataset models
- `adapters/alpsbench_value.py` — converts AlpsBench Task 2 into scenarios

---

## Containerised execution mode

The API-level mode intercepts tool calls at the provider layer — fast iteration, but the LLM never actually uses the memory system. The containerised mode tests the full stack: a real coding agent with a real memory system installed, running inside Docker. There is no tool interception — the agent uses the memory system for real, and we query what was actually stored afterward.

This runs the same extraction benchmark (comparing stored items against ground truth) but through the real agent's complete prompt stack — system prompt, CLAUDE.md, MCP tools, plugins — rather than an isolated API call with a synthetic prompt.

### Data flow

```
  Config YAML              bench.py orchestrator
  ┌──────────────────┐     ┌──────────────────────────────────────────────┐
  │ host:             │     │                                              │
  │   name: claude-   │     │  1. Build host image (Dockerfile)            │
  │     code          │     │  2. Start container                          │
  │   version: 1.0.33 │────▶│  3. Mount system scripts + driver            │
  │ system:           │     │  4. Run install.sh                           │
  │   name: gbrain    │     │  5. For each conversation:                   │
  │ model: ...        │     │     a. Feed turns via Agent SDK driver       │
  │ dataset: ...      │     │     b. query.sh → dump stored items          │
  └──────────────────┘     │     c. cleanup.sh → reset between convos     │
                           │  6. Stop container, output results            │
                           └──────────────────────────────────────────────┘
```

### Directory structure

```
hosts/
  claude-code/
    Dockerfile          # node:22 + Claude Code CLI + Agent SDK
    entrypoint.sh       # Runs system installer on container start
    driver.py           # Feeds conversations via claude-agent-sdk query()

systems/                # Top-level — containerised system definitions
  gbrain/               # (distinct from src/.../systems/ which are API-level)
    install.sh          # npm install + claude mcp add
    cleanup.sh          # Reset stored data between conversations
    query.sh            # Dump what was stored as JSON
  claude-code-memory/
    install.sh          # Configure auto-memory
    cleanup.sh
    query.sh

configs/
  claude-gbrain.yaml    # Claude Code + gbrain
  claude-memory.yaml    # Claude Code + its own auto-memory
```

### Driver flow

The driver uses the Claude Agent SDK to feed user turns sequentially, resuming the same session so the agent accumulates context with its full prompt stack (system prompt, CLAUDE.md, MCP tools).

```
  driver.py
  ┌───────────────────────────────────────────────┐
  │  for each user turn in /tmp/turns.json:       │
  │                                               │
  │    query(                                     │
  │      prompt = turn["content"],                │
  │      options = ClaudeAgentOptions(             │
  │        resume = session_id,  ← same session   │
  │        model = "claude-sonnet-4@20250514",    │
  │        permission_mode = "auto",              │
  │      )                                        │
  │    )                                          │
  │                                               │
  │    The agent decides whether to call          │
  │    memory tools based on its own logic        │
  └───────────────────────────────────────────────┘
          │
          │ Agent makes real MCP tool calls
          ▼
  ┌───────────────────────────────────────────────┐
  │  Memory system (e.g. gbrain MCP server)       │
  │  actually stores data → can be queried later  │
  └───────────────────────────────────────────────┘
          │
          │ query.sh
          ▼
  ┌───────────────────────────────────────────────┐
  │  {"stored": [{"fact": "...", ...}, ...]}      │
  └───────────────────────────────────────────────┘
```

### Key modules

- `bench.py` — orchestrator: builds image, starts container, drives conversations, collects results
- `hosts/claude-code/driver.py` — feeds turns via the Claude Agent SDK `query()` function
- `systems/*/install.sh` — installs the memory system inside the container
- `systems/*/query.sh` — dumps stored memories as JSON
- `systems/*/cleanup.sh` — resets state between conversations

---

## Module structure

```
src/llm_memory_bench/
├── cli.py                  # Click CLI — all user-facing commands
├── config.py               # RunConfig, ValueRunConfig, provider factory
├── bench.py                # Containerised benchmark orchestrator (Docker + Agent SDK)
│
├── dataset.py              # Extraction benchmark data model (Conversation, Turn, GroundTruth)
├── runner.py               # Extraction benchmark runner (conversation replay + tool interception)
├── evaluator.py            # Extraction benchmark evaluator (LLM-as-judge fact matching)
│
├── systems/                # Memory system definitions (API-level)
│   ├── base.py             # MemorySystem ABC
│   ├── simple.py           # Baseline: add_memory(fact, category)
│   ├── claude_code.py      # Claude Code: save_memory(name, description, type, body)
│   ├── gbrain.py           # GBrain: put_page(slug, body, tags) + capture(text)
│   └── memoryhub.py        # MemoryHub: memory(action="write", content=...) unified dispatcher
│
├── providers/              # LLM API adapters
│   ├── base.py             # LLMProvider ABC, ToolCall, ProviderResponse
│   ├── anthropic.py        # Anthropic direct API
│   ├── vertex.py           # Claude on Vertex AI
│   ├── openai.py           # OpenAI API
│   └── litellm.py          # LiteLLM catch-all (Gemini, Mistral, Ollama, etc.)
│
├── adapters/               # External dataset converters
│   ├── alpsbench.py        # AlpsBench → extraction benchmark format
│   └── alpsbench_value.py  # AlpsBench → value benchmark scenarios
│
└── value/                  # Value benchmark
    ├── dataset.py          # ValueScenario, ValueDataset models
    ├── runner.py           # Paired trial runner (baseline vs memory-augmented)
    ├── evaluator.py        # LLM-as-judge quality scoring
    └── metrics.py          # Token efficiency, turn compression, quality delta
```

## Extension points

### Adding a memory system

1. Create `src/llm_memory_bench/systems/my_system.py`:

```python
from ..providers.base import ToolCall
from .base import MemorySystem

class MySystem(MemorySystem):
    name = "my_system"
    description = "One-line description"

    def system_prompt(self) -> str:
        # Paste the ACTUAL prompt your system uses
        return "..."

    def tool_definitions(self) -> list[dict]:
        # Return the ACTUAL tool schemas
        return [{
            "name": "store",
            "description": "...",
            "input_schema": { ... }
        }]

    def extract_stored_fact(self, tool_call: ToolCall) -> str:
        return tool_call.arguments.get("content", "")
```

2. Register in `systems/__init__.py`:

```python
from .my_system import MySystem
SYSTEMS["my_system"] = MySystem
```

### Adding a provider

1. Create `src/llm_memory_bench/providers/my_provider.py` implementing `LLMProvider`
2. Register in `config.py`'s `get_provider()` dict

## Configuration

### Extraction benchmark (`RunConfig`)

```
  RunConfig
  ├── provider: str           # "anthropic" | "openai" | "vertex" | "litellm"
  ├── model: str              # Model identifier
  ├── system: str             # Memory system name
  ├── judge_provider: str     # Provider for evaluation judge
  ├── judge_model: str        # Model for evaluation judge
  ├── max_conversations: int  # Limit for quick tests
  └── api_key_env: str        # Custom env var for API key
```

### Value benchmark (`ValueRunConfig`)

```
  ValueRunConfig (extends RunConfig)
  ├── injection_mode: str             # "prompt" | "tool" | "both"
  ├── max_turns: int                  # Max conversation turns per trial
  ├── user_simulator_provider: str    # Provider for user simulator
  ├── user_simulator_model: str       # Model for user simulator
  └── max_scenarios: int              # Limit scenarios
```

### Containerised benchmark (`bench.RunConfig`)

```
  bench.RunConfig (loaded from YAML)
  ├── host_name: str          # "claude-code"
  ├── host_version: str       # Pinned version (e.g. "1.0.33")
  ├── system_name: str        # Top-level system dir (e.g. "gbrain")
  ├── system_version: str     # Optional version to install
  ├── system_env: dict        # Env vars passed to the container
  ├── model: str              # Model identifier
  ├── dataset_path: str       # Path to converted dataset
  └── max_conversations: int  # Limit for quick tests
```

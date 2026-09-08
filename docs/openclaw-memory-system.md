# OpenClaw Memory System

API-level adapter for [OpenClaw](https://docs.openclaw.ai/concepts/memory-architecture) memory.

OpenClaw does not expose a dedicated `save_memory` tool. Durable state is plain Markdown in the agent workspace. This adapter reproduces that write surface so extraction scores are comparable with `simple`, `memoryhub`, `gbrain`, and the others.

## What OpenClaw actually does

Tiers (from the upstream architecture docs):

| Tier | Surface | Written by | Injected |
|------|---------|------------|----------|
| Instructions | `AGENTS.md` | Human only | Always |
| Curated core | `MEMORY.md`, `USER.md` | Dreaming; direct user request | Session start, budgeted |
| Episodic | `memory/YYYY-MM-DD.md` | Agent during work; memory flush | Searchable, not injected every turn |
| Prospective | Standing intents / cron | `intent` tool | On trigger |
| Review | `DREAMS.md` | Dreaming phases | Human reading |

Recall tools from `memory-core`: `memory_search`, `memory_get`. Those are read-only.

Curation of daily notes into `MEMORY.md` is a **background dreaming sweep**, not a turn-level tool call. Pre-compaction **memory flush** is a silent extra turn, not the mid-conversation write this benchmark measures.

## What this adapter models

**Extraction (`--system openclaw`)** tests the in-conversation write path:

- Prompt: verbatim workspace `AGENTS.md` Session Startup + Memory sections ([template](https://github.com/openclaw/openclaw/blob/f95adf1738a01f7c65af2b3d332521d2c09a7369/docs/reference/templates/AGENTS.md)). The USER.md HTML metadata line is taken from the USER.md template (AGENTS.md’s HTML is stripped in some renders).
- Tools: real `write(path, content)` and `edit(path, edits)` schemas from OpenClaw `src/agents/sessions/tools/` — the only bench-specific addition; AGENTS.md does not name those tools.
- Stored fact: file content (or `edits[].newText`) **only when the path is a memory file** (`USER.md`, `MEMORY.md`, `memory/*.md`)

Writes to `AGENTS.md` / `SOUL.md` / other workspace files do not count as stored facts, even though Write It Down tells the agent to put lessons in `AGENTS.md`.

**Value benchmark** uses OpenClaw recall:

- Prompt injection formats memories as `MEMORY.md` bootstrap content
- Tool mode exposes `memory_search` / `memory_get` instead of the generic `recall_memory`

## Explicitly out of scope (API-level)

- Dreaming (light / REM / deep promotion into `MEMORY.md`)
- Pre-compaction memory flush
- Provenance / taint gates / standing intents
- Containerised end-to-end OpenClaw (would need a real workspace + `memory-core` plugin)

Those are real OpenClaw subsystems. They are not turn-level tool-calling judgment, which is what extraction measures.

## Run

```bash
llm-memory-bench list-systems

llm-memory-bench run \
  --dataset datasets/converted/alpsbench-task1.yaml \
  --system openclaw \
  --provider vertex \
  --model claude-sonnet-4@20250514 \
  --output results/run-openclaw.json
```

Prompt artifacts: `prompts/openclaw/` (version `v1_2026-09`, source commit `f95adf17` on `openclaw/openclaw`).

## Implementation

- `src/llm_memory_bench/systems/openclaw.py`
- `prompts/openclaw/metadata.yaml`
- `prompts/openclaw/v1_2026-09.yaml`

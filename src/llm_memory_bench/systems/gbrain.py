from __future__ import annotations

from ..providers.base import ToolCall
from .base import MemorySystem

# All prompts and schemas sourced from github.com/garrytan/gbrain master branch.
# gbrain version at time of capture: 0.46.19.0 (package.json)
# Last verified against GitHub master: 2026-08-18
#
# Prompt: docs/tutorials/connect-coding-agent.md "Brain-first protocol" block
#   — the canonical text gbrain tells users to paste into CLAUDE.md / AGENTS.md.
# Tool schemas: src/core/verbs.ts (remember), src/core/ops/pages.ts (put_page,
#   capture operations)
# Tool descriptions: src/core/operations-descriptions.ts (CAPTURE_DESCRIPTION)

GBRAIN_VERSION = "0.46.19.0"
PROMPT_SOURCE = "docs/tutorials/connect-coding-agent.md"
TOOL_SCHEMA_SOURCE = "src/core/verbs.ts + src/core/ops/pages.ts"

# The gbrain MCP server does not inject its own system prompt. The prompt is
# user-provided via CLAUDE.md/AGENTS.md. This is the recommended "Brain-first
# protocol" block from the gbrain project's connect tutorial, which users are
# told to paste into their agent's instructions file.
SYSTEM_PROMPT = """\
You are a helpful assistant. You have a knowledge brain connected over MCP.

## Brain-first protocol

Before answering any question about people, companies, decisions, projects, \
or past context:

1. **Brain first — route by the shape of the question.** Exact names or known \
tokens → `search` (cheap hybrid, no expansion). Concept, landscape, or \
"all the X that do Y" questions → `query` FIRST — it recovers synonym \
phrasings `search` misses, and a populated `search` result set is not proof \
of coverage. On the verbs surface the same split is `recall` (retrieve) \
vs `synthesize` (reasoned answer). Check the brain BEFORE answering from \
memory or asking me. Never ask "who is X?" or "what did we decide about Y?" \
before checking — the brain probably already knows.
2. **Write back.** When I make a decision, mention a new person/company, or \
land on an idea worth keeping, write it to the brain: `remember` on the \
verbs surface (one fact, with provenance), or `put_page` on the full surface \
(entity pages under people/, companies/; decisions under decisions/ or \
notes/). One insight, one page, linked.
3. **Cite.** When you answer from the brain, name the page you used."""

# Tool schemas from src/core/verbs.ts and src/core/ops/pages.ts, converted
# from Operation format to MCP tool definition format via
# src/mcp/tool-defs.ts buildToolDefs().
# Server-stamped fields (source_kind, source_uri, ingested_via) excluded —
# remote MCP callers have values overwritten by the server (CV6 trust gate).

# Schema from src/core/verbs.ts remember Operation.
REMEMBER_TOOL = {
    "name": "remember",
    "description": (
        "MEMORY VERB (v1): save one fact to durable agent memory — the "
        "protocol write verb. provenance is REQUIRED (free text, e.g. "
        '"conversation 2026-06-12", "user said in chat", "import: notes.md"). '
        "Set `entity` whenever the fact is about a specific "
        "person/company/project — entity-scoped recall will not find it "
        "otherwise. ttl accepts duration shorthand (\"30d\", \"12h\") or an "
        "absolute ISO 8601 timestamp; ISO-8601 durations like \"P30D\" are "
        "rejected with a fix. visibility defaults to \"world\" (readable by "
        "every agent connected to this brain; pass \"private\" for "
        "local-CLI-only facts). Response: branch on `status` "
        "(inserted|duplicate|superseded), never on `status_text` (human "
        "rendering only). On duplicate, `id` is the EXISTING fact's id. For "
        "bulk extraction from a raw transcript use extract_facts instead."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "fact": {
                "type": "string",
                "description": "The fact to remember, one claim per call.",
            },
            "provenance": {
                "type": "string",
                "description": (
                    "Where this fact came from (REQUIRED, free text, max 500 "
                    'chars). Examples: "conversation 2026-06-12", "user said '
                    'in chat", "import: meeting-notes.md".'
                ),
            },
            "ttl": {
                "type": "string",
                "description": (
                    "Optional expiry: duration shorthand (\"30d\", \"12h\", "
                    "\"45m\") or absolute ISO 8601 timestamp. NOT ISO-8601 "
                    "durations (\"P30D\" is rejected). Omit = never expires."
                ),
            },
            "entity": {
                "type": "string",
                "description": (
                    "Person/company/project this fact is about (name or slug; "
                    "canonicalized server-side). Set it whenever the fact has "
                    "a subject — entity-scoped recall misses unattributed facts."
                ),
            },
            "kind": {
                "type": "string",
                "enum": ["event", "preference", "commitment", "belief", "fact"],
                "description": (
                    "Fact kind: event | preference | commitment | belief | "
                    "fact (default)."
                ),
            },
            "visibility": {
                "type": "string",
                "enum": ["world", "private"],
                "description": (
                    "world (default): readable by every agent connected to "
                    "this brain. private: local CLI reads only."
                ),
            },
        },
        "required": ["fact", "provenance"],
    },
}

PUT_PAGE_TOOL = {
    "name": "put_page",
    "description": (
        "Write/update a page (markdown with frontmatter). Chunks, embeds, "
        "reconciles tags, and (when auto_link/auto_timeline are enabled) "
        "extracts + reconciles graph links and timeline entries. For large "
        "content on Windows (pipe-buffer limit ~45KB) or any file-as-input "
        "workflow, use `gbrain capture --file PATH --slug SLUG` — capture "
        "reads the file as a Buffer with a binary-NUL guard and adds "
        "provenance write-through (v0.39.3.0)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "slug": {
                "type": "string",
                "description": "Page slug",
            },
            "content": {
                "type": "string",
                "description": "Full markdown content with YAML frontmatter",
            },
            "allow_empty": {
                "type": "boolean",
                "description": (
                    "Allow overwriting an existing non-empty page with "
                    "empty/whitespace-only content (default: false). Without "
                    "it, put_page rejects the empty overwrite — the "
                    "empty-stdin failure class."
                ),
            },
        },
        "required": ["slug", "content"],
    },
}

# Description from src/core/operations-descriptions.ts CAPTURE_DESCRIPTION.
# Schema from src/core/ops/pages.ts capture Operation.
CAPTURE_TOOL = {
    "name": "capture",
    "description": (
        'Capture a quick note into the brain — the "just remember this" '
        "write. Auto-derives a stable inbox/ slug from the content date + "
        "hash (recapturing identical text is idempotent), merges frontmatter, "
        "refuses binary/empty payloads, then delegates to put_page "
        "(inheriting its fences and provenance stamping). Prefer capture for "
        "quick notes and put_page when you need to control the slug, type, "
        "or an existing page's content. For structured facts about entities, "
        "prefer remember."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": (
                    "Markdown or plain text to capture. File paths are NOT "
                    "accepted over MCP — read the file yourself and pass its "
                    "content (the CLI --file lane is local-only)."
                ),
            },
            "slug": {
                "type": "string",
                "description": (
                    "Target slug. Default: inbox/YYYY-MM-DD-<sha8-of-content> "
                    "(stable per content — recapturing identical text hits the "
                    "same slug); type diary/event routes under life/. Fenced "
                    "clients: the default lands under your first bound prefix."
                ),
            },
            "type": {
                "type": "string",
                "description": (
                    "Page type for the stamped frontmatter (default 'note')."
                ),
            },
        },
        "required": ["content"],
    },
}


class GBrainMemorySystem(MemorySystem):
    name = "gbrain"
    description = (
        "GBrain MCP knowledge-brain system with remember (atomic fact write), "
        "put_page (structured markdown pages), and capture (quick notes)"
    )
    prompt_version = GBRAIN_VERSION
    tool_schema_version = GBRAIN_VERSION

    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def tool_definitions(self) -> list[dict]:
        return [REMEMBER_TOOL, PUT_PAGE_TOOL, CAPTURE_TOOL]

    def extract_stored_fact(self, tool_call: ToolCall) -> str:
        if tool_call.name == "remember":
            return tool_call.arguments.get("fact", "")
        if tool_call.name == "put_page":
            return tool_call.arguments.get("content", "")
        if tool_call.name == "capture":
            return tool_call.arguments.get("content", "")
        return ""

    def format_tool_result(self, tool_call: ToolCall) -> dict:
        if tool_call.name == "remember":
            return {
                "id": "1",
                "status": "inserted",
                "status_text": "remembered as fact #1",
                "entity_slug": None,
                "valid_until": None,
                "protocol_version": 1,
            }
        if tool_call.name == "put_page":
            slug = tool_call.arguments.get("slug", "unknown")
            return {"status": "ok", "slug": slug, "version": 1}
        if tool_call.name == "capture":
            slug = tool_call.arguments.get("slug", "inbox/auto")
            return {
                "status": "ok",
                "slug": slug,
                "channel": "capture",
                "dedupe": "identical normalized content produces the same "
                "default slug and hash",
            }
        return {"status": "ok"}

    def version_info(self) -> dict:
        return {
            "system": self.name,
            "gbrain_version": GBRAIN_VERSION,
            "prompt_source": f"github.com/garrytan/gbrain {PROMPT_SOURCE}",
            "tool_schema_source": f"github.com/garrytan/gbrain {TOOL_SCHEMA_SOURCE}",
            "last_verified": "2026-08-18",
        }

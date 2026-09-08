from __future__ import annotations

from pathlib import Path

import yaml

from ..providers.base import ToolCall
from .base import MemorySystem

PROMPTS_DIR = Path(__file__).parent.parent.parent.parent / "prompts" / "gbrain"

# All prompts and schemas sourced from github.com/garrytan/gbrain master branch.
# gbrain version at time of capture: 0.46.19.0 (package.json)
# Last verified against GitHub master: 2026-08-18
#
# Prompt: docs/tutorials/connect-coding-agent.md "Brain-first protocol" block
#   — the canonical text gbrain tells users to paste into CLAUDE.md / AGENTS.md.
# Tool schemas: src/core/verbs.ts (remember), src/core/ops/pages.ts (put_page,
#   capture operations)
# Tool descriptions: src/core/operations-descriptions.ts (CAPTURE_DESCRIPTION)
#
# TODO: Migrate tool schemas to YAML files (currently kept in Python)

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
    """GBrain MCP knowledge-brain system.

    Loads prompts from versioned YAML files at design time.
    Tool schemas kept in Python for now (migration TODO).
    """

    name = "gbrain"
    description = (
        "GBrain MCP knowledge-brain system with remember (atomic fact write), "
        "put_page (structured markdown pages), and capture (quick notes)"
    )

    def __init__(self, prompt_version: str | None = None, **kwargs):
        """Initialize with optional prompt version.

        Args:
            prompt_version: Prompt version to use (e.g., "v0.46.19_2024-12").
                           Defaults to default_version from metadata.yaml.
            **kwargs: Ignored (for compatibility with systems that use scenario, etc.)

        Raises:
            FileNotFoundError: If prompt files not found
        """
        self.prompt_version = prompt_version or self._get_default_version()
        self._prompts = self._load_prompts(self.prompt_version)

    @classmethod
    def _get_default_version(cls) -> str:
        """Load default version from metadata.yaml."""
        metadata_path = PROMPTS_DIR / "metadata.yaml"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Prompt metadata not found: {metadata_path}. "
                "Run design-time prompt extraction first."
            )
        with open(metadata_path) as f:
            metadata = yaml.safe_load(f)
        return metadata["default_version"]

    @classmethod
    def _load_prompts(cls, version: str) -> dict:
        """Load prompts from versioned YAML file."""
        prompt_path = PROMPTS_DIR / f"{version}.yaml"
        if not prompt_path.exists():
            raise FileNotFoundError(
                f"Prompt file not found: {prompt_path}. "
                f"Available versions: {list(p.stem for p in PROMPTS_DIR.glob('v*.yaml'))}"
            )
        with open(prompt_path) as f:
            return yaml.safe_load(f)

    def system_prompt(self) -> str:
        """Return the system prompt from loaded YAML."""
        return self._prompts["system_prompt"]

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
        """Include prompt version, source, and metadata in version info."""
        # Load metadata for source information
        metadata_path = PROMPTS_DIR / "metadata.yaml"
        with open(metadata_path) as f:
            metadata = yaml.safe_load(f)

        pulled_date = self._prompts.get("pulled_date")
        return {
            "system": self.name,
            "prompt_version": self.prompt_version,
            "prompt_source": f"{metadata['source']['repo']} {metadata['source']['path']}",
            "prompt_pulled_date": str(pulled_date) if pulled_date else None,
            "source_commit": self._prompts.get("source_commit"),
            "tool_schema_source": f"{metadata['source']['repo']} {metadata['source']['tool_schemas']}",
        }

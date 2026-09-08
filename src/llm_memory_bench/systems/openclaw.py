"""OpenClaw file-backed memory system.

OpenClaw has no dedicated save-memory tool. The agent persists facts by writing
plain Markdown in the workspace:

- ``USER.md`` — durable preference / profile directives
- ``MEMORY.md`` — curated non-profile facts and decisions
- ``memory/YYYY-MM-DD.md`` — episodic daily notes

Writes go through the generic ``write`` / ``edit`` file tools. Recall tools
(``memory_search`` / ``memory_get``) are modeled for the value benchmark only.

Source: https://github.com/openclaw/openclaw
Docs: https://docs.openclaw.ai/concepts/memory-architecture
"""
from __future__ import annotations

from pathlib import Path

import yaml

from ..providers.base import ToolCall
from .base import MemorySystem

PROMPTS_DIR = Path(__file__).parent.parent.parent.parent / "prompts" / "openclaw"

MEMORY_FILENAMES = frozenset({"MEMORY.md", "USER.md", "memory.md"})

WRITE_TOOL = {
    "name": "write",
    "description": (
        "Write/overwrite file; creates parent directories. "
        "Use only for new files or complete rewrites."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path; relative/absolute.",
            },
            "content": {
                "type": "string",
                "description": "File content.",
            },
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    },
}

EDIT_TOOL = {
    "name": "edit",
    "description": (
        "Exact single-file replacements. oldText unique/non-overlapping "
        "against original. Merge nearby changes; omit large unchanged spans."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path; relative/absolute.",
            },
            "edits": {
                "type": "array",
                "description": (
                    "Targeted replacements against original file; "
                    "no overlap/nesting. Merge nearby changes."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "oldText": {
                            "type": "string",
                            "description": (
                                "Exact original text; unique and "
                                "non-overlapping in this call."
                            ),
                        },
                        "newText": {
                            "type": "string",
                            "description": "Replacement text.",
                        },
                    },
                    "required": ["oldText", "newText"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["path", "edits"],
        "additionalProperties": False,
    },
}

# From extensions/memory-core/src/memory-tool-contract.ts (value benchmark).
MEMORY_SEARCH_TOOL = {
    "name": "memory_search",
    "description": (
        "Mandatory recall step: semantically search MEMORY.md, USER.md, "
        "Markdown files recursively under memory/ before answering questions "
        "about prior work, decisions, dates, people, preferences, or todos. "
        "If response has disabled=true or stale=true, tell the user and include "
        "the warning/action guidance."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "maxResults": {"type": "integer", "minimum": 1},
            "minScore": {"type": "number"},
            "corpus": {
                "type": "string",
                "enum": ["memory", "wiki", "all", "sessions"],
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}

MEMORY_GET_TOOL = {
    "name": "memory_get",
    "description": (
        "Safe exact excerpt read from MEMORY.md, USER.md, Markdown files "
        "recursively under memory/. Defaults to a bounded excerpt when lines "
        "are omitted and includes truncation/continuation info when more "
        "content exists. status=ok means the requested excerpt was read; "
        "status=not_found means every requested available corpus missed."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "from": {"type": "integer", "minimum": 1},
            "lines": {"type": "integer", "minimum": 1},
            "corpus": {
                "type": "string",
                "enum": ["memory", "wiki", "all"],
            },
        },
        "required": ["path"],
        "additionalProperties": False,
    },
}


def _normalize_path(path: str) -> str:
    return path.replace("\\", "/").lstrip("./")


def is_memory_path(path: str) -> bool:
    """True when the path is an OpenClaw memory surface (not AGENTS.md, etc.)."""
    if not path:
        return False
    normalized = _normalize_path(path)
    name = normalized.rsplit("/", 1)[-1]
    if name in MEMORY_FILENAMES:
        return True
    if "/memory/" in f"/{normalized}" and normalized.endswith(".md"):
        return True
    if normalized.startswith("memory/") and normalized.endswith(".md"):
        return True
    return False


def _edit_texts(arguments: dict) -> list[str]:
    texts: list[str] = []
    edits = arguments.get("edits")
    if isinstance(edits, list):
        for edit in edits:
            if isinstance(edit, dict):
                new_text = edit.get("newText") or edit.get("new_string") or ""
                if new_text:
                    texts.append(str(new_text))
    # Legacy flat shape some models emit (oldText/newText at top level).
    top = arguments.get("newText") or arguments.get("new_string")
    if top:
        texts.append(str(top))
    return texts


class OpenClawMemorySystem(MemorySystem):
    """OpenClaw file-backed memory: write/edit to USER.md, MEMORY.md, daily notes."""

    name = "openclaw"
    description = (
        "OpenClaw file-backed memory: write/edit to USER.md, MEMORY.md, "
        "and memory/YYYY-MM-DD.md daily notes"
    )

    def __init__(self, prompt_version: str | None = None, **kwargs):
        """Initialize with optional prompt version.

        Args:
            prompt_version: Prompt version to use (e.g., "v1_2026-09").
                           Defaults to default_version from metadata.yaml.
            **kwargs: Ignored (for compatibility with systems that use scenario, etc.)
        """
        self.prompt_version = prompt_version or self._get_default_version()
        self._prompts = self._load_prompts(self.prompt_version)

    @classmethod
    def _get_default_version(cls) -> str:
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
        prompt_path = PROMPTS_DIR / f"{version}.yaml"
        if not prompt_path.exists():
            raise FileNotFoundError(
                f"Prompt file not found: {prompt_path}. "
                f"Available versions: {list(p.stem for p in PROMPTS_DIR.glob('v*.yaml'))}"
            )
        with open(prompt_path) as f:
            return yaml.safe_load(f)

    def system_prompt(self) -> str:
        return self._prompts["system_prompt"]

    def tool_definitions(self) -> list[dict]:
        return [WRITE_TOOL, EDIT_TOOL]

    def extract_stored_fact(self, tool_call: ToolCall) -> str:
        path = str(
            tool_call.arguments.get("path")
            or tool_call.arguments.get("file_path")
            or ""
        )
        if not is_memory_path(path):
            return ""

        if tool_call.name == "write":
            return str(tool_call.arguments.get("content") or "")

        if tool_call.name == "edit":
            return "\n".join(_edit_texts(tool_call.arguments))

        return ""

    def format_tool_result(self, tool_call: ToolCall) -> dict:
        path = tool_call.arguments.get("path") or tool_call.arguments.get(
            "file_path", "unknown"
        )
        if tool_call.name == "write":
            content = tool_call.arguments.get("content") or ""
            nbytes = len(content.encode("utf-8"))
            return {
                "changed": True,
                "created": True,
                "message": f"Successfully wrote {nbytes} bytes to {path}",
            }
        if tool_call.name == "edit":
            n = len(_edit_texts(tool_call.arguments)) or 1
            return {
                "changed": True,
                "message": f"Successfully replaced {n} block(s) in {path}.",
            }
        return {"status": "ok"}

    def inject_memories(self, base_prompt: str, memories: list[dict]) -> str:
        if not memories:
            return base_prompt
        lines = [f"- {m['fact']}" for m in memories]
        block = (
            "The following MEMORY.md content was loaded at session start "
            "(main/private session, budgeted):\n\n"
            "# MEMORY.md\n\n" + "\n".join(lines)
        )
        return base_prompt + "\n\n" + block

    def recall_tool_definitions(self) -> list[dict]:
        return [MEMORY_SEARCH_TOOL, MEMORY_GET_TOOL]

    def format_recall_result(self, memories: list[dict]) -> dict:
        if not memories:
            return {
                "results": [],
                "message": "No memories found.",
            }
        results = []
        for i, memory in enumerate(memories, start=1):
            fact = memory.get("fact", "")
            results.append(
                {
                    "path": "MEMORY.md",
                    "startLine": i,
                    "endLine": i,
                    "snippet": fact,
                    "score": 1.0,
                    "source": f"MEMORY.md#{i}",
                }
            )
        return {"results": results}

    def version_info(self) -> dict:
        metadata_path = PROMPTS_DIR / "metadata.yaml"
        with open(metadata_path) as f:
            metadata = yaml.safe_load(f)

        pulled_date = self._prompts.get("pulled_date")
        source = metadata["source"]
        return {
            "system": self.name,
            "prompt_version": self.prompt_version,
            "prompt_source": f"{source['repo']} {source['paths']['prompt']}",
            "prompt_pulled_date": str(pulled_date) if pulled_date else None,
            "source_commit": self._prompts.get("source_commit"),
            "tool_schema_source": (
                f"{source['repo']} {source['paths']['write_tool']}"
            ),
        }

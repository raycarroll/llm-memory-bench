from __future__ import annotations

from pathlib import Path

import yaml

from ..providers.base import ToolCall
from .base import MemorySystem

# Faithfully reproduces the auto-memory instructions from Claude Code's
# system prompt. The LLM receives these as part of its system context and
# must decide what to write, which type to use, and how to structure the
# frontmatter and body.

PROMPTS_DIR = Path(__file__).parent.parent.parent.parent / "prompts" / "claude_code"

SAVE_MEMORY_TOOL = {
    "name": "save_memory",
    "description": (
        "Save a memory to the persistent file-based memory system. "
        "Each memory has a name (short kebab-case slug), a one-line description "
        "used to decide relevance in future conversations, a type (user, feedback, "
        "project, or reference), and a body with the memory content."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": (
                    "Short kebab-case slug for the memory file, e.g. "
                    "'user-role', 'feedback-testing', 'project-auth-rewrite'."
                ),
            },
            "description": {
                "type": "string",
                "description": (
                    "One-line summary used to decide relevance in future "
                    "conversations. Be specific."
                ),
            },
            "type": {
                "type": "string",
                "enum": ["user", "feedback", "project", "reference"],
                "description": "The type of memory.",
            },
            "body": {
                "type": "string",
                "description": (
                    "The memory content. For feedback/project types, structure "
                    "as: rule/fact, then a Why: line and a How to apply: line."
                ),
            },
        },
        "required": ["name", "description", "type", "body"],
    },
}


class ClaudeCodeMemorySystem(MemorySystem):
    name = "claude_code"
    description = (
        "Claude Code's file-based auto-memory system with typed memories "
        "(user, feedback, project, reference) and structured frontmatter"
    )

    def __init__(self, prompt_version: str | None = None, **kwargs):
        """Initialize with optional prompt version.

        Args:
            prompt_version: Prompt version to use (e.g., "v1_2026-08").
                           Defaults to default_version from metadata.yaml.
            **kwargs: Ignored (for compatibility with systems that use scenario, etc.)
        """
        self.prompt_version = prompt_version or self._get_default_version()
        self._prompts = self._load_prompts(self.prompt_version)

    @classmethod
    def _get_default_version(cls) -> str:
        """Load default version from metadata.yaml."""
        metadata_path = PROMPTS_DIR / "metadata.yaml"
        with open(metadata_path) as f:
            metadata = yaml.safe_load(f)
        return metadata["default_version"]

    @classmethod
    def _load_prompts(cls, version: str) -> dict:
        """Load prompts from versioned YAML file."""
        prompt_path = PROMPTS_DIR / f"{version}.yaml"
        with open(prompt_path) as f:
            return yaml.safe_load(f)

    def system_prompt(self) -> str:
        return self._prompts["system_prompt"]

    def tool_definitions(self) -> list[dict]:
        return [SAVE_MEMORY_TOOL]

    def extract_stored_fact(self, tool_call: ToolCall) -> str:
        body = tool_call.arguments.get("body", "")
        desc = tool_call.arguments.get("description", "")
        if body:
            return body
        return desc

    def format_tool_result(self, tool_call: ToolCall) -> dict:
        name = tool_call.arguments.get("name", "unknown")
        return {
            "status": "success",
            "message": f"Memory saved to {name}.md",
        }

    def version_info(self) -> dict:
        """Include prompt version and source."""
        metadata_path = PROMPTS_DIR / "metadata.yaml"
        with open(metadata_path) as f:
            metadata = yaml.safe_load(f)

        pulled_date = self._prompts.get("pulled_date")
        return {
            "system": self.name,
            "prompt_version": self.prompt_version,
            "prompt_source": f"{metadata['source']['repo']} {metadata['source']['path']}",
            "prompt_pulled_date": str(pulled_date) if pulled_date else None,
        }

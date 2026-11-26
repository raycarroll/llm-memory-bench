from __future__ import annotations

from pathlib import Path

import yaml

from ..providers.base import ToolCall
from .base import MemorySystem

PROMPTS_DIR = Path(__file__).parent.parent.parent.parent / "prompts" / "simple"

# TODO: Migrate tool schemas to YAML (keeping in Python for now)

TOOL_DEFINITION = {
    "name": "add_memory",
    "description": (
        "Store a long-term fact about the user for future sessions. "
        "Only store persistent personal information, preferences, or constraints. "
        "Do not store ephemeral requests, greetings, or task-specific details."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "fact": {
                "type": "string",
                "description": "The fact to remember about the user.",
            },
            "category": {
                "type": "string",
                "enum": ["preference", "personal", "technical", "contextual"],
                "description": "Category of the memory.",
            },
        },
        "required": ["fact"],
    },
}


class SimpleMemorySystem(MemorySystem):
    name = "simple"
    description = "Baseline single-tool memory system with add_memory(fact, category)"

    def __init__(self, prompt_version: str | None = None, **kwargs):
        """Initialize with optional prompt version.

        Args:
            prompt_version: Prompt version to use (e.g., "v1_2024-11").
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
        """Return system prompt from loaded YAML."""
        return self._prompts["system_prompt"]

    def system_prompt_for_mode(self, mode: str = "default") -> str:
        """Return the normal or capture-first prompt variant."""
        if mode == "capture_only" and self._prompts.get("capture_only_system_prompt"):
            return self._prompts["capture_only_system_prompt"]
        return self.system_prompt()

    def tool_definitions(self) -> list[dict]:
        return [TOOL_DEFINITION]

    def extract_stored_fact(self, tool_call: ToolCall) -> str:
        return tool_call.arguments.get("fact", "")

    def format_tool_result(self, tool_call: ToolCall) -> dict:
        return {
            "status": "success",
            "message": f"Memory saved: {tool_call.arguments.get('fact', '')}",
        }

    def version_info(self) -> dict:
        """Include prompt version and source in version info for tracking."""
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
        }

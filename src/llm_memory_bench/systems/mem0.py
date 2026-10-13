from __future__ import annotations

from pathlib import Path

import yaml

from ..providers.base import ToolCall
from .base import MemorySystem

PROMPTS_DIR = Path(__file__).parent.parent.parent.parent / "prompts" / "mem0"


class Mem0MemorySystem(MemorySystem):
    """Mem0 memory system with automatic fact extraction.

    Based on Mem0's agent plugin core and Platform API.
    Uses their extraction instructions for personal memory capture.
    """

    name = "mem0"
    description = (
        "Mem0 memory extraction with automatic fact extraction from conversations. "
        "Stores personal facts, preferences, and user context."
    )

    def __init__(self, prompt_version: str | None = None, **kwargs):
        """Initialize with optional prompt version.

        Args:
            prompt_version: Prompt version to use (e.g., "v1_2026-10").
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

    def tool_definitions(self) -> list[dict]:
        """Convert tool schema from YAML to provider tool format."""
        tool_schema = self._prompts["tool_schema"]
        return [
            {
                "name": tool_schema["name"],
                "description": tool_schema["description"],
                "input_schema": tool_schema["input_schema"],
            }
        ]

    def extract_stored_fact(self, tool_call: ToolCall) -> str:
        """Extract content from storage actions only (not search).

        Mem0 automatically extracts facts from messages, so we return the
        messages content for add actions.
        """
        if tool_call.name == "mem0_memory":
            action = tool_call.arguments.get("action", "")
            # Only count storage actions (add/update)
            # Skip retrieval actions (search)
            if action == "add":
                # For add, the fact is in the messages being processed
                messages = tool_call.arguments.get("messages", [])
                if messages:
                    # Extract user messages
                    user_msgs = [m.get("content", "") for m in messages if m.get("role") == "user"]
                    return " ".join(user_msgs)
            elif action == "update":
                # For update, return the new content
                return tool_call.arguments.get("content", "")
        return ""

    def format_tool_result(self, tool_call: ToolCall) -> dict:
        """Return mock success response for memory operations."""
        if tool_call.name == "mem0_memory":
            action = tool_call.arguments.get("action", "")

            if action == "add":
                return {
                    "status": "PENDING",
                    "message": "Memories are being extracted and stored",
                    "event_id": "mock_event_id"
                }
            elif action == "search":
                # Mock search result
                query = tool_call.arguments.get("query", "")
                return {
                    "results": [],
                    "count": 0,
                    "message": f"Searched for: {query}"
                }
            elif action == "update":
                memory_id = tool_call.arguments.get("memory_id", "")
                return {
                    "status": "success",
                    "message": f"Updated memory {memory_id}"
                }
            elif action == "delete":
                memory_id = tool_call.arguments.get("memory_id", "")
                return {
                    "status": "success",
                    "message": f"Deleted memory {memory_id}"
                }

        return {"status": "ok"}

    def version_info(self) -> dict:
        return {
            "system": self.name,
            "prompt_version": self.prompt_version,
            "prompt_source": "https://github.com/mem0ai/mem0 agent-plugin-core",
            "prompt_pulled_date": self._prompts.get("pulled_date", "2026-10-05"),
            "source_commit": self._prompts.get("source_commit", "unknown"),
        }

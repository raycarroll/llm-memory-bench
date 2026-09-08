from __future__ import annotations

import uuid
from pathlib import Path

import yaml

from ..providers.base import ToolCall
from .base import MemorySystem

PROMPTS_DIR = Path(__file__).parent.parent.parent.parent / "prompts" / "memoryhub"

# TODO: Migrate tool schemas to YAML (currently kept in Python for now)


class MemoryHubMemorySystem(MemorySystem):
    """MemoryHub unified memory interface with lazy loading pattern.

    Loads prompts from versioned YAML files at design time.
    Tool schemas remain in Python for now (TODO: migrate to YAML).
    """

    name = "memoryhub"
    description = (
        "MemoryHub unified memory(action=...) interface with scoped writes, "
        "weighted memories, and content type classification"
    )

    def __init__(self, prompt_version: str | None = None, **kwargs):
        """Initialize with optional prompt version.

        Args:
            prompt_version: Prompt version to use (e.g., "v1_lazy_2026-08").
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
        """Convert tool schema from YAML to Anthropic tool format."""
        tool_schema = self._prompts["tool_schema"]
        return [
            {
                "name": tool_schema["name"],
                "description": tool_schema["description"],
                "input_schema": tool_schema["input_schema"],
            }
        ]

    def extract_stored_fact(self, tool_call: ToolCall) -> str:
        if tool_call.name == "memory":
            action = tool_call.arguments.get("action", "")
            if action == "write":
                return tool_call.arguments.get("content", "")
        return ""

    def format_tool_result(self, tool_call: ToolCall) -> dict:
        if tool_call.name != "memory":
            return {"status": "ok"}

        action = tool_call.arguments.get("action", "")

        if action == "write":
            memory_id = str(uuid.uuid4())
            content = tool_call.arguments.get("content", "")
            stub = content[:80] + ("..." if len(content) > 80 else "")
            return {
                "memory": {
                    "id": memory_id,
                    "content": content,
                    "stub": stub,
                    "scope": tool_call.arguments.get("scope", "user"),
                    "weight": (
                        tool_call.arguments.get("options", {}).get("weight", 0.7)
                        if isinstance(tool_call.arguments.get("options"), dict)
                        else 0.7
                    ),
                    "content_type": "experiential",
                    "is_current": True,
                    "created_at": "2026-01-01T00:00:00Z",
                },
                "curation": {
                    "blocked": False,
                    "similar_count": 0,
                    "nearest_id": None,
                    "nearest_score": None,
                    "flags": [],
                },
            }

        if action == "search":
            return {"memories": [], "total": 0}

        if action == "read":
            return {"error": "Memory not found"}

        if action == "status":
            return {
                "session_id": "bench-session",
                "user_id": "bench-user",
                "scopes": ["user"],
            }

        return {"status": "ok"}

    def version_info(self) -> dict:
        """Include prompt version, source commit, and pulled date for tracking."""
        # Load metadata for source information
        metadata_path = PROMPTS_DIR / "metadata.yaml"
        with open(metadata_path) as f:
            metadata = yaml.safe_load(f)

        pulled_date = self._prompts.get("pulled_date")
        return {
            "system": self.name,
            "prompt_version": self.prompt_version,
            "prompt_source": f"{metadata['source']['repo']} {metadata['source']['paths']['prompt']}",
            "prompt_pulled_date": str(pulled_date) if pulled_date else None,
            "source_commit": self._prompts.get("source_commit"),
        }

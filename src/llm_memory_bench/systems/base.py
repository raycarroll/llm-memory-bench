from __future__ import annotations

from abc import ABC, abstractmethod

from ..providers.base import ToolCall


class MemorySystem(ABC):
    """A real memory system's prompt + tools + extraction logic."""

    name: str
    description: str

    def __init__(self, **kwargs):
        """Initialize system with optional configuration.

        Most systems don't need configuration and can use the default no-op implementation.
        BFCL systems require 'scenario' parameter.
        """
        pass

    @abstractmethod
    def system_prompt(self) -> str:
        """The actual system prompt this memory system uses."""
        ...

    @abstractmethod
    def tool_definitions(self) -> list[dict]:
        """The actual MCP/tool schemas this system exposes to the LLM."""
        ...

    @abstractmethod
    def extract_stored_fact(self, tool_call: ToolCall) -> str | dict:
        """Extract the semantic content from a tool call for evaluation.

        Returns:
            - str: For most systems, the fact value as a string
            - dict: For AlpsBench, a dict with (label, value, type, preference_attitude, time_scope, emotion)
        """
        ...

    def format_tool_result(self, tool_call: ToolCall) -> dict:
        """Simulated success response the system would return."""
        return {"status": "success"}

    def accepted_tool_names(self) -> set[str]:
        return {t["name"] for t in self.tool_definitions()}

    def version_info(self) -> dict:
        return {"system": self.name}

    def requires_final_sweep(self) -> bool:
        """Whether this system needs a final extraction sweep after conversation.

        Override to return True in sweep variants.
        """
        return False

    def final_sweep_prompt(self) -> str:
        """System prompt for final extraction sweep.

        Override in sweep variants to provide extraction-focused instructions.
        Returns empty string for non-sweep systems.
        """
        return ""

    def inject_memories(self, base_prompt: str, memories: list[dict]) -> str:
        if not memories:
            return base_prompt
        lines = [f"- {m['fact']}" for m in memories]
        block = "Known facts about this user:\n" + "\n".join(lines)
        return base_prompt + "\n\n" + block

    def recall_tool_definitions(self) -> list[dict]:
        return [
            {
                "name": "recall_memory",
                "description": (
                    "Retrieve stored memories about the current user. "
                    "Call this to check what you already know before asking "
                    "the user for information you may have previously saved."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Optional search query to filter memories.",
                        }
                    },
                    "required": [],
                },
            }
        ]

    def format_recall_result(self, memories: list[dict]) -> dict:
        if not memories:
            return {"memories": [], "message": "No memories found."}
        return {
            "memories": [
                {"fact": m["fact"], "category": m.get("category", "general")}
                for m in memories
            ]
        }

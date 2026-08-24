from __future__ import annotations

from abc import ABC, abstractmethod

from ..providers.base import ToolCall


class MemorySystem(ABC):
    """A real memory system's prompt + tools + extraction logic."""

    name: str
    description: str

    @abstractmethod
    def system_prompt(self) -> str:
        """The actual system prompt this memory system uses."""
        ...

    @abstractmethod
    def tool_definitions(self) -> list[dict]:
        """The actual MCP/tool schemas this system exposes to the LLM."""
        ...

    @abstractmethod
    def extract_stored_fact(self, tool_call: ToolCall) -> str:
        """Extract the semantic content from a tool call for evaluation."""
        ...

    def format_tool_result(self, tool_call: ToolCall) -> dict:
        """Simulated success response the system would return."""
        return {"status": "success"}

    def accepted_tool_names(self) -> set[str]:
        return {t["name"] for t in self.tool_definitions()}

    def version_info(self) -> dict:
        return {"system": self.name}

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

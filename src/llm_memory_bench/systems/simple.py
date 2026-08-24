from __future__ import annotations

from ..providers.base import ToolCall
from .base import MemorySystem

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to a memory tool that persists "
    "information across conversations.\n\n"
    "SAVE using the add_memory tool when the user shares:\n"
    "- Personal preferences (likes, dislikes, styles)\n"
    "- Background information (job, education, expertise, location)\n"
    "- Constraints or requirements (allergies, accessibility needs)\n"
    "- Long-term goals or ongoing projects\n"
    "- Technical environment (languages, tools, OS)\n\n"
    "DO NOT save:\n"
    "- Greetings or small talk\n"
    "- One-off task requests ('fix this bug', 'translate this sentence')\n"
    "- Information the user is asking about, not stating about themselves\n"
    "- Ephemeral context ('I'm in a hurry', 'just a quick question')\n"
    "- Facts about other people unless the user explicitly asks you to remember them\n\n"
    "When in doubt, do not save. Only store facts you'd want available in a "
    "completely new conversation with this user."
)

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

    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def tool_definitions(self) -> list[dict]:
        return [TOOL_DEFINITION]

    def extract_stored_fact(self, tool_call: ToolCall) -> str:
        return tool_call.arguments.get("fact", "")

    def format_tool_result(self, tool_call: ToolCall) -> dict:
        return {
            "status": "success",
            "message": f"Memory saved: {tool_call.arguments.get('fact', '')}",
        }

"""Hindsight memory system.

Hindsight is an agent memory system built to create smarter agents that learn over time.
Uses biomimetic data structures: world facts, experiences, observations, and mental models.

MCP Tools:
- retain(content, context, tags, timestamp, metadata, document_id): Store information
- recall(query, budget, tags, max_tokens, ...): Search memories
- reflect(query, context, budget, tags, ...): Generate analysis from memories
"""

from __future__ import annotations

from ..providers.base import ToolCall
from .base import MemorySystem


class HindsightMemorySystem(MemorySystem):
    """Hindsight memory system using retain/recall/reflect MCP tools."""

    name = "hindsight"
    description = (
        "Hindsight agent memory with retain (store), recall (search), "
        "and reflect (analyze). Uses biomimetic data structures for "
        "world facts, experiences, observations, and mental models."
    )

    def system_prompt(self) -> str:
        """Return system prompt - not used as MCP provides instructions via tool descriptions."""
        return ""

    def tool_definitions(self) -> list[dict]:
        """Return tool definitions - not used as tools are provided by MCP server."""
        return []

    def is_memory_tool(self, tool_call: ToolCall) -> bool:
        """Check if a tool call is a Hindsight memory operation."""
        # Hindsight coding-agents package uses hindsight_* tool names
        # Storage tools: hindsight_capture_initiative, hindsight_ingest_document
        # (NOT search/read/reflect which are retrieval-only)
        if tool_call.name in (
            "hindsight_capture_initiative",
            "hindsight_ingest_document",
            # Also support direct MCP calls if used without the package
            "retain",
            "mcp__hindsight__retain",
            "sync_retain",
            "mcp__hindsight__sync_retain",
        ):
            return True
        return False

    def extract_stored_fact(self, tool_call: ToolCall) -> str:
        """Extract content from storage actions only (retain/sync_retain)."""
        # Hindsight coding-agents uses different tools
        if tool_call.name == "hindsight_capture_initiative":
            # Captures initiatives/plans - concatenate title + summary
            title = tool_call.arguments.get("title", "")
            summary = tool_call.arguments.get("summary", "")
            return f"{title}: {summary}" if title and summary else title or summary

        elif tool_call.name == "hindsight_ingest_document":
            # Ingests documents - use content
            return tool_call.arguments.get("content", "")

        # Direct MCP retain calls (if used without the package)
        elif tool_call.name in ("retain", "mcp__hindsight__retain", "sync_retain", "mcp__hindsight__sync_retain"):
            return tool_call.arguments.get("content", "")

        return ""

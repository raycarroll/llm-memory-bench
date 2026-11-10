"""Smoke test MCP tool connectivity.

Tests that Claude can call MCP tools (write_memory) through the configured
MCP servers. Exits with 0 on success, 1 on failure.
"""

import asyncio
import json
import sys
import os
from pathlib import Path

from claude_agent_sdk import query, ClaudeAgentOptions


def _expand_env(value):
    """Expand environment variables in config values."""
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    return value


def _load_project_mcp(cwd: str) -> dict:
    """Load and expand MCP config from .mcp.json."""
    path = Path(cwd) / ".mcp.json"
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return _expand_env(payload.get("mcpServers", {}))


async def test_mcp_write():
    """Test memory storage tool is callable (generic across memory systems)."""
    test_content = f"Smoke test at {os.environ.get('SMOKE_TEST_TIMESTAMP', 'unknown')}"

    # Load MCP config with environment variable expansion
    mcp_servers = _load_project_mcp("/workspace")

    # Craft a system-specific prompt based on configured MCP server
    if mcp_servers:
        server_names = list(mcp_servers.keys())
        if "hindsight" in server_names:
            prompt = f"""Use the hindsight_capture_initiative tool to save this:
title="Smoke Test"
summary="{test_content}"

Don't explain, just call the tool."""
        elif "memoryhub" in server_names:
            prompt = f"""Use the write_memory tool to save this fact: "{test_content}"

Don't explain, just call the tool."""
        else:
            # Generic fallback
            prompt = f"""Use your available memory tool to store: "{test_content}"

Call the tool without explaining."""
    else:
        # No MCP servers configured - use generic prompt
        prompt = f"""Use write_memory to save: "{test_content}"

Don't explain, just call the tool."""

    # Determine allowed tools based on configured MCP servers
    allowed_tool_patterns = []
    if mcp_servers:
        for server_name in mcp_servers.keys():
            # Allow all tools from configured MCP servers
            allowed_tool_patterns.append(f"mcp__{server_name}__*")
    # Fallback for legacy configs
    if not allowed_tool_patterns:
        allowed_tool_patterns = ["mcp__memoryhub__*"]

    options = ClaudeAgentOptions(
        cwd="/workspace",
        permission_mode="dontAsk",
        setting_sources=["project", "user"],  # Include user settings for user-level MCP registrations
        mcp_servers=mcp_servers if mcp_servers else None,
        strict_mcp_config=bool(mcp_servers),
        allowed_tools=allowed_tool_patterns,
        cli_path="/usr/local/bin/claude",
        max_turns=3,
        model=os.environ.get("SMOKE_TEST_MODEL"),
    )

    tool_calls = []
    error = None

    try:
        async for message in query(prompt=prompt, options=options):
            if message.__class__.__name__ == "AssistantMessage":
                content = getattr(message, "content", []) or []
                for block in content:
                    # SDK returns ToolUseBlock objects, not dicts with type field
                    block_class = block.__class__.__name__
                    if block_class == "ToolUseBlock":
                        tool_name = getattr(block, "name", "unknown")
                        tool_args = getattr(block, "input", {}) or {}
                        tool_calls.append({
                            "name": tool_name,
                            "arguments": tool_args
                        })
    except Exception as e:
        error = str(e)

    result = {
        "tool_calls": tool_calls,
        "test_content": test_content,
        "error": error
    }

    print(json.dumps(result, indent=2))

    # Exit with success if we got at least one memory storage tool call
    # Match common memory storage tools across different systems:
    # - MemoryHub: write_memory, update_memory
    # - Hindsight: hindsight_capture_initiative, hindsight_ingest_document, retain, sync_retain
    # - Others: remember, store, save, etc.
    memory_keywords = ["memory", "hindsight", "retain", "remember", "store", "save"]
    memory_tools = [
        tc for tc in tool_calls
        if any(keyword in tc["name"].lower() for keyword in memory_keywords)
    ]
    if memory_tools:
        return 0
    else:
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(test_mcp_write())
    sys.exit(exit_code)

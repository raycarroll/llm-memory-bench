"""MemoryHub-specific smoke test.

Tests that Claude can call write_memory through the MemoryHub MCP server.
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


async def test_memoryhub():
    """Test write_memory tool is callable."""
    test_content = f"Smoke test at {os.environ.get('SMOKE_TEST_TIMESTAMP', 'unknown')}"

    prompt = f"""Use the write_memory tool to save this fact: "{test_content}"

Don't explain, just call the tool."""

    mcp_servers = _load_project_mcp("/workspace")

    options = ClaudeAgentOptions(
        cwd="/workspace",
        permission_mode="dontAsk",
        setting_sources=["project", "user"],
        mcp_servers=mcp_servers if mcp_servers else None,
        strict_mcp_config=bool(mcp_servers),
        allowed_tools=["mcp__memoryhub__*"],
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

    # Success if write_memory was called
    write_memory_calls = [tc for tc in tool_calls if tc["name"] == "mcp__memoryhub__write_memory"]

    # If we got the tool call, "max turns" isn't an error - it just means Claude kept going
    if write_memory_calls and error and "maximum number of turns" in error.lower():
        error = None

    result = {
        "tool_calls": tool_calls,
        "test_content": test_content,
        "error": error
    }

    print(json.dumps(result, indent=2))

    if write_memory_calls:
        return 0
    else:
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(test_memoryhub())
    sys.exit(exit_code)

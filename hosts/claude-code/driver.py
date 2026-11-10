"""Drive multi-turn conversations through Claude Code via the Agent SDK.

Feeds user messages sequentially, resuming the same session so the agent
accumulates context and decides what to save — with its full prompt stack
(system prompt, CLAUDE.md, MCP tools, plugins).
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

from claude_agent_sdk import query, ClaudeAgentOptions, ResultMessage


def _expand_env(value):
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    return value


def _load_project_mcp(cwd: str) -> dict:
    path = Path(cwd) / ".mcp.json"
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return _expand_env(payload.get("mcpServers", {}))


def _configured_cli_path() -> str:
    return os.environ.get("CLAUDE_CODE_CLI_PATH", "/usr/local/bin/claude")


def _cli_metadata(cli_path: str) -> dict[str, str]:
    """Report the actual Claude Code executable selected for the run."""
    try:
        result = subprocess.run(
            [cli_path, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"cli_path": cli_path, "cli_version_error": str(exc)[:200]}
    version = (result.stdout.strip() or result.stderr.strip())[:200]
    metadata = {"cli_path": cli_path}
    if version:
        metadata["cli_version"] = version
    if result.returncode != 0:
        metadata["cli_version_exit_code"] = str(result.returncode)
    return metadata


def _field(value, name: str, default=None):
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _event_path() -> Path | None:
    configured = os.environ.get("LLM_MEMORY_BENCH_EVENT_LOG", "").strip()
    return Path(configured) if configured else None


def _event_offset(path: Path | None) -> int:
    if path is None or not path.is_file():
        return 0
    return path.stat().st_size


def _read_events(path: Path | None, offset: int) -> list[dict]:
    """Read new integration events while keeping the event stream metadata-only."""
    if path is None or not path.is_file():
        return []
    events = []
    with path.open("rb") as handle:
        handle.seek(offset)
        for raw_line in handle:
            try:
                event = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if not isinstance(event, dict):
                continue
            if not all(isinstance(event.get(key), str) for key in ("name", "source", "status")):
                continue
            # The emitter filters sensitive details. Keep this boundary
            # defensive because profiles are independently versioned.
            details = event.get("details")
            if isinstance(details, dict):
                event["details"] = {
                    str(key): str(value)[:120]
                    for key, value in details.items()
                    if not any(
                        word in str(key).lower()
                        for word in (
                            "content", "credential", "password", "prompt",
                            "query", "secret", "token", "authorization", "api_key",
                        )
                    )
                }
            else:
                event.pop("details", None)
            events.append(event)
    return events


def _capture_hook_output() -> bool:
    """Return whether hook stdout should be retained in benchmark artifacts."""
    return os.environ.get("LLM_MEMORY_BENCH_CAPTURE_HOOK_OUTPUT", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _sdk_hook_event(message, capture_output: bool = False) -> dict | None:
    """Convert an SDK hook lifecycle message to benchmark-observable metadata."""
    if message.__class__.__name__ != "HookEventMessage":
        return None
    hook_name = _field(message, "hook_event_name", "unknown")
    phase = _field(message, "subtype", "unknown")
    raw_data = _field(message, "data", {}) or {}
    if not isinstance(raw_data, dict):
        raw_data = {}
    details = {
        "hook_event": str(hook_name),
        "phase": str(phase),
    }
    if capture_output and phase == "hook_response":
        output = raw_data.get("output")
        if output is not None:
            # Hook output is intentionally opt-in because it may contain the
            # memory/context payload delivered to the agent.
            details["hook_output"] = str(output)[:20000]
        for key in ("exit_code", "outcome"):
            if key in raw_data:
                details[key] = str(raw_data[key])[:120]
    return {
        "name": "integration.hook_lifecycle",
        "source": "claude_agent_sdk",
        "status": "observed",
        "details": details,
    }


async def _query_mcp_tools(mcp_servers: dict | None) -> list[dict]:
    """Query configured MCP servers for their available tools."""
    if not mcp_servers:
        return []

    tools_by_server = []

    for server_name, server_config in mcp_servers.items():
        # Return config info - actual tools are available once Claude connects
        tools_by_server.append({
            "server": server_name,
            "status": "configured",
            "type": server_config.get("type", "stdio"),
        })

    return tools_by_server


async def feed_conversation(
    turns: list[dict],
    model: str | None = None,
    cwd: str = "/workspace",
    max_turns: int | None = None,
    profile_mode: bool = False,
    initial_session_id: str | None = None,
) -> dict:
    responses = []
    session_id = initial_session_id
    mcp_servers = _load_project_mcp(cwd) if profile_mode else None
    event_path = _event_path()
    capture_hook_output = _capture_hook_output()
    cli_path = _configured_cli_path()
    runtime = _cli_metadata(cli_path)
    available_tools = await _query_mcp_tools(mcp_servers)

    for i, turn in enumerate(turns):
        if turn["role"] != "user":
            continue

        option_values = {
            "resume": session_id,
            "permission_mode": "dontAsk" if profile_mode else "auto",
            "model": model,
            "cwd": cwd,
            "max_turns": max_turns,
            # Use the explicitly installed Claude Code binary instead of the
            # SDK's bundled default so the benchmarked runtime is versioned by
            # CLAUDE_CODE_VERSION in the image.
            "cli_path": cli_path,
        }
        if profile_mode:
            # Determine allowed tools based on MCP servers
            allowed_tool_patterns = []
            if mcp_servers:
                for server_name in mcp_servers.keys():
                    # Allow all tools from configured MCP servers
                    # Pattern must be mcp__<server>__* - bare wildcards not supported
                    allowed_tool_patterns.append(f"mcp__{server_name}__*")
            # Fallback for legacy memoryhub configs
            if not allowed_tool_patterns:
                allowed_tool_patterns = ["mcp__memoryhub__*"]

            option_values.update({
                # Load both project and user settings to allow user-level MCP registrations
                # (e.g., Hindsight installed via their npm package)
                "setting_sources": ["project", "user"],
                # Pass the materialized MCP map explicitly and reject ambient
                # MCP servers so a benchmark cannot inherit the operator's setup.
                # If no .mcp.json exists, don't set mcp_servers to allow user-level registrations
                "mcp_servers": mcp_servers if mcp_servers else None,
                "strict_mcp_config": bool(mcp_servers),  # Only strict if we have explicit config
                # Observe the installed integration externally through the SDK
                # stream; do not modify or wrap the integration hook script.
                "include_hook_events": True,
                # A headless benchmark cannot answer Claude's interactive
                # permission prompt. Allow only tools exposed by configured
                # MCP servers; dontAsk denies unrelated tools instead of
                # hanging or silently bypassing policy.
                "allowed_tools": allowed_tool_patterns,
            })

        options = ClaudeAgentOptions(**option_values)

        result_text = ""
        subtype = None
        usage = {}
        tool_calls = []
        sdk_integration_events = []
        event_offset = _event_offset(event_path)
        try:
            async for message in query(prompt=turn["content"], options=options):
                # AssistantMessage content includes the actual MCP/tool-use
                # blocks emitted while Claude is working. ResultMessage only
                # contains the final text, so collect calls from the stream.
                if message.__class__.__name__ == "AssistantMessage":
                    content = _field(message, "content", []) or []
                    for block in content:
                        # SDK returns ToolUseBlock objects, not dicts with type field
                        block_class = block.__class__.__name__
                        # Debug: log all block types to stderr
                        if block_class not in ("TextBlock", "ThinkingBlock"):
                            print(f"DEBUG: Found block type: {block_class}", file=sys.stderr)
                        if block_class == "ToolUseBlock":
                            tool_name = _field(block, "name", "unknown")
                            print(f"DEBUG: Capturing tool call: {tool_name}", file=sys.stderr)
                            tool_calls.append({
                                "id": _field(block, "id"),
                                "name": tool_name,
                                "arguments": _field(block, "input", {}) or {},
                            })
                hook_event = _sdk_hook_event(message, capture_output=capture_hook_output)
                if hook_event is not None:
                    sdk_integration_events.append(hook_event)
                if isinstance(message, ResultMessage):
                    session_id = message.session_id
                    subtype = message.subtype
                    raw_usage = getattr(message, "usage", None)
                    if isinstance(raw_usage, dict):
                        usage = raw_usage
                    if message.subtype == "success":
                        result_text = message.result
        except Exception as e:
            responses.append({"turn": i, "error": str(e)[:500]})
            continue

        responses.append({
            "turn": i,
            "response": result_text[:2000],
            "session_id": session_id,
            "subtype": subtype,
            "usage": usage,
            "tool_calls": tool_calls,
            "integration_events": sdk_integration_events + _read_events(event_path, event_offset),
        })

    return {
        "turns_processed": len(responses),
        "session_id": session_id,
        "responses": responses,
        "runtime": runtime,
        "available_tools": available_tools,
        "integration_events": [
            event
            for response in responses
            for event in response.get("integration_events", [])
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--turns", help="JSON array of turns (inline)")
    group.add_argument("--turns-file", help="Path to JSON file with turns")
    parser.add_argument("--model", help="Model to use (e.g. claude-sonnet-4-20250514)")
    parser.add_argument("--cwd", default="/workspace", help="Working directory")
    parser.add_argument("--max-turns", type=int, default=None, help="Max agent turns per user message")
    parser.add_argument("--resume", help="Resume an existing Claude Agent SDK session")
    parser.add_argument(
        "--profile-mode",
        action="store_true",
        help="Load only the materialized project integration and its settings",
    )
    args = parser.parse_args()

    if args.turns_file:
        with open(args.turns_file) as f:
            turns = json.load(f)
    else:
        turns = json.loads(args.turns)

    try:
        cli_path = _configured_cli_path()
        result = asyncio.run(feed_conversation(
            turns,
            model=args.model,
            cwd=args.cwd,
            max_turns=args.max_turns,
            profile_mode=args.profile_mode,
            initial_session_id=args.resume,
        ))
    except Exception as exc:
        # stdout is a machine-readable boundary consumed by the host runner.
        # Preserve that contract even when SDK startup/configuration fails.
        result = {
            "turns_processed": 0,
            "session_id": args.resume,
            "responses": [{
                "turn": 0,
                "error": f"{type(exc).__name__}: {exc}",
            }],
            "runtime": _cli_metadata(cli_path),
        }
    json.dump(result, sys.stdout, indent=2)


if __name__ == "__main__":
    main()

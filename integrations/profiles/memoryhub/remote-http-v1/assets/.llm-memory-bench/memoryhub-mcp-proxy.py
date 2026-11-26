"""Transparent benchmark proxy for a remote MemoryHub MCP server.

The agent sees the upstream MCP tool definitions unchanged.  The proxy only
applies the benchmark arm policy to calls after the schema has been exposed:
capture suppresses reads while allowing isolated writes; replay forwards reads
and writes to the isolated replay project.
"""

from __future__ import annotations

import asyncio
import argparse
from contextvars import ContextVar
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from fastmcp import Client
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequestParams,
    CallToolResult,
    ListToolsResult,
    PaginatedRequestParams,
    TextContent,
)
from starlette.middleware.base import BaseHTTPMiddleware


READ_ACTIONS = {
    "search",
    "list",
    "read",
    "similar",
    "relationships",
    "status",
    "reconstruct",
    "focus_history",
    "describe_project",
    "search_memory",
    "list_memory",
    "read_memory",
    "similar_memory",
}
WRITE_ACTIONS = {
    "write",
    "write_memory",
    "remember",
    "put_page",
    "capture",
}
MUTATION_ACTIONS = {"update", "delete", "report", "resolve"}
_request_arm: ContextVar[str | None] = ContextVar("request_arm", default=None)
_request_project: ContextVar[str | None] = ContextVar("request_project", default=None)


def _arm() -> str:
    value = (
        _request_arm.get()
        or os.environ.get("MEMORYHUB_BENCH_ARM", "normal")
    ).strip().lower()
    if value not in {"capture", "replay", "normal"}:
        raise RuntimeError(
            "MEMORYHUB_BENCH_ARM must be capture, replay, or normal"
        )
    return value


def _audit(event: dict) -> None:
    path_value = os.environ.get("LLM_MEMORY_BENCH_PROXY_AUDIT", "").strip()
    if not path_value:
        return
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "arm": _arm(),
        **event,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _is_read(name: str, arguments: dict) -> bool:
    action = arguments.get("action") if name == "memory" else name
    return str(action or "").lower() in READ_ACTIONS


def _is_write(name: str, arguments: dict) -> bool:
    action = arguments.get("action") if name == "memory" else name
    return str(action or "").lower() in WRITE_ACTIONS


def _isolation_arguments(name: str, arguments: dict | None) -> dict:
    """Force supported MemoryHub calls into the benchmark project."""

    project_id = (
        _request_project.get()
        or os.environ.get("MEMORYHUB_PROJECT_ID", "")
    ).strip()
    args = dict(arguments or {})
    if not project_id:
        return args

    if name == "memory":
        action = str(args.get("action", "")).lower()
        if action in READ_ACTIONS | WRITE_ACTIONS | MUTATION_ACTIONS:
            args["project_id"] = project_id
            args["scope"] = "project"
        return args

    if name in READ_ACTIONS | WRITE_ACTIONS | MUTATION_ACTIONS:
        args["project_id"] = project_id
        if name in {"search_memory", "list_memory", "write_memory"}:
            args["scope"] = "project"
    return args


def _empty_result(name: str, arguments: dict) -> CallToolResult:
    """Return a valid, non-error empty response for a suppressed read."""

    action = arguments.get("action") if name == "memory" else name
    action = str(action or "").lower()
    if action in {"read", "read_memory"}:
        payload = {"error": "Memory not found"}
    elif action in {"status", "describe_project"}:
        payload = {"status": "ok", "memories": [], "total": 0}
    else:
        payload = {"memories": [], "results": [], "total": 0}
    _audit(
        {
            "tool": name,
            "action": action,
            "decision": "suppressed_read",
            "argument_keys": sorted(arguments),
        }
    )
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(payload))],
        is_error=False,
    )


def _argument_fingerprint(arguments: dict) -> str:
    return hashlib.sha256(
        json.dumps(arguments, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]


async def _remote_call(call):
    url = os.environ["MEMORYHUB_URL"]
    api_key = os.environ.get("MEMORYHUB_API_KEY", "")
    bearer_token = os.environ.get("MEMORYHUB_AUTH_TOKEN", "")
    remote = Client(url, auth=bearer_token or None)

    async with remote:
        if api_key:
            await remote.call_tool("register_session", {"api_key": api_key}, raise_on_error=True)
        return await call(remote)


def _text_content(value) -> TextContent:
    """Convert an upstream content block to this server's MCP model."""

    if isinstance(value, TextContent):
        return value
    if isinstance(value, dict):
        text = value.get("text")
        if text is None:
            text = json.dumps(value, default=str)
    else:
        text = getattr(value, "text", None)
        if text is None:
            text = str(value)
    return TextContent(type="text", text=str(text))


def _normalize_remote_result(result) -> CallToolResult:
    """Rebuild FastMCP's result as the low-level server's MCP result type."""

    content = getattr(result, "content", None)
    if content is None and isinstance(result, dict):
        content = result.get("content", [])
    if not isinstance(content, list):
        content = [content] if content is not None else []
    structured = getattr(result, "structuredContent", None)
    if structured is None:
        structured = getattr(result, "structured_content", None)
    if structured is None and isinstance(result, dict):
        structured = result.get("structuredContent", result.get("structured_content"))
    is_error = getattr(result, "isError", None)
    if is_error is None:
        is_error = getattr(result, "is_error", None)
    if is_error is None and isinstance(result, dict):
        is_error = result.get("isError", result.get("is_error", False))
    return CallToolResult(
        content=[_text_content(item) for item in content],
        structuredContent=structured,
        isError=bool(is_error),
    )


async def _forward_tool_call(name: str, arguments: dict) -> CallToolResult:
    """Forward a call and preserve upstream errors in audit and MCP output."""

    try:
        result = await _remote_call(
            lambda remote: remote.call_tool(name, arguments, raise_on_error=False)
        )
        normalized = _normalize_remote_result(result)
        _audit(
            {
                "tool": name,
                "decision": "upstream_result",
                "is_error": normalized.is_error,
                "content_preview": [
                    item.text[:500] for item in normalized.content if item.text
                ],
            }
        )
        return normalized
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        _audit(
            {
                "tool": name,
                "decision": "upstream_error",
                "error_type": type(exc).__name__,
                "error": str(exc)[:2000],
            }
        )
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps({"error": detail}))],
            isError=True,
        )


async def _stdio() -> None:
    async def list_tools(_context, _params) -> ListToolsResult:
        tools = await _remote_call(lambda remote: remote.list_tools())
        _audit({"decision": "tools_list", "tool_count": len(tools)})
        return ListToolsResult(tools=tools)

    async def call_tool(_context, params) -> CallToolResult:
        name = params.name
        original = dict(params.arguments or {})
        if _arm() == "capture" and _is_read(name, original):
            return _empty_result(name, original)

        if _arm() == "capture" and not _is_write(name, original):
            _audit({
                "tool": name,
                "decision": "rejected_non_write_capture_call",
                "argument_keys": sorted(original),
            })
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps({"error": "Capture mode is write-only"}))],
                is_error=False,
            )

        arguments = _isolation_arguments(name, original)
        _audit({
            "tool": name,
            "decision": "forwarded",
            "argument_keys": sorted(arguments),
            "argument_fingerprint": _argument_fingerprint(arguments),
        })
        return await _forward_tool_call(name, arguments)

    server = Server("llm-memory-bench-memoryhub-proxy")
    server.add_request_handler("tools/list", PaginatedRequestParams, list_tools)
    server.add_request_handler("tools/call", CallToolRequestParams, call_tool)

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


class _ArmMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        arm = request.headers.get("x-llm-memory-bench-arm")
        project = request.headers.get("x-llm-memory-bench-project-id")
        arm_token = _request_arm.set(arm)
        project_token = _request_project.set(project)
        try:
            return await call_next(request)
        finally:
            _request_arm.reset(arm_token)
            _request_project.reset(project_token)


async def _http(host: str, port: int) -> None:
    import uvicorn

    async def list_tools(_context, _params) -> ListToolsResult:
        tools = await _remote_call(lambda remote: remote.list_tools())
        _audit({"decision": "tools_list", "tool_count": len(tools)})
        return ListToolsResult(tools=tools)

    async def call_tool(_context, params) -> CallToolResult:
        name = params.name
        original = dict(params.arguments or {})
        if _arm() == "capture" and _is_read(name, original):
            return _empty_result(name, original)
        if _arm() == "capture" and not _is_write(name, original):
            _audit({"tool": name, "decision": "rejected_non_write_capture_call"})
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps({"error": "Capture mode is write-only"}))],
                is_error=False,
            )
        arguments = _isolation_arguments(name, original)
        _audit({"tool": name, "decision": "forwarded", "argument_keys": sorted(arguments)})
        return await _forward_tool_call(name, arguments)

    server = Server("llm-memory-bench-memoryhub-proxy")
    server.add_request_handler("tools/list", PaginatedRequestParams, list_tools)
    server.add_request_handler("tools/call", CallToolRequestParams, call_tool)
    app = server.streamable_http_app(
        streamable_http_path="/mcp",
        json_response=True,
        stateless_http=True,
        host=host,
    )
    app.add_middleware(_ArmMiddleware)
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    await uvicorn.Server(config).serve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--http", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.http:
        asyncio.run(_http(args.host, args.port))
    else:
        asyncio.run(_stdio())

if __name__ == "__main__":
    main()

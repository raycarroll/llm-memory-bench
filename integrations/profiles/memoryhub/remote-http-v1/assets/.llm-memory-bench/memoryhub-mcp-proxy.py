"""Authenticate and proxy a remote MemoryHub MCP session over stdio.

Claude Code launches this process as its local MCP server. The process keeps
the API key in its environment, registers the remote MCP session itself, and
forwards the remote tool schemas/results without exposing the key to the
agent's prompt context.
"""

from __future__ import annotations

import asyncio
import os

from fastmcp import Client
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolRequestParams, ListToolsResult, PaginatedRequestParams


def _isolation_arguments(name: str, arguments: dict | None) -> dict:
    """Force benchmark MCP calls into the materialized project namespace."""

    project_id = os.environ.get("MEMORYHUB_PROJECT_ID", "").strip()
    args = dict(arguments or {})
    if not project_id:
        return args

    if name == "memory":
        action = args.get("action")
        project_actions = {
            "search", "list", "read", "similar", "relationships", "reconstruct",
            "focus_history", "describe_project", "write", "update", "delete",
            "set_focus", "relate", "report", "promote", "graduate",
        }
        if action in project_actions:
            args["project_id"] = project_id
        if action in {"search", "list", "reconstruct"}:
            args["scope"] = "project"
        elif action == "write":
            args["scope"] = "project"
        return args

    if name in {"search_memory", "list_memory", "reconstruct"}:
        args["project_id"] = project_id
        args["scope"] = "project"
    elif name in {
        "read_memory", "update_memory", "delete_memory", "similar_memory",
        "manage_graph", "manage_curation",
    }:
        args["project_id"] = project_id
    elif name == "write_memory":
        args["project_id"] = project_id
        args["scope"] = "project"
    elif name == "thread":
        action = args.get("action")
        options = dict(args.get("options") or {})
        if action in {"create", "list"}:
            args["scope"] = "project"
            options["scope_id"] = project_id
        args["options"] = options
    return args


async def run() -> None:
    url = os.environ["MEMORYHUB_URL"]
    api_key = os.environ.get("MEMORYHUB_API_KEY", "")
    bearer_token = os.environ.get("MEMORYHUB_AUTH_TOKEN", "")
    remote = Client(url, auth=bearer_token or None)

    async with remote:
        if api_key:
            await remote.call_tool(
                "register_session",
                {"api_key": api_key},
                raise_on_error=True,
            )

        server = Server("memoryhub-authenticated-proxy")

        async def list_tools(_context, _params) -> ListToolsResult:
            return ListToolsResult(tools=await remote.list_tools())

        async def call_tool(_context, params):
            return await remote.call_tool(
                params.name,
                _isolation_arguments(params.name, params.arguments),
                raise_on_error=False,
            )

        server.add_request_handler("tools/list", PaginatedRequestParams, list_tools)
        server.add_request_handler("tools/call", CallToolRequestParams, call_tool)

        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )


if __name__ == "__main__":
    asyncio.run(run())

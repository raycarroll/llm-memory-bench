"""Scenario-scoped MCP tools and mutation guards for paired demos."""

from __future__ import annotations

import asyncio
import os
from contextlib import AsyncExitStack
from typing import Any

import yaml

from .mcp import MCPToolSession, result_is_error, result_text
from .specs import DemoSideSpec, DemoSpec

MEMORYHUB_SEARCH_NAME = "memoryhub_search"


class DemoToolRuntime:
    """Expose only the tools configured for one demo side."""

    def __init__(
        self,
        *,
        side: str,
        spec: DemoSpec,
        memoryhub_api_key: str | None = None,
        timeout_seconds: float | None = None,
    ):
        if side not in spec.sides:
            raise ValueError(f"Unknown demo side: {side}")
        self.side = side
        self.spec = spec
        self.side_spec: DemoSideSpec = spec.sides[side]
        self.namespace = self.side_spec.namespace
        self.timeout_seconds = timeout_seconds or spec.limits.tool_timeout_seconds
        self.memoryhub_api_key = memoryhub_api_key
        self._stack: AsyncExitStack | None = None
        self.kubernetes: MCPToolSession | None = None
        self.memoryhub: MCPToolSession | None = None
        self._tools: dict[str, dict[str, Any]] = {}
        self._memoryhub_registered = False
        self.memory_search_completed = False

    async def __aenter__(self) -> "DemoToolRuntime":
        self._stack = AsyncExitStack()
        await self._stack.__aenter__()
        try:
            endpoint = os.getenv(self.side_spec.kubernetes_endpoint_env)
            if not endpoint:
                raise ValueError(
                    f"Set {self.side_spec.kubernetes_endpoint_env} for the {self.side} MCP endpoint"
                )
            self.kubernetes = await self._open_session(endpoint)
            discovered = await self.kubernetes.list_tools()
            by_name = {str(tool.name): self._tool_definition(tool) for tool in discovered}
            missing = set(self.side_spec.kubernetes_tools) - set(by_name)
            if missing:
                raise ValueError(
                    f"MCP endpoint for {self.side} is missing configured tools: "
                    + ", ".join(sorted(missing))
                )
            self._tools.update({name: by_name[name] for name in self.side_spec.kubernetes_tools})

            if self.side_spec.memoryhub_tools:
                await self._open_memoryhub()
            return self
        except BaseException as exc:
            await self._stack.__aexit__(type(exc), exc, exc.__traceback__)
            self._stack = None
            self.kubernetes = None
            self.memoryhub = None
            raise

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._stack is not None:
            await self._stack.__aexit__(exc_type, exc, tb)
        self._stack = None
        self.kubernetes = None
        self.memoryhub = None

    def tools_for_turn(self) -> list[dict[str, Any]]:
        """Hide memory-side mutation tools until the required search succeeds."""

        if self._requires_memory_search and not self.memory_search_completed:
            allowed = set(self.spec.read_only_tools) | {MEMORYHUB_SEARCH_NAME}
            return [tool for name, tool in self._tools.items() if name in allowed]
        return list(self._tools.values())

    @property
    def _requires_memory_search(self) -> bool:
        return (
            self.side == "memory"
            and self.spec.memory_search_required_before_change
            and bool(self.side_spec.memoryhub_tools)
        )

    async def call(self, name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        """Execute one configured tool after applying scenario guards."""

        if name == MEMORYHUB_SEARCH_NAME:
            return await self._search_memory(arguments)
        if name not in self._tools:
            return f"Blocked: {name!r} is not enabled for this demo side.", True
        if name in self.spec.change_making_tools:
            if self._requires_memory_search and not self.memory_search_completed:
                return "Blocked: search MemoryHub before making a Kubernetes change.", True
            allowed, reason = self._validate_change(name, arguments)
            if not allowed:
                return f"Blocked: {reason}", True
        if self.kubernetes is None:
            return "Kubernetes MCP is not connected.", True
        arguments = self._enforce_namespace(name, arguments)
        try:
            result = await asyncio.wait_for(
                self.kubernetes.call_tool(name, arguments), timeout=self.timeout_seconds
            )
        except Exception as exc:
            return f"Tool failed: {type(exc).__name__}: {exc}", True
        return result_text(result), result_is_error(result)

    async def _search_memory(self, arguments: dict[str, Any]) -> tuple[str, bool]:
        if self.memoryhub is None or not self._memoryhub_registered:
            return "MemoryHub search is not configured for this side.", True
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            return "MemoryHub search requires a non-empty query.", True
        try:
            result = await asyncio.wait_for(
                self.memoryhub.call_tool(
                    str(self.spec.memoryhub.get("action_tool", "memory")),
                    {
                        "action": "search",
                        "query": query.strip(),
                        "scope": "project",
                        "project_id": self.spec.memoryhub_project_id,
                        "options": {"max_results": self.spec.memoryhub.get("max_results", 5)},
                    },
                ),
                timeout=self.timeout_seconds,
            )
        except Exception as exc:
            return f"MemoryHub search failed: {type(exc).__name__}: {exc}", True
        if not result_is_error(result):
            self.memory_search_completed = True
        return result_text(result), result_is_error(result)

    def _validate_change(self, name: str, arguments: dict[str, Any]) -> tuple[bool, str]:
        if name != "resources_create_or_update":
            return False, f"change tool {name!r} has no demo-specific write guard"
        resource_text = arguments.get("resource")
        if not isinstance(resource_text, str) or not resource_text.strip():
            return False, "the apply manifest must be a non-empty string"
        try:
            manifest = yaml.safe_load(resource_text)
        except yaml.YAMLError as exc:
            return False, f"the apply manifest is invalid YAML: {exc}"
        if not isinstance(manifest, dict):
            return False, "the apply manifest must describe one Kubernetes resource"
        metadata = manifest.get("metadata")
        if not isinstance(metadata, dict):
            return False, "the resource manifest must include metadata"
        if metadata.get("namespace") != self.namespace:
            return False, f"the manifest namespace must be {self.namespace!r}"
        resource_id = f"{str(manifest.get('kind', '')).lower()}/{metadata.get('name', '')}"
        if resource_id not in set(self.spec.change_resources):
            return False, f"resource {resource_id!r} is not writable in this demo"
        if manifest.get("apiVersion") != "networking.k8s.io/v1":
            return False, "only networking.k8s.io/v1 NetworkPolicy resources can be changed"
        return True, ""

    def _enforce_namespace(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        properties = self._tools[name].get("parameters", {}).get("properties", {})
        names = {key for key in ("namespace", "namespace_name") if key in properties}
        arguments = dict(arguments)
        for key in names:
            if arguments.get(key) and arguments[key] != self.namespace:
                raise ValueError(f"this side is restricted to namespace {self.namespace!r}")
            arguments[key] = self.namespace
        return arguments

    async def _open_memoryhub(self) -> None:
        if not self.memoryhub_api_key:
            raise ValueError("MemoryHub API key is required for the memory side")
        endpoint_env = str(self.spec.memoryhub.get("endpoint_env", "MEMORYHUB_MCP_URL"))
        endpoint = os.getenv(endpoint_env)
        if not endpoint:
            raise ValueError(f"Set {endpoint_env} for MemoryHub MCP")
        self.memoryhub = await self._open_session(endpoint)
        tools = {str(tool.name) for tool in await self.memoryhub.list_tools()}
        action_tool = str(self.spec.memoryhub.get("action_tool", "memory"))
        if "register_session" not in tools or action_tool not in tools:
            raise ValueError("MemoryHub MCP must expose register_session and the configured action tool")
        registered = await asyncio.wait_for(
            self.memoryhub.call_tool("register_session", {"api_key": self.memoryhub_api_key}),
            timeout=self.timeout_seconds,
        )
        if result_is_error(registered):
            raise RuntimeError("MemoryHub session registration failed: " + result_text(registered))
        self._memoryhub_registered = True
        self._tools[MEMORYHUB_SEARCH_NAME] = {
            "name": MEMORYHUB_SEARCH_NAME,
            "description": "Search the configured MemoryHub project for relevant operational guidance.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
        }

    async def _open_session(self, endpoint: str) -> MCPToolSession:
        if self._stack is None:
            raise RuntimeError("runtime is not entering")
        return await self._stack.enter_async_context(
            MCPToolSession(
                endpoint,
                timeout_seconds=self.timeout_seconds,
                sse_read_timeout_seconds=self.timeout_seconds,
            )
        )

    @staticmethod
    def _tool_definition(tool: Any) -> dict[str, Any]:
        schema = getattr(tool, "inputSchema", None)
        if schema is None:
            schema = getattr(tool, "input_schema", None)
        return {
            "name": str(tool.name),
            "description": str(getattr(tool, "description", "")),
            "parameters": schema or {"type": "object", "properties": {}},
        }

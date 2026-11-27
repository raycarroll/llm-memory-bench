"""Small, independently managed Streamable HTTP MCP client sessions."""

from __future__ import annotations

import json
from contextlib import AsyncExitStack
from typing import Any


def result_text(result: Any) -> str:
    """Return stable readable text from an MCP result."""

    structured = getattr(result, "structuredContent", None)
    if structured is None:
        structured = getattr(result, "structured_content", None)
    if structured is not None:
        return json.dumps(structured, ensure_ascii=False, default=str)
    parts = [
        block.text
        for block in (getattr(result, "content", []) or [])
        if getattr(block, "text", None) is not None
    ]
    return "\n".join(parts) if parts else "(MCP tool returned no readable content)"


def result_is_error(result: Any) -> bool:
    """Support both Python MCP field naming conventions."""

    return bool(getattr(result, "isError", getattr(result, "is_error", False)))


class MCPToolSession:
    """One MCP client session for one side-specific endpoint."""

    def __init__(
        self,
        endpoint: str,
        *,
        timeout_seconds: float = 30.0,
        sse_read_timeout_seconds: float = 300.0,
    ):
        if not endpoint:
            raise ValueError("MCP endpoint must not be empty")
        if timeout_seconds <= 0 or sse_read_timeout_seconds <= 0:
            raise ValueError("MCP timeouts must be positive")
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds
        self.sse_read_timeout_seconds = sse_read_timeout_seconds
        self._stack: AsyncExitStack | None = None
        self._session: Any = None

    async def __aenter__(self) -> "MCPToolSession":
        try:
            from mcp import ClientSession
            from mcp.client.streamable_http import streamablehttp_client
        except ImportError as exc:
            raise RuntimeError(
                "The MCP client dependency is required for demo tool sessions."
            ) from exc

        self._stack = AsyncExitStack()
        await self._stack.__aenter__()
        try:
            transport = await self._stack.enter_async_context(
                streamablehttp_client(
                    self.endpoint,
                    timeout=self.timeout_seconds,
                    sse_read_timeout=self.sse_read_timeout_seconds,
                )
            )
            read_stream, write_stream, _ = transport
            self._session = await self._stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await self._session.initialize()
            return self
        except BaseException as exc:
            await self._stack.__aexit__(type(exc), exc, exc.__traceback__)
            self._stack = None
            self._session = None
            raise

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._stack is not None:
            await self._stack.__aexit__(exc_type, exc, tb)
        self._stack = None
        self._session = None

    async def list_tools(self) -> list[Any]:
        return list((await self._require_session().list_tools()).tools)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        return await self._require_session().call_tool(name, arguments)

    def _require_session(self) -> Any:
        if self._session is None:
            raise RuntimeError("MCP session is not connected")
        return self._session

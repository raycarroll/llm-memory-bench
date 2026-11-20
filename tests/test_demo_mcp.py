import asyncio
from types import SimpleNamespace

import pytest

from llm_memory_bench.demo.mcp import MCPToolSession, result_is_error, result_text


def test_result_text_prefers_structured_content():
    result = SimpleNamespace(structuredContent={"ok": True}, content=[])

    assert result_text(result) == '{"ok": true}'


def test_result_text_joins_text_blocks():
    result = SimpleNamespace(
        content=[SimpleNamespace(text="first"), SimpleNamespace(text="second")]
    )

    assert result_text(result) == "first\nsecond"


def test_result_text_handles_empty_result():
    assert result_text(SimpleNamespace(content=[])) == "(MCP tool returned no readable content)"


def test_result_is_error_supports_mcp_field_names():
    assert result_is_error(SimpleNamespace(isError=True)) is True
    assert result_is_error(SimpleNamespace(is_error=True)) is True
    assert result_is_error(SimpleNamespace(isError=False)) is False


def test_session_requires_positive_timeouts():
    with pytest.raises(ValueError, match="positive"):
        MCPToolSession("http://example/mcp", timeout_seconds=0)


def test_session_rejects_operations_before_connection():
    session = MCPToolSession("http://example/mcp")

    with pytest.raises(RuntimeError, match="not connected"):
        asyncio.run(session.list_tools())


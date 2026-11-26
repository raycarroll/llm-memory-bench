from __future__ import annotations

import importlib.util
import asyncio
import json
from pathlib import Path


PROXY = (
    Path(__file__).parents[1]
    / "integrations/profiles/memoryhub/remote-http-v1/assets/.llm-memory-bench/memoryhub-mcp-proxy.py"
)


def _load_proxy():
    spec = importlib.util.spec_from_file_location("memoryhub_bench_proxy", PROXY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_capture_policy_suppresses_reads_but_not_writes(monkeypatch) -> None:
    proxy = _load_proxy()
    monkeypatch.setenv("MEMORYHUB_BENCH_ARM", "capture")

    assert proxy._is_read("memory", {"action": "search"})
    assert not proxy._is_read("memory", {"action": "write"})
    assert proxy._is_write("memory", {"action": "write"})

    result = proxy._empty_result("memory", {"action": "search", "query": "x"})
    assert result.is_error is False
    assert json.loads(result.content[0].text) == {
        "memories": [],
        "results": [],
        "total": 0,
    }


def test_proxy_forces_project_scope_for_reads_and_writes(monkeypatch) -> None:
    proxy = _load_proxy()
    monkeypatch.setenv("MEMORYHUB_PROJECT_ID", "project-123")

    read_args = proxy._isolation_arguments("memory", {"action": "search", "query": "x"})
    write_args = proxy._isolation_arguments("memory", {"action": "write", "content": "x"})

    assert read_args["project_id"] == "project-123"
    assert read_args["scope"] == "project"
    assert write_args["project_id"] == "project-123"
    assert write_args["scope"] == "project"


def test_proxy_normalizes_upstream_result_and_audits_errors(monkeypatch, tmp_path) -> None:
    proxy = _load_proxy()
    audit = tmp_path / "audit.jsonl"
    monkeypatch.setenv("LLM_MEMORY_BENCH_PROXY_AUDIT", str(audit))

    class ForeignResult:
        content = [{"type": "text", "text": "backend rejected content_type"}]
        is_error = True

    async def fake_remote_call(_call):
        return ForeignResult()

    monkeypatch.setattr(proxy, "_remote_call", fake_remote_call)
    result = asyncio.run(
        proxy._forward_tool_call("memory", {"action": "write"})
    )

    assert result.is_error is True
    assert result.content[0].text == "backend rejected content_type"
    events = [json.loads(line) for line in audit.read_text().splitlines()]
    assert events[-1]["decision"] == "upstream_result"
    assert events[-1]["is_error"] is True

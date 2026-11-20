import asyncio

from llm_memory_bench.demo import DemoAgent, load_demo_spec
from llm_memory_bench.providers.base import ProviderResponse, ToolCall


SPEC = load_demo_spec("scenarios/openshift/network-policy.yaml")


class FakeProvider:
    model = "fake-model"

    def __init__(self):
        self.calls = 0

    async def generate(
        self,
        messages,
        tools,
        system,
        tool_choice=None,
        disable_parallel_tool_calls=False,
    ):
        self.calls += 1
        return ProviderResponse(text="done", input_tokens=2, output_tokens=1)

    def build_tool_result_messages(self, response, format_tool_result=None):
        return []


def test_provider_tool_schema_uses_input_schema():
    provider = FakeProvider()
    agent = DemoAgent(side="baseline", provider=provider, spec=SPEC)

    assert agent._provider_tool({"name": "pods_get", "parameters": {"type": "object"}}) == {
        "name": "pods_get",
        "description": "",
        "input_schema": {"type": "object"},
    }


def test_agent_reports_provider_error_without_affecting_other_state(monkeypatch):
    provider = FakeProvider()
    agent = DemoAgent(side="baseline", provider=provider, spec=SPEC)

    async def fail_runtime(*args, **kwargs):
        raise RuntimeError("not connected")

    monkeypatch.setattr("llm_memory_bench.demo.agent.DemoToolRuntime.__aenter__", fail_runtime)
    result = asyncio.run(agent.run())

    assert result.status == "failed"
    assert "not connected" in result.error


def test_call_key_is_stable():
    key_a = DemoAgent._call_key(ToolCall(name="pods_get", arguments={"namespace": "x"}))
    key_b = DemoAgent._call_key(ToolCall(name="pods_get", arguments={"namespace": "x"}))

    assert key_a == key_b

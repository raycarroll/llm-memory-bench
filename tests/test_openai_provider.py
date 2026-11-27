import asyncio
import os
from types import SimpleNamespace

from llm_memory_bench.providers.anthropic import AnthropicProvider
from llm_memory_bench.providers.openai import OpenAIProvider


def test_openai_provider_uses_standard_environment_variable(monkeypatch):
    captured = {}

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("openai.AsyncOpenAI", FakeAsyncOpenAI)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    OpenAIProvider(model="test-model")

    assert captured["api_key"] == "test-key"


def test_openai_provider_accepts_custom_environment_variable(monkeypatch):
    captured = {}

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("openai.AsyncOpenAI", FakeAsyncOpenAI)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("MY_OPENAI_KEY", "test-key")

    OpenAIProvider(model="test-model", api_key_env="MY_OPENAI_KEY")

    assert captured["api_key"] == "test-key"


def test_anthropic_provider_scopes_developer_context_per_request(monkeypatch):
    captured = {}

    class FakeClient:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                content=[],
                usage=SimpleNamespace(input_tokens=2, output_tokens=1),
                model="test-model",
            )

        @property
        def messages(self):
            return SimpleNamespace(create=self.create)

    client = FakeClient()
    monkeypatch.setattr("anthropic.AsyncAnthropic", lambda **kwargs: client)

    provider = AnthropicProvider(model="test-model")
    asyncio.run(
        provider.generate(
            messages=[
                {"role": "developer", "content": "developer context"},
                {"role": "user", "content": "task"},
            ],
            tools=[
                {
                    "name": "memoryhub_search",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
            system="system context",
            tool_choice="memoryhub_search",
            disable_parallel_tool_calls=True,
        )
    )

    assert captured["system"] == "system context\n\ndeveloper context"
    assert captured["messages"] == [{"role": "user", "content": "task"}]
    assert captured["tool_choice"] == {
        "type": "tool",
        "name": "memoryhub_search",
        "disable_parallel_tool_use": True,
    }

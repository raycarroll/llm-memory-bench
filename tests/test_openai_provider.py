import os

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

from llm_memory_bench.demo.driver import normalize_provider_model, resolve_memoryhub_key


def test_openai_catalog_model_prefix_is_normalized():
    assert normalize_provider_model("openai", "openai/gpt-4.1") == "gpt-4.1"
    assert normalize_provider_model("anthropic", "anthropic/claude-sonnet-4") == "anthropic/claude-sonnet-4"


def test_demo_defaults_to_memoryhub_api_key(monkeypatch):
    monkeypatch.delenv("CUSTOM_MEMORY_KEY", raising=False)
    monkeypatch.setenv("MEMORYHUB_API_KEY", "configured")
    assert resolve_memoryhub_key() == "configured"

    monkeypatch.setenv("CUSTOM_MEMORY_KEY", "custom")
    assert resolve_memoryhub_key("CUSTOM_MEMORY_KEY") == "custom"

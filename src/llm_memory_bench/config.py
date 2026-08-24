from __future__ import annotations

from pydantic import BaseModel


class RunConfig(BaseModel):
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    system: str = "simple"
    judge_provider: str = "anthropic"
    judge_model: str = "claude-sonnet-4-20250514"
    max_conversations: int | None = None
    api_key_env: str | None = None


class ValueRunConfig(RunConfig):
    injection_mode: str = "prompt"  # "prompt" | "tool" | "both"
    max_turns: int = 10
    user_simulator_provider: str = "anthropic"
    user_simulator_model: str = "claude-sonnet-4-20250514"
    max_scenarios: int | None = None


def get_provider(config: RunConfig):
    from .providers.anthropic import AnthropicProvider
    from .providers.litellm import LiteLLMProvider
    from .providers.openai import OpenAIProvider
    from .providers.vertex import VertexProvider

    kwargs = {}
    if config.api_key_env:
        kwargs["api_key_env"] = config.api_key_env

    providers = {
        "anthropic": AnthropicProvider,
        "openai": OpenAIProvider,
        "litellm": LiteLLMProvider,
        "vertex": VertexProvider,
    }

    if config.provider not in providers:
        raise ValueError(f"Unknown provider: {config.provider}. Available: {list(providers.keys())}")

    return providers[config.provider](model=config.model, **kwargs)

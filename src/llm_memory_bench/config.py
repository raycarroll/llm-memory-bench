from __future__ import annotations

from pydantic import BaseModel


class RunConfig(BaseModel):
    provider: str = "openai"
    model: str = "gpt-4o"
    system: str = "simple"
    scenario: str | None = None  # For BFCL systems
    batch: bool = False  # Batch extraction mode (AlpsBench-style)
    judge_provider: str = "openai"
    judge_model: str = "gpt-4o"
    max_conversations: int | None = None
    api_key_env: str | None = None
    max_total_tokens: int | None = None
    max_cost_usd: float | None = None
    save_responses: bool = False  # Save raw text responses and errors in run.json


class ValueRunConfig(RunConfig):
    injection_mode: str = "prompt"  # "prompt" | "tool" | "both"
    max_turns: int = 10
    judge_during_run: bool = True
    user_simulator_provider: str = "openai"
    user_simulator_model: str = "gpt-4o"
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

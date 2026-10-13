from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from ..config import RunConfig, ValueRunConfig


class RunSpec(BaseModel):
    """Serializable description of a benchmark run.

    This is deliberately UI- and backend-neutral.  The CLI, Streamlit UI,
    and a future EvalHub adapter can all submit the same specification.
    """

    kind: Literal["value", "extraction"] = "value"
    execution: Literal["api", "container"] = "api"
    dataset_path: str
    container_config_path: str | None = None
    goal_id: str | None = None
    provider: str = "openai"
    model: str = "gpt-4o"
    system: str = "simple"
    mode: str = "prompt"
    max_turns: int = Field(default=10, ge=1)
    max_scenarios: int | None = Field(default=None, ge=1)
    max_conversations: int | None = Field(default=None, ge=1)
    judge_during_run: bool = True
    judge_provider: str | None = None
    judge_model: str | None = None
    user_simulator_provider: str | None = None
    user_simulator_model: str | None = None
    api_key_env: str | None = None
    max_total_tokens: int | None = Field(default=None, ge=1)
    max_cost_usd: float | None = Field(default=None, gt=0)

    def value_config(self) -> ValueRunConfig:
        if self.kind != "value":
            raise ValueError("value_config() requires kind='value'")
        return ValueRunConfig(
            provider=self.provider,
            model=self.model,
            system=self.system,
            injection_mode=self.mode,
            max_turns=self.max_turns,
            max_scenarios=self.max_scenarios,
            judge_during_run=self.judge_during_run,
            judge_provider=self.judge_provider or self.provider,
            judge_model=self.judge_model or self.model,
            user_simulator_provider=self.user_simulator_provider or self.provider,
            user_simulator_model=self.user_simulator_model or self.model,
            api_key_env=self.api_key_env,
            max_total_tokens=self.max_total_tokens,
            max_cost_usd=self.max_cost_usd,
        )

    def container_value_config(self, container_model: str) -> ValueRunConfig:
        if self.kind != "value":
            raise ValueError("container_value_config() requires kind='value'")
        if self.execution != "container":
            raise ValueError("container_value_config() requires execution='container'")
        return ValueRunConfig(
            # The container hosts Claude Code; these provider settings apply to
            # the simulated user and the outcome judge instead.
            provider="anthropic",
            model=container_model,
            system="memoryhub",
            injection_mode="integration",
            max_turns=self.max_turns,
            max_scenarios=self.max_scenarios,
            judge_during_run=self.judge_during_run,
            judge_provider=self.judge_provider or "openai",
            judge_model=self.judge_model or "gpt-4o",
            user_simulator_provider=self.user_simulator_provider or "openai",
            user_simulator_model=self.user_simulator_model or "gpt-4o",
            api_key_env=self.api_key_env,
            max_total_tokens=self.max_total_tokens,
            max_cost_usd=self.max_cost_usd,
        )

    def extraction_config(self) -> RunConfig:
        if self.kind != "extraction":
            raise ValueError("extraction_config() requires kind='extraction'")
        return RunConfig(
            provider=self.provider,
            model=self.model,
            system=self.system,
            max_conversations=self.max_conversations,
            api_key_env=self.api_key_env,
            max_total_tokens=self.max_total_tokens,
            max_cost_usd=self.max_cost_usd,
        )

    def dataset(self) -> Path:
        return Path(self.dataset_path)

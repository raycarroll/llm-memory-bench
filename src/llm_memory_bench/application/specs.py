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

    kind: Literal["value", "tau2", "extraction", "demo"] = "value"
    execution: Literal["api", "container"] = "api"
    dataset_path: str
    container_config_path: str | None = None
    demo_config_path: str | None = None
    goal_id: str | None = None
    provider: str = "openai"
    model: str = "gpt-4o"
    system: str = "simple"
    mode: str = "prompt"
    memory_setup_mode: Literal["seeded", "capture_first"] = "seeded"
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
    tau2_domain: Literal["airline", "retail", "telecom"] | None = None
    tau2_task_ids: list[str] = Field(default_factory=list)
    tau2_memory_plan: str | None = None
    tau2_agent_model: str = "openai/gpt-4.1"
    tau2_user_model: str = "openai/gpt-4.1"
    tau2_memory_system: str = "simple"
    tau2_seed: int = 300
    tau2_max_steps: int = Field(default=20, ge=1)
    tau2_memoryhub_live: bool = False
    tau2_memoryhub_cleanup: bool = True

    def value_config(self) -> ValueRunConfig:
        if self.kind != "value":
            raise ValueError("value_config() requires kind='value'")
        return ValueRunConfig(
            provider=self.provider,
            model=self.model,
            system=self.system,
            injection_mode=self.mode,
            memory_setup_mode=self.memory_setup_mode,
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
            memory_setup_mode=self.memory_setup_mode,
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


class EvaluationSpec(BaseModel):
    """Serializable configuration for evaluating saved benchmark results."""

    result_paths: list[str] = Field(min_length=1)
    dataset_path: str | None = None
    matcher: Literal["hybrid", "embedding", "llm", "exact", "alpsbench"] = "hybrid"
    judge_provider: Literal["openai", "anthropic", "litellm", "vertex"] = "openai"
    judge_model: str = "gpt-4o"
    api_key_env: str | None = None

    def dataset(self) -> Path | None:
        return Path(self.dataset_path) if self.dataset_path else None

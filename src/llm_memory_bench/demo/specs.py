"""Validated, scenario-neutral contracts for paired operational demos."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class DemoLimits(BaseModel):
    """Execution limits shared by both sides of a demo."""

    model_config = ConfigDict(extra="forbid")

    max_turns: int = Field(default=30, ge=1)
    max_tool_calls: int = Field(default=36, ge=1)
    timeout_seconds: float = Field(default=300.0, gt=0)
    tool_timeout_seconds: float = Field(default=45.0, gt=0)


class DemoSideSpec(BaseModel):
    """Tool and namespace boundary for one side of a paired demo."""

    model_config = ConfigDict(extra="forbid")

    namespace: str = Field(min_length=1)
    kubernetes_endpoint_env: str = Field(min_length=1)
    kubernetes_tools: list[str] = Field(min_length=1)
    memoryhub_tools: list[str] = Field(default_factory=list)

    @field_validator("kubernetes_tools", "memoryhub_tools")
    @classmethod
    def unique_tools(cls, tools: list[str]) -> list[str]:
        if len(tools) != len(set(tools)):
            raise ValueError("tool lists must not contain duplicates")
        if any(not tool.strip() for tool in tools):
            raise ValueError("tool names must not be empty")
        return tools


class DemoSpec(BaseModel):
    """Complete validated description of a paired operational demo."""

    model_config = ConfigDict(extra="forbid")

    demo_id: str = Field(min_length=1)
    demo_version: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    system_prompt: str = ""
    memoryhub_project_id: str | None = None
    memoryhub: dict[str, Any] = Field(default_factory=dict)
    sides: dict[str, DemoSideSpec]
    read_only_tools: list[str] = Field(min_length=1)
    change_making_tools: list[str] = Field(min_length=1)
    change_resources: list[str] = Field(min_length=1)
    limits: DemoLimits = Field(default_factory=DemoLimits)
    fixture: dict[str, Any] = Field(default_factory=dict)
    verification: dict[str, Any] = Field(default_factory=dict)
    memory_search_required_before_change: bool = False

    @field_validator("read_only_tools", "change_making_tools", "change_resources")
    @classmethod
    def unique_nonempty_values(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("configured values must not contain duplicates")
        if any(not value.strip() for value in values):
            raise ValueError("configured values must not be empty")
        return values

    @model_validator(mode="after")
    def validate_boundaries(self) -> "DemoSpec":
        if set(self.sides) != {"memory", "baseline"}:
            raise ValueError("a paired demo must define exactly memory and baseline sides")

        read_only = set(self.read_only_tools)
        change_making = set(self.change_making_tools)
        overlap = read_only & change_making
        if overlap:
            raise ValueError(
                "tools cannot be both read-only and change-making: "
                + ", ".join(sorted(overlap))
            )

        configured = read_only | change_making
        for side_name, side in self.sides.items():
            unknown = set(side.kubernetes_tools) - configured
            if unknown:
                raise ValueError(
                    f"{side_name} config contains unclassified Kubernetes tools: "
                    + ", ".join(sorted(unknown))
                )

        if self.sides["memory"].memoryhub_tools and not self.memoryhub_project_id:
            raise ValueError(
                "memoryhub_project_id is required when the memory side has MemoryHub tools"
            )
        if self.sides["baseline"].memoryhub_tools:
            raise ValueError("the baseline side must not have MemoryHub tools")
        return self

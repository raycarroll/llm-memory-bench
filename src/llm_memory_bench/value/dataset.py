from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class MemoryEntry(BaseModel):
    fact: str
    category: str = "general"


class Outcome(BaseModel):
    """A specific result the goal is intended to produce."""

    id: str
    statement: str


class ValidatorSpec(BaseModel):
    """Configuration for a mechanism that validates one or more outcomes."""

    type: str = "llm_judge"
    outcomes: list[str] = Field(default_factory=list)
    config: dict[str, Any] = Field(default_factory=dict)


class ValueScenario(BaseModel):
    id: str
    memory_bank: list[MemoryEntry]
    task_prompt: str
    persona_prompt: str
    outcomes: list[Outcome]
    validators: list[ValidatorSpec] = Field(default_factory=lambda: [ValidatorSpec()])
    noise_memories: list[MemoryEntry] = Field(default_factory=list)

    def effective_outcomes(self) -> list[Outcome]:
        return self.outcomes

    def effective_validators(self) -> list[ValidatorSpec]:
        return self.validators


class ValueDataset(BaseModel):
    scenarios: list[ValueScenario]

    @property
    def total_scenarios(self) -> int:
        return len(self.scenarios)

    @property
    def total_memories(self) -> int:
        return sum(len(s.memory_bank) for s in self.scenarios)

    def summary(self) -> dict:
        categories: dict[str, int] = {}
        for s in self.scenarios:
            for m in s.memory_bank:
                categories[m.category] = categories.get(m.category, 0) + 1

        noise_count = sum(1 for s in self.scenarios if s.noise_memories)

        return {
            "scenarios": self.total_scenarios,
            "total_memories": self.total_memories,
            "avg_memories_per_scenario": round(
                self.total_memories / self.total_scenarios, 1
            )
            if self.total_scenarios
            else 0,
            "memory_categories": categories,
            "scenarios_with_noise": noise_count,
        }


def load_value_dataset(path: Path) -> ValueDataset:
    with open(path) as f:
        data = yaml.safe_load(f)
    return ValueDataset.model_validate(data)

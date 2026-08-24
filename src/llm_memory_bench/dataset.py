from __future__ import annotations

from enum import Enum
from pathlib import Path

import yaml
from pydantic import BaseModel


class FactType(str, Enum):
    DIRECT = "direct"
    INDIRECT = "indirect"


class ExpectedFact(BaseModel):
    fact: str
    type: FactType = FactType.DIRECT
    source_id: str | None = None


class GroundTruth(BaseModel):
    should_store: list[ExpectedFact] = []
    should_not_store: list[str] = []


class Turn(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    ground_truth: GroundTruth = GroundTruth()

    @property
    def is_noise(self) -> bool:
        return len(self.ground_truth.should_store) == 0


class Conversation(BaseModel):
    id: str
    source: str = "custom"
    turns: list[Turn]

    @property
    def noise_turn_count(self) -> int:
        return sum(1 for t in self.turns if t.is_noise)

    @property
    def fact_count(self) -> int:
        return sum(len(t.ground_truth.should_store) for t in self.turns)


class Dataset(BaseModel):
    conversations: list[Conversation]

    @property
    def total_turns(self) -> int:
        return sum(len(c.turns) for c in self.conversations)

    @property
    def total_facts(self) -> int:
        return sum(c.fact_count for c in self.conversations)

    def summary(self) -> dict:
        fact_types: dict[str, int] = {}
        for conv in self.conversations:
            for turn in conv.turns:
                for fact in turn.ground_truth.should_store:
                    fact_types[fact.type.value] = fact_types.get(fact.type.value, 0) + 1

        return {
            "conversations": len(self.conversations),
            "total_turns": self.total_turns,
            "total_facts": self.total_facts,
            "noise_turns": sum(c.noise_turn_count for c in self.conversations),
            "fact_types": fact_types,
        }


def load_dataset(path: Path) -> Dataset:
    with open(path) as f:
        data = yaml.safe_load(f)
    return Dataset.model_validate(data)

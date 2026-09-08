from __future__ import annotations

from enum import Enum
from pathlib import Path

import yaml
from pydantic import BaseModel, field_validator, model_validator


class FactType(str, Enum):
    DIRECT = "direct"
    INDIRECT = "indirect"


class GroundTruthType(str, Enum):
    PER_TURN = "per_turn"
    CUMULATIVE = "cumulative"


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
    cumulative_ground_truth: GroundTruth | None = None

    @property
    def noise_turn_count(self) -> int:
        return sum(1 for t in self.turns if t.is_noise)

    @property
    def fact_count(self) -> int:
        """Return fact count based on whether cumulative or per-turn."""
        if self.cumulative_ground_truth:
            return len(self.cumulative_ground_truth.should_store)
        return sum(len(t.ground_truth.should_store) for t in self.turns)


class Dataset(BaseModel):
    conversations: list[Conversation]
    ground_truth_type: GroundTruthType = GroundTruthType.PER_TURN
    cumulative_ground_truth: GroundTruth | None = None  # Dataset-level cumulative ground truth

    @model_validator(mode="after")
    def validate_ground_truth_consistency(self):
        """Ensure ground truth type matches conversation structure."""

        if self.ground_truth_type == GroundTruthType.CUMULATIVE:
            # Two modes for cumulative:
            # 1. Dataset-level: single cumulative_ground_truth for all conversations
            # 2. Per-conversation: each conversation has its own cumulative_ground_truth

            has_dataset_level = self.cumulative_ground_truth is not None
            has_conv_level = any(c.cumulative_ground_truth is not None for c in self.conversations)

            if not has_dataset_level and not has_conv_level:
                raise ValueError(
                    "Dataset type is 'cumulative' but no cumulative_ground_truth found. "
                    "Add either dataset.cumulative_ground_truth or conversation.cumulative_ground_truth."
                )

            if has_dataset_level and has_conv_level:
                raise ValueError(
                    "Dataset has both dataset-level and conversation-level cumulative_ground_truth. "
                    "Use only one approach."
                )

            # Per-turn ground truth should be empty in cumulative datasets
            for conv in self.conversations:
                for turn_idx, turn in enumerate(conv.turns):
                    if len(turn.ground_truth.should_store) > 0:
                        raise ValueError(
                            f"Dataset type is 'cumulative' but conversation '{conv.id}' turn {turn_idx} "
                            f"has per-turn ground_truth. Use cumulative_ground_truth instead."
                        )

        elif self.ground_truth_type == GroundTruthType.PER_TURN:
            # Per-turn datasets should NOT have cumulative ground truth
            if self.cumulative_ground_truth is not None:
                raise ValueError(
                    "Dataset type is 'per_turn' but has dataset-level cumulative_ground_truth."
                )
            for conv in self.conversations:
                if conv.cumulative_ground_truth is not None:
                    raise ValueError(
                        f"Dataset type is 'per_turn' but conversation '{conv.id}' "
                        f"has cumulative_ground_truth. Remove it or change dataset type to 'cumulative'."
                    )

        return self

    @property
    def total_turns(self) -> int:
        return sum(len(c.turns) for c in self.conversations)

    @property
    def total_facts(self) -> int:
        # Dataset-level cumulative ground truth
        if self.cumulative_ground_truth:
            return len(self.cumulative_ground_truth.should_store)
        # Per-conversation cumulative or per-turn
        return sum(c.fact_count for c in self.conversations)

    def summary(self) -> dict:
        fact_types: dict[str, int] = {}

        if self.ground_truth_type == GroundTruthType.PER_TURN:
            # Count facts from per-turn ground truth
            for conv in self.conversations:
                for turn in conv.turns:
                    for fact in turn.ground_truth.should_store:
                        fact_types[fact.type.value] = fact_types.get(fact.type.value, 0) + 1
        elif self.cumulative_ground_truth:
            # Count facts from dataset-level cumulative ground truth
            for fact in self.cumulative_ground_truth.should_store:
                fact_types[fact.type.value] = fact_types.get(fact.type.value, 0) + 1
        else:
            # Count facts from per-conversation cumulative ground truth
            for conv in self.conversations:
                if conv.cumulative_ground_truth:
                    for fact in conv.cumulative_ground_truth.should_store:
                        fact_types[fact.type.value] = fact_types.get(fact.type.value, 0) + 1

        return {
            "conversations": len(self.conversations),
            "total_turns": self.total_turns,
            "total_facts": self.total_facts,
            "noise_turns": sum(c.noise_turn_count for c in self.conversations),
            "fact_types": fact_types,
            "ground_truth_type": self.ground_truth_type.value,
        }


def load_dataset(path: Path) -> Dataset:
    with open(path) as f:
        data = yaml.safe_load(f)
    return Dataset.model_validate(data)

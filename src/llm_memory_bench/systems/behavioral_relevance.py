from __future__ import annotations

from pathlib import Path

import yaml

from ..providers.base import ToolCall
from .base import MemorySystem

PROMPTS_DIR = Path(__file__).parent.parent.parent.parent / "prompts" / "behavioral_relevance"


# Simple store_fact tool schema
STORE_FACT_TOOL = {
    "name": "store_fact",
    "description": (
        "Store a fact that changes how you should behave in future conversations. "
        "Apply the behavioral relevance test: 'If I forgot this, would my next response "
        "be less helpful or misaligned?' Only store if YES. "
        "Categories: preferences, constraints, domain context, corrections, patterns. "
        "Weight 0.6-1.0 based on behavioral impact (see system prompt for details)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "fact": {
                "type": "string",
                "description": "The fact to store. Must be clear and self-contained.",
            },
            "category": {
                "type": "string",
                "enum": ["preference", "constraint", "domain_context", "correction", "pattern"],
                "description": (
                    "Type of fact: preference (how user wants things), constraint (rules), "
                    "domain_context (background knowledge), correction (fixing mistakes), "
                    "pattern (repeated behaviors)."
                ),
            },
            "weight": {
                "type": "number",
                "description": (
                    "Impact on future behavior (0.6-1.0): "
                    "1.0 = critical constraint, 0.9 = strong preference, "
                    "0.8 = important context/correction, 0.7 = moderate preference/pattern, "
                    "0.6 = nice-to-know context."
                ),
                "minimum": 0.6,
                "maximum": 1.0,
            },
        },
        "required": ["fact", "category", "weight"],
    },
}


class BehavioralRelevanceSystem(MemorySystem):
    """Behavioral Relevance Filter - optimized for both high recall and precision.

    Core principle: Store facts that change how you should behave in future interactions.
    Target: >75% recall, >30% precision (vs current best 55%/22%).
    """

    name = "behavioral_relevance"
    description = (
        "Behavioral Relevance Filter with clear STORE/SKIP categories and examples. "
        "Optimized for both high recall (>75%) and precision (>30%)."
    )

    def __init__(self, prompt_version: str | None = None, **kwargs):
        """Initialize with optional prompt version.

        Args:
            prompt_version: Prompt version to use (e.g., "v1_2026-08").
                           Defaults to default_version from metadata.yaml.
            **kwargs: Ignored (for compatibility with systems that use scenario, etc.)

        Raises:
            FileNotFoundError: If prompt files not found
        """
        self.prompt_version = prompt_version or self._get_default_version()
        self._prompts = self._load_prompts(self.prompt_version)

    @classmethod
    def _get_default_version(cls) -> str:
        """Load default version from metadata.yaml."""
        metadata_path = PROMPTS_DIR / "metadata.yaml"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Prompt metadata not found: {metadata_path}. "
                "Run design-time prompt extraction first."
            )
        with open(metadata_path) as f:
            metadata = yaml.safe_load(f)
        return metadata["default_version"]

    @classmethod
    def _load_prompts(cls, version: str) -> dict:
        """Load prompts from versioned YAML file."""
        prompt_path = PROMPTS_DIR / f"{version}.yaml"
        if not prompt_path.exists():
            raise FileNotFoundError(
                f"Prompt file not found: {prompt_path}. "
                f"Available versions: {list(p.stem for p in PROMPTS_DIR.glob('v*.yaml'))}"
            )
        with open(prompt_path) as f:
            return yaml.safe_load(f)

    def system_prompt(self) -> str:
        return self._prompts["system_prompt"]

    def tool_definitions(self) -> list[dict]:
        return [STORE_FACT_TOOL]

    def extract_stored_fact(self, tool_call: ToolCall) -> str:
        """Extract the fact from a store_fact tool call."""
        return tool_call.arguments.get("fact", "")

    def format_tool_result(self, tool_call: ToolCall) -> dict:
        """Simulated success response."""
        category = tool_call.arguments.get("category", "unknown")
        weight = tool_call.arguments.get("weight", 0.7)
        return {
            "status": "success",
            "message": f"Stored {category} fact with weight {weight}",
        }

    def version_info(self) -> dict:
        """Include prompt version and source."""
        metadata_path = PROMPTS_DIR / "metadata.yaml"
        with open(metadata_path) as f:
            metadata = yaml.safe_load(f)

        pulled_date = self._prompts.get("pulled_date")
        return {
            "system": self.name,
            "prompt_version": self.prompt_version,
            "prompt_source": f"{metadata['source']['repo']} {metadata['source']['path']}",
            "prompt_pulled_date": str(pulled_date) if pulled_date else None,
        }

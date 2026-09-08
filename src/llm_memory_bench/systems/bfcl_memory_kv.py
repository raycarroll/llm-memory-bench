"""BFCL Memory system with Key-Value backend.

This implements the BFCL v4 memory category's KV backend with optional domain scenarios.
Scenarios (student, customer, finance, healthcare, notetaker) add persona flavor but don't
change how the memory system works.

Usage:
    # Without scenario (generic memory system)
    llm-memory-bench run --dataset alpsbench-task1.yaml --system bfcl_memory_kv

    # With scenario (adds domain-specific persona)
    llm-memory-bench run --dataset bfcl_memory_student.yaml --system bfcl_memory_kv --scenario student

Source: https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard
"""
from __future__ import annotations

from pathlib import Path

import yaml

from ..providers.base import ToolCall
from .base import MemorySystem

PROMPTS_DIR = Path(__file__).parent.parent.parent.parent / "prompts" / "bfcl_memory_kv"


class BFCLMemoryKVSystem(MemorySystem):
    """BFCL Memory with Key-Value backend.

    Requires scenario parameter: student, customer, finance, healthcare, or notetaker.
    Loads prompts from versioned YAML files at design time.
    """

    name = "bfcl_memory_kv"
    description = "BFCL Memory KV backend with optional domain scenarios"

    def __init__(self, scenario: str | None = None, prompt_version: str | None = None):
        """Initialize with optional scenario and prompt version.

        Args:
            scenario: Optional persona: student, customer, finance, healthcare, notetaker.
                     The memory system works the same with or without a scenario.
                     Scenarios just add domain-specific conversational context.
            prompt_version: Prompt version to use (e.g., "v4_2025-07").
                           Defaults to default_version from metadata.yaml.

        Raises:
            ValueError: If scenario is invalid or prompt files not found
        """
        self.scenario = scenario
        self.prompt_version = prompt_version or self._get_default_version()
        self._prompts = self._load_prompts(self.prompt_version)
        # Scenario validation happens in system_prompt() if/when needed

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
        """Combine scenario persona with BFCL memory instructions.

        Scenario is optional - it just adds domain-specific persona flavor.
        The memory system works the same regardless of scenario.
        """
        memory_instruction = self._prompts["memory_instruction"]

        # Scenario is optional - if provided, prepend the persona
        if not self.scenario:
            return memory_instruction

        # Validate scenario if provided
        if self.scenario not in self._prompts["scenarios"]:
            available = list(self._prompts["scenarios"].keys())
            raise ValueError(
                f"Invalid scenario: {self.scenario}. "
                f"Available: {available}"
            )

        persona = self._prompts["scenarios"][self.scenario]
        return f"{persona}\n\n{memory_instruction}"

    def tool_definitions(self) -> list[dict]:
        """BFCL KV memory tool schema."""
        return [
            {
                "name": "core_memory_add",
                "description": "Add a key-value pair to the short-term memory. Make sure to use meaningful keys for easy retrieval later.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": "The key under which the value is stored. The key should be unique and case-sensitive. Keys must be snake_case and cannot contain spaces.",
                        },
                        "value": {
                            "type": "string",
                            "description": "The value to store in the short-term memory.",
                        },
                    },
                    "required": ["key", "value"],
                },
            },
            {
                "name": "core_memory_remove",
                "description": "Remove a key-value pair from the short-term memory.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": "The key to remove from core memory.",
                        },
                    },
                    "required": ["key"],
                },
            },
            {
                "name": "core_memory_replace",
                "description": "Replace the value of an existing key in the short-term memory.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": "The key whose value should be replaced.",
                        },
                        "value": {
                            "type": "string",
                            "description": "The new value to store.",
                        },
                    },
                    "required": ["key", "value"],
                },
            },
            {
                "name": "archival_memory_add",
                "description": "Add a key-value pair to the long-term archival memory.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": "The key for archival storage. Keys must be snake_case.",
                        },
                        "value": {
                            "type": "string",
                            "description": "The value to store in archival memory.",
                        },
                    },
                    "required": ["key", "value"],
                },
            },
            {
                "name": "archival_memory_key_search",
                "description": "Search for keys matching the query in archival memory using BM25+ similarity.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query.",
                        },
                        "k": {
                            "type": "integer",
                            "description": "Number of results to return (default 5).",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            },
        ]

    def extract_stored_fact(self, tool_call: ToolCall) -> str:
        """Extract semantic content from a BFCL KV tool call.

        For storage calls (add/replace), we extract the value.
        For removal/search, we return empty (not storing new facts).
        """
        if tool_call.name in ("core_memory_add", "core_memory_replace", "archival_memory_add"):
            key = tool_call.arguments.get("key", "")
            value = tool_call.arguments.get("value", "")
            # Format as "key: value" for evaluation
            if key and value:
                return f"{key}: {value}"
            return value
        # Remove and search operations don't store new facts
        return ""

    def format_tool_result(self, tool_call: ToolCall) -> dict:
        """Simulated success response from BFCL KV backend."""
        if tool_call.name in ("core_memory_add", "archival_memory_add"):
            return {"status": "Key-value pair added."}
        elif tool_call.name in ("core_memory_remove", "archival_memory_remove"):
            return {"status": "Key removed."}
        elif tool_call.name in ("core_memory_replace", "archival_memory_replace"):
            return {"status": "Value replaced."}
        elif tool_call.name == "archival_memory_key_search":
            return {"ranked_results": []}  # Empty search results
        return {"status": "success"}

    def version_info(self) -> dict:
        """Include scenario, prompt version, and source in version info for tracking."""
        # Load metadata for source information
        metadata_path = PROMPTS_DIR / "metadata.yaml"
        with open(metadata_path) as f:
            metadata = yaml.safe_load(f)

        pulled_date = self._prompts.get("pulled_date")
        info = {
            "system": self.name,
            "prompt_version": self.prompt_version,
            "prompt_source": f"{metadata['source']['repo']} {metadata['source']['path']}",
            "prompt_pulled_date": str(pulled_date) if pulled_date else None,
            "source_commit": self._prompts.get("source_commit"),
        }
        if self.scenario:
            info["scenario"] = self.scenario
        return info

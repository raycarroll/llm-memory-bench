from __future__ import annotations

import importlib
from pathlib import Path

import yaml

from .base import MemorySystem
from .alpsbench_batch import AlpsBenchBatchSystem
from .behavioral_relevance import BehavioralRelevanceSystem
from .bfcl_memory_kv import BFCLMemoryKVSystem
from .claude_code import ClaudeCodeMemorySystem
from .gbrain import GBrainMemorySystem
from .gbrain_sweep import GBrainSweepSystem
from .hindsight import HindsightMemorySystem
from .mem0 import Mem0MemorySystem
from .memoryhub import MemoryHubMemorySystem
from .memoryhub_sweep import MemoryHubSweepSystem
from .openclaw import OpenClawMemorySystem
from .simple import SimpleMemorySystem
from .simple_sweep import SimpleSweepSystem

SYSTEMS: dict[str, type[MemorySystem]] = {
    "simple": SimpleMemorySystem,
    "simple_sweep": SimpleSweepSystem,
    "claude_code": ClaudeCodeMemorySystem,
    "gbrain": GBrainMemorySystem,
    "gbrain_sweep": GBrainSweepSystem,
    "hindsight": HindsightMemorySystem,
    "mem0": Mem0MemorySystem,
    "memoryhub": MemoryHubMemorySystem,
    "memoryhub_sweep": MemoryHubSweepSystem,
    "bfcl_memory_kv": BFCLMemoryKVSystem,
    "behavioral_relevance": BehavioralRelevanceSystem,
    "openclaw": OpenClawMemorySystem,
    "alpsbench_batch": AlpsBenchBatchSystem,
}


def get_system(name: str, **kwargs) -> MemorySystem:
    """Get a memory system instance.

    Args:
        name: System name
        **kwargs: System-specific configuration (e.g., scenario for BFCL systems)

    Returns:
        Instantiated memory system

    Raises:
        ValueError: If system name is unknown or configuration is invalid
    """
    if name not in SYSTEMS:
        raise ValueError(
            f"Unknown memory system: {name}. Available: {list(SYSTEMS.keys())}"
        )
    return SYSTEMS[name](**kwargs)


def list_systems() -> list[str]:
    return list(SYSTEMS.keys())


def list_prompt_versions(name: str) -> list[str]:
    """Return versioned prompt choices for a memory system.

    Prompt files are owned by the concrete system implementation.  Sweep
    variants inherit the prompt directory from their base system.  Systems
    without a versioned prompt return an empty list.
    """
    if name not in SYSTEMS:
        raise ValueError(
            f"Unknown memory system: {name}. Available: {list(SYSTEMS.keys())}"
        )

    prompt_dir: Path | None = None
    for system_class in SYSTEMS[name].__mro__:
        module = importlib.import_module(system_class.__module__)
        candidate = getattr(module, "PROMPTS_DIR", None)
        if candidate is not None:
            prompt_dir = Path(candidate)
            break
    if prompt_dir is None:
        return []

    versions = sorted(path.stem for path in prompt_dir.glob("v*.yaml"))
    metadata_path = prompt_dir / "metadata.yaml"
    if metadata_path.is_file():
        with metadata_path.open(encoding="utf-8") as handle:
            metadata = yaml.safe_load(handle) or {}
        default = metadata.get("default_version")
        if default in versions:
            versions.remove(default)
            versions.insert(0, default)
    return versions

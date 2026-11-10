from __future__ import annotations

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

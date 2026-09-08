from __future__ import annotations

from .base import MemorySystem
from .behavioral_relevance import BehavioralRelevanceSystem
from .bfcl_memory_kv import BFCLMemoryKVSystem
from .claude_code import ClaudeCodeMemorySystem
from .gbrain import GBrainMemorySystem
from .memoryhub import MemoryHubMemorySystem
from .openclaw import OpenClawMemorySystem
from .simple import SimpleMemorySystem

SYSTEMS: dict[str, type[MemorySystem]] = {
    "simple": SimpleMemorySystem,
    "claude_code": ClaudeCodeMemorySystem,
    "gbrain": GBrainMemorySystem,
    "memoryhub": MemoryHubMemorySystem,
    "bfcl_memory_kv": BFCLMemoryKVSystem,
    "behavioral_relevance": BehavioralRelevanceSystem,
    "openclaw": OpenClawMemorySystem,
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

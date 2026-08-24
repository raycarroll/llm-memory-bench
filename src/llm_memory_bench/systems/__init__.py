from __future__ import annotations

from .base import MemorySystem
from .claude_code import ClaudeCodeMemorySystem
from .gbrain import GBrainMemorySystem
from .memoryhub import MemoryHubMemorySystem
from .simple import SimpleMemorySystem

SYSTEMS: dict[str, type[MemorySystem]] = {
    "simple": SimpleMemorySystem,
    "claude_code": ClaudeCodeMemorySystem,
    "gbrain": GBrainMemorySystem,
    "memoryhub": MemoryHubMemorySystem,
}


def get_system(name: str) -> MemorySystem:
    if name not in SYSTEMS:
        raise ValueError(
            f"Unknown memory system: {name}. Available: {list(SYSTEMS.keys())}"
        )
    return SYSTEMS[name]()


def list_systems() -> list[str]:
    return list(SYSTEMS.keys())

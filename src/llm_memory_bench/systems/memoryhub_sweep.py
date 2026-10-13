"""MemoryHub with post-conversation extraction sweep.

Combines realistic turn-by-turn behavior with high-recall batch extraction.
"""
from __future__ import annotations

from .memoryhub import MemoryHubMemorySystem


class MemoryHubSweepSystem(MemoryHubMemorySystem):
    """MemoryHub turn-by-turn + final extraction pass for higher recall.

    Execution flow:
    1. Normal turn-by-turn conversation with memory tool usage
    2. After conversation ends, run final extraction sweep
    3. Combine results from both phases

    This provides:
    - Realistic streaming behavior (phase 1)
    - High recall coverage (phase 2)
    - Fair comparison with batch-only systems
    """

    name = "memoryhub_sweep"
    description = (
        "MemoryHub turn-by-turn extraction with final sweep pass. "
        "Combines realistic streaming behavior with batch extraction recall."
    )

    def requires_final_sweep(self) -> bool:
        """Signal to harness that this system needs final extraction pass."""
        return True

    def final_sweep_prompt(self) -> str:
        """Prompt for final extraction sweep.

        Extends base MemoryHub prompt with instruction to review full conversation.
        """
        base_prompt = self.system_prompt()

        sweep_instruction = """

## Final Extraction Pass

Review the complete conversation below and store any user memories you may have
missed during the conversation.

Apply the same memory hygiene guidelines above, but review ALL turns to ensure
comprehensive coverage. Use the memory() tool to store any facts you find."""

        return base_prompt + sweep_instruction

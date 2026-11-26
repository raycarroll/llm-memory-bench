"""Contracts and execution support for paired operational demos."""

from .results import DemoRunResult, DemoSideResult, VerificationResult
from .config import load_demo_spec
from .runtime import (
    DemoToolRuntime,
    MEMORYHUB_SEARCH_NAME,
    MEMORYHUB_WRITE_NAME,
    clear_memoryhub_project,
)
from .agent import DemoAgent, DemoAgentResult, DemoToolCall
from .verification import OpenShiftVerifier
from .driver import MEMORY_PREPARATION_INSTRUCTIONS, run_demo
from .specs import DemoLimits, DemoSideSpec, DemoSpec

__all__ = [
    "DemoLimits",
    "DemoRunResult",
    "DemoSideResult",
    "DemoSideSpec",
    "DemoSpec",
    "VerificationResult",
    "load_demo_spec",
    "DemoToolRuntime",
    "MEMORYHUB_SEARCH_NAME",
    "MEMORYHUB_WRITE_NAME",
    "clear_memoryhub_project",
    "DemoAgent",
    "DemoAgentResult",
    "DemoToolCall",
    "OpenShiftVerifier",
    "run_demo",
    "MEMORY_PREPARATION_INSTRUCTIONS",
]

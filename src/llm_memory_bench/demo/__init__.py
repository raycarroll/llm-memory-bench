"""Contracts and execution support for paired operational demos."""

from .results import DemoRunResult, DemoSideResult, VerificationResult
from .specs import DemoLimits, DemoSideSpec, DemoSpec

__all__ = [
    "DemoLimits",
    "DemoRunResult",
    "DemoSideResult",
    "DemoSideSpec",
    "DemoSpec",
    "VerificationResult",
]

"""Serializable result contracts for paired operational demos."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


VerificationStatus = Literal["passed", "failed", "unavailable"]


class VerificationResult(BaseModel):
    """Independent cluster verification and its evidence."""

    model_config = ConfigDict(extra="forbid")

    status: VerificationStatus
    passed: bool
    evidence: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None

    @model_validator(mode="after")
    def validate_consistency(self) -> "VerificationResult":
        """Ensure the status and boolean outcome cannot contradict each other."""
        if self.status == "passed" and not self.passed:
            raise ValueError("passed verification must have passed=True")
        if self.status != "passed" and self.passed:
            raise ValueError("only passed verification may have passed=True")
        return self


class DemoSideResult(BaseModel):
    """Execution and verification result for one demo side."""

    model_config = ConfigDict(extra="forbid")

    side: Literal["memory", "baseline"]
    namespace: str
    status: Literal["completed", "failed", "timed_out"]
    content: str = ""
    turns: int = Field(default=0, ge=0)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    memory_queries: list[str] = Field(default_factory=list)
    tokens_in: int = Field(default=0, ge=0)
    tokens_out: int = Field(default=0, ge=0)
    verification: VerificationResult | None = None
    error: str | None = None


class DemoRunResult(BaseModel):
    """Complete paired result, preserving independent side outcomes."""

    model_config = ConfigDict(extra="forbid")

    demo_id: str
    demo_version: str
    provider: str
    model: str
    memory_system: str
    memory_mode: str
    sides: dict[str, DemoSideResult]
    fixture_reset: VerificationResult | None = None
    preflight: dict[str, Any] = Field(default_factory=dict)

    def independent_passed(self) -> bool:
        """Return true only when both sides independently pass verification."""
        return all(
            side.verification is not None and side.verification.passed
            for side in self.sides.values()
        )

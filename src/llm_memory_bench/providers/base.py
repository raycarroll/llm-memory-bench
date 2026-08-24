from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class ToolCall:
    name: str
    arguments: dict


@dataclass
class ProviderResponse:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


class ProviderError(Exception):
    """Raised for provider configuration or API errors with user-friendly messages."""


class LLMProvider(ABC):
    def __init__(self, model: str, **kwargs):
        self.model = model

    @abstractmethod
    async def generate(
        self,
        messages: list[dict],
        tools: list[dict],
        system: str,
    ) -> ProviderResponse:
        ...

    @abstractmethod
    def build_tool_result_messages(
        self,
        response: ProviderResponse,
        format_tool_result: Callable[[ToolCall], dict] | None = None,
    ) -> list[dict]:
        ...

    @staticmethod
    def _default_tool_result(tool_call: ToolCall) -> dict:
        return {"status": "success"}

    async def preflight(self) -> None:
        """Quick API call to verify credentials and model access before a full run."""
        try:
            await self.generate(
                messages=[{"role": "user", "content": "Hi"}],
                tools=[],
                system="Reply with OK.",
            )
        except Exception as e:
            raise ProviderError(self._friendly_error(e)) from e

    def _friendly_error(self, exc: Exception) -> str:
        """Convert provider SDK exceptions into actionable messages."""
        msg = str(exc)
        cls = type(exc).__name__

        if "authentication" in msg.lower() or "api_key" in msg.lower() or "auth_token" in msg.lower():
            return self._auth_hint()

        if "404" in msg or "not found" in msg.lower() or "NOT_FOUND" in msg:
            return (
                f"Model not found: {self.model}\n"
                f"The provider returned a 404. Check that the model ID is correct "
                f"for this provider.\n"
                f"Detail: {msg}"
            )

        if "429" in msg or "rate" in msg.lower():
            return f"Rate limited by provider. Try again shortly.\nDetail: {msg}"

        if "permission" in msg.lower() or "403" in msg:
            return (
                f"Permission denied for model {self.model}.\n"
                f"Check that your credentials have access to this model.\n"
                f"Detail: {msg}"
            )

        return f"Provider API error ({cls}): {msg}"

    def _auth_hint(self) -> str:
        return (
            f"Authentication failed for provider '{type(self).__name__}'.\n"
            f"Check your API key or credentials."
        )

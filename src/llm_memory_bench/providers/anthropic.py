from __future__ import annotations

import json
import os

import anthropic

from .base import LLMProvider, ProviderError, ProviderResponse, ToolCall


class AnthropicProvider(LLMProvider):
    def __init__(self, model: str = "claude-sonnet-4-20250514", **kwargs):
        super().__init__(model, **kwargs)
        api_key_env = kwargs.get("api_key_env")
        api_key = os.environ.get(api_key_env) if api_key_env else None
        try:
            self.client = anthropic.AsyncAnthropic(api_key=api_key)
        except Exception as e:
            raise ProviderError(self._auth_hint()) from e

    def _auth_hint(self) -> str:
        return (
            "Authentication failed for Anthropic API.\n"
            "Set the ANTHROPIC_API_KEY environment variable, or pass --api-key-env "
            "with the name of the env var containing your key."
        )

    def _to_anthropic_tools(self, tools: list[dict]) -> list[dict]:
        return [
            {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "input_schema": tool["input_schema"],
            }
            for tool in tools
        ]

    async def generate(
        self,
        messages: list[dict],
        tools: list[dict],
        system: str,
    ) -> ProviderResponse:
        try:
            kwargs = dict(
                model=self.model,
                max_tokens=1024,
                system=system,
                messages=messages,
            )
            if tools:
                kwargs["tools"] = self._to_anthropic_tools(tools)
            response = await self.client.messages.create(**kwargs)
        except Exception as e:
            raise ProviderError(self._friendly_error(e)) from e

        text_parts = []
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(name=block.name, arguments=block.input)
                )

        return ProviderResponse(
            text=" ".join(text_parts),
            tool_calls=tool_calls,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    def build_tool_result_messages(
        self,
        response: ProviderResponse,
        format_tool_result=None,
    ) -> list[dict]:
        if not response.tool_calls:
            return []

        fmt = format_tool_result or self._default_tool_result

        assistant_content = []
        for tc in response.tool_calls:
            assistant_content.append(
                {
                    "type": "tool_use",
                    "id": f"toolu_{hash(json.dumps(tc.arguments, sort_keys=True)) % (10**12):012d}",
                    "name": tc.name,
                    "input": tc.arguments,
                }
            )
        if response.text:
            assistant_content.insert(0, {"type": "text", "text": response.text})

        tool_results = []
        for tc in response.tool_calls:
            result = fmt(tc)
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": f"toolu_{hash(json.dumps(tc.arguments, sort_keys=True)) % (10**12):012d}",
                    "content": json.dumps(result),
                }
            )

        return [
            {"role": "assistant", "content": assistant_content},
            {"role": "user", "content": tool_results},
        ]

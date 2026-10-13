from __future__ import annotations

import json
import os

import openai

from .base import LLMProvider, ProviderError, ProviderResponse, ToolCall


class OpenAIProvider(LLMProvider):
    def __init__(self, model: str = "gpt-4o", **kwargs):
        super().__init__(model, **kwargs)
        api_key_env = kwargs.get("api_key_env") or "OPENAI_API_KEY"
        api_key = os.environ.get(api_key_env)
        try:
            # Omitting api_key lets the SDK resolve its standard environment
            # variable. Passing None explicitly makes the SDK reject a valid
            # OPENAI_API_KEY in the process environment.
            self.client = openai.AsyncOpenAI(api_key=api_key) if api_key else openai.AsyncOpenAI()
        except Exception as e:
            raise ProviderError(self._auth_hint()) from e

    def _auth_hint(self) -> str:
        return (
            "Authentication failed for OpenAI API.\n"
            "Set the OPENAI_API_KEY environment variable, or pass --api-key-env "
            "with the name of the env var containing your key."
        )

    def _to_openai_tools(self, tools: list[dict]) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool["input_schema"],
                },
            }
            for tool in tools
        ]

    def _is_reasoning_model(self) -> bool:
        """Check if model is a reasoning model (o1/o3/luna series)."""
        model_lower = self.model.lower()
        return any(x in model_lower for x in ["o1", "o3", "luna", "reasoning"])

    async def generate(
        self,
        messages: list[dict],
        tools: list[dict],
        system: str,
    ) -> ProviderResponse:
        api_messages = [{"role": "system", "content": system}]
        api_messages.extend(messages)

        try:
            # Increase token limit for batch extraction (no tools = JSON output)
            max_tokens = 4096 if not tools else 1024

            kwargs = dict(
                model=self.model,
                messages=api_messages,
                max_completion_tokens=max_tokens,
            )

            # Reasoning models require reasoning_effort='none' with tools in /v1/chat/completions
            # or use /v1/responses endpoint (not yet in SDK)
            if self._is_reasoning_model() and tools:
                kwargs["reasoning_effort"] = "none"

            if tools:
                kwargs["tools"] = self._to_openai_tools(tools)

            # Debug: log the actual request size
            import json
            import os
            if os.getenv("DEBUG_TOKENS"):
                print(f"\n[OpenAI Request Debug]")
                print(f"  Messages: {len(api_messages)}")
                print(f"  Tools: {len(tools) if tools else 0}")
                print(f"  System length: {len(system)} chars")
                print(f"  Request JSON size: {len(json.dumps(kwargs, default=str))} chars")

            response = await self.client.chat.completions.create(**kwargs)
        except Exception as e:
            raise ProviderError(self._friendly_error(e)) from e

        # Warn if API returned a different model than requested
        actual_model = response.model
        if actual_model != self.model:
            import warnings
            warnings.warn(
                f"OpenAI API returned model '{actual_model}' but '{self.model}' was requested. "
                f"This may indicate an invalid model name or API fallback behavior.",
                stacklevel=2
            )

        choice = response.choices[0]
        text = choice.message.content or ""
        tool_calls = []

        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                tool_calls.append(
                    ToolCall(
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments),
                    )
                )

        return ProviderResponse(
            text=text,
            tool_calls=tool_calls,
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
            metadata={"actual_model": actual_model},  # Track what model was actually used
        )

    def build_tool_result_messages(
        self,
        response: ProviderResponse,
        format_tool_result=None,
    ) -> list[dict]:
        if not response.tool_calls:
            return []

        fmt = format_tool_result or self._default_tool_result

        assistant_msg: dict = {"role": "assistant", "content": response.text or None}
        openai_tool_calls = []
        for i, tc in enumerate(response.tool_calls):
            call_id = f"call_{hash(json.dumps(tc.arguments, sort_keys=True)) % (10**12):012d}"
            openai_tool_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
            )
        assistant_msg["tool_calls"] = openai_tool_calls

        result_msgs = [assistant_msg]
        for i, tc in enumerate(response.tool_calls):
            call_id = f"call_{hash(json.dumps(tc.arguments, sort_keys=True)) % (10**12):012d}"
            result = fmt(tc)
            result_msgs.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": json.dumps(result),
                }
            )

        return result_msgs

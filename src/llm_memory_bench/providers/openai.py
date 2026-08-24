from __future__ import annotations

import json
import os

import openai

from .base import LLMProvider, ProviderError, ProviderResponse, ToolCall


class OpenAIProvider(LLMProvider):
    def __init__(self, model: str = "gpt-4o", **kwargs):
        super().__init__(model, **kwargs)
        api_key_env = kwargs.get("api_key_env")
        api_key = os.environ.get(api_key_env) if api_key_env else None
        try:
            self.client = openai.AsyncOpenAI(api_key=api_key)
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

    async def generate(
        self,
        messages: list[dict],
        tools: list[dict],
        system: str,
    ) -> ProviderResponse:
        api_messages = [{"role": "system", "content": system}]
        api_messages.extend(messages)

        try:
            kwargs = dict(
                model=self.model,
                messages=api_messages,
                max_tokens=1024,
            )
            if tools:
                kwargs["tools"] = self._to_openai_tools(tools)
            response = await self.client.chat.completions.create(**kwargs)
        except Exception as e:
            raise ProviderError(self._friendly_error(e)) from e

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

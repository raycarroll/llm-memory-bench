from __future__ import annotations

import json

import litellm

from .base import LLMProvider, ProviderError, ProviderResponse, ToolCall


class LiteLLMProvider(LLMProvider):
    """Catch-all provider using LiteLLM for Ollama, Gemini, Mistral, etc."""

    def __init__(self, model: str = "ollama/llama3", **kwargs):
        super().__init__(model, **kwargs)

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
            response = await litellm.acompletion(**kwargs)
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

        usage = response.usage
        return ProviderResponse(
            text=text,
            tool_calls=tool_calls,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
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
        for tc in response.tool_calls:
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
        for tc in response.tool_calls:
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

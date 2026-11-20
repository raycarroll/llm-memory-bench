"""Bounded provider/tool loop for one side of a paired demo."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..providers.base import LLMProvider, ProviderResponse, ToolCall
from .runtime import DemoToolRuntime, MEMORYHUB_SEARCH_NAME
from .specs import DemoSpec


@dataclass
class DemoToolCall:
    name: str
    arguments: dict[str, Any]
    result: str
    is_error: bool
    duration_seconds: float


@dataclass
class DemoAgentResult:
    side: str
    namespace: str
    status: str
    content: str = ""
    turns: int = 0
    tool_calls: list[DemoToolCall] = field(default_factory=list)
    memory_queries: list[str] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    error: str | None = None


class DemoAgent:
    """Run one provider against one side-scoped demo runtime."""

    def __init__(
        self,
        *,
        side: str,
        provider: LLMProvider,
        spec: DemoSpec,
        memoryhub_api_key: str | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ):
        self.side = side
        self.provider = provider
        self.spec = spec
        self.on_event = on_event
        self.memoryhub_api_key = memoryhub_api_key

    async def run(self) -> DemoAgentResult:
        result = DemoAgentResult(
            side=self.side,
            namespace=self.spec.sides[self.side].namespace,
            status="failed",
        )
        started = time.monotonic()
        deadline = started + self.spec.limits.timeout_seconds
        messages: list[dict[str, Any]] = [{"role": "user", "content": self.spec.prompt}]
        force_memory_search = False
        read_only_calls = 0

        try:
            async with DemoToolRuntime(
                side=self.side,
                spec=self.spec,
                memoryhub_api_key=self.memoryhub_api_key,
                timeout_seconds=self.spec.limits.tool_timeout_seconds,
            ) as runtime:
                for turn in range(1, self.spec.limits.max_turns + 1):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        result.status = "timed_out"
                        break
                    result.turns = turn
                    self._emit("turn_started", turn=turn)
                    tools = [self._provider_tool(tool) for tool in runtime.tools_for_turn()]
                    response = await asyncio.wait_for(
                        self.provider.generate(
                            messages=messages,
                            tools=tools,
                            system=self.spec.system_prompt,
                            tool_choice=MEMORYHUB_SEARCH_NAME if force_memory_search else None,
                            disable_parallel_tool_calls=True,
                        ),
                        timeout=remaining,
                    )
                    result.tokens_in += response.input_tokens
                    result.tokens_out += response.output_tokens
                    self._emit(
                        "model_response",
                        turn=turn,
                        content=response.text,
                        tools=[call.name for call in response.tool_calls],
                    )
                    if not response.tool_calls:
                        if runtime._requires_memory_search and not runtime.memory_search_completed:
                            messages.append({"role": "assistant", "content": response.text})
                            messages.append({
                                "role": "user",
                                "content": "Search MemoryHub before finalizing, using the live symptoms observed.",
                            })
                            force_memory_search = True
                            continue
                        result.content = response.text
                        result.status = "completed"
                        break

                    tool_results: dict[str, tuple[str, bool]] = {}
                    for call in response.tool_calls:
                        if len(result.tool_calls) >= self.spec.limits.max_tool_calls:
                            result.status = "timed_out"
                            raise RuntimeError("demo tool-call limit reached")
                        call_started = time.monotonic()
                        if force_memory_search and call.name != MEMORYHUB_SEARCH_NAME:
                            text, failed = "Blocked: search MemoryHub first.", True
                        else:
                            try:
                                text, failed = await asyncio.wait_for(
                                    runtime.call(call.name, call.arguments),
                                    timeout=min(
                                        self.spec.limits.tool_timeout_seconds,
                                        max(0.01, deadline - call_started),
                                    ),
                                )
                            except Exception as exc:
                                text, failed = f"Tool failed: {type(exc).__name__}: {exc}", True
                        result.tool_calls.append(
                            DemoToolCall(
                                name=call.name,
                                arguments=call.arguments,
                                result=text,
                                is_error=failed,
                                duration_seconds=round(time.monotonic() - call_started, 3),
                            )
                        )
                        tool_results[self._call_key(call)] = (text, failed)
                        self._emit("tool_completed", name=call.name, result=text, is_error=failed)
                        if call.name == MEMORYHUB_SEARCH_NAME and not failed:
                            result.memory_queries.append(str(call.arguments.get("query", "")))
                        elif call.name in self.spec.read_only_tools and not failed:
                            read_only_calls += 1
                    messages.extend(self._tool_result_messages(response, tool_results))
                    if runtime._requires_memory_search and read_only_calls and not runtime.memory_search_completed:
                        messages.append({
                            "role": "user",
                            "content": "You inspected the workload. Search MemoryHub now before any Kubernetes change.",
                        })
                        force_memory_search = True
                    else:
                        force_memory_search = False
                else:
                    result.status = "timed_out"
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
            if result.status != "timed_out":
                result.status = "failed"
            self._emit("run_error", error=result.error)
        return result

    @staticmethod
    def _provider_tool(tool: dict[str, Any]) -> dict[str, Any]:
        return {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "input_schema": tool.get("parameters", {"type": "object", "properties": {}}),
        }

    @staticmethod
    def _call_key(call: ToolCall) -> str:
        return json.dumps({"name": call.name, "arguments": call.arguments}, sort_keys=True)

    def _tool_result_messages(
        self,
        response: ProviderResponse,
        results: dict[str, tuple[str, bool]],
    ) -> list[dict[str, Any]]:
        def format_result(call: ToolCall) -> dict[str, Any]:
            text, failed = results.get(self._call_key(call), ("Tool result unavailable", True))
            return {"content": text, "is_error": failed}

        return self.provider.build_tool_result_messages(response, format_result)

    def _emit(self, event: str, **values: Any) -> None:
        if self.on_event:
            self.on_event({"event": event, "side": self.side, **values})

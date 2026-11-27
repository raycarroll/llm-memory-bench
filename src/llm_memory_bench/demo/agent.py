"""Bounded provider/tool loop for one side of a paired demo."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..providers.base import LLMProvider, ProviderResponse, ToolCall
from .runtime import DemoToolRuntime, MEMORYHUB_SEARCH_NAME, MEMORYHUB_WRITE_NAME
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
    elapsed_seconds: float = 0.0
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
        verification_callback: Callable[[], Any] | None = None,
    ):
        self.side = side
        self.provider = provider
        self.spec = spec
        self.on_event = on_event
        self.memoryhub_api_key = memoryhub_api_key
        self.verification_callback = verification_callback

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
        forced_memory_tool: str | None = None
        read_only_calls = 0

        try:
            self._emit("prompt", content=self.spec.prompt)
            async with DemoToolRuntime(
                side=self.side,
                spec=self.spec,
                memoryhub_api_key=self.memoryhub_api_key,
                timeout_seconds=self.spec.limits.tool_timeout_seconds,
            ) as runtime:
                developer_context = (
                    f"You are operating only in namespace {runtime.namespace}. "
                    "Pass that namespace to namespaced tools. The MCP server's RBAC "
                    "is the cluster access boundary."
                )
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
                            messages=[
                                {"role": "developer", "content": developer_context},
                                *messages,
                            ],
                            tools=tools,
                            system=self.spec.system_prompt,
                            tool_choice=(
                                forced_memory_tool
                                or (MEMORYHUB_SEARCH_NAME if force_memory_search else "auto")
                            ),
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
                                "role": "developer",
                                "content": "Search MemoryHub before finalizing, using the live symptoms observed.",
                            })
                            force_memory_search = True
                            continue
                        if self.verification_callback is not None:
                            verification = await self.verification_callback()
                            self._emit("verification", check=verification)
                            if not verification.get("passed", False):
                                messages.append({"role": "assistant", "content": response.text})
                                messages.append({
                                    "role": "user",
                                    "content": (
                                        "Independent verification has not passed. Continue investigating and repairing "
                                        "the live workload. Verification evidence: "
                                        + json.dumps(verification, default=str)
                                    ),
                                })
                                force_memory_search = False
                                forced_memory_tool = None
                                continue
                            if runtime._requires_memory_write:
                                runtime.enable_memory_writes()
                                messages.append({"role": "assistant", "content": response.text})
                                messages.append({
                                    "role": "developer",
                                    "content": (
                                        "Independent verification passed. Now write the durable "
                                        "MemoryHub lesson as the final action."
                                    ),
                                })
                                forced_memory_tool = MEMORYHUB_WRITE_NAME
                                continue
                        result.content = response.text
                        result.status = "completed"
                        break

                    remaining_calls = self.spec.limits.max_tool_calls - len(result.tool_calls)
                    if len(response.tool_calls) > remaining_calls:
                        result.status = "timed_out"
                        result.error = (
                            f"maximum tool-call limit ({self.spec.limits.max_tool_calls}) exceeded"
                        )
                        self._emit("run_error", error=result.error)
                        break

                    tool_results: dict[str, tuple[str, bool]] = {}
                    for call in response.tool_calls:
                        call_started = time.monotonic()
                        if force_memory_search and call.name != MEMORYHUB_SEARCH_NAME:
                            text, failed = "Blocked: search MemoryHub first.", True
                        elif forced_memory_tool and call.name != forced_memory_tool:
                            text, failed = f"Blocked: call {forced_memory_tool} first.", True
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
                        elif call.name == MEMORYHUB_WRITE_NAME and not failed:
                            forced_memory_tool = None
                        elif call.name in self.spec.read_only_tools and not failed:
                            read_only_calls += 1
                    messages.extend(self._tool_result_messages(response, tool_results))
                    if len(result.tool_calls) >= self.spec.limits.max_tool_calls:
                        result.status = "timed_out"
                        result.error = (
                            f"maximum tool-call limit ({self.spec.limits.max_tool_calls}) reached"
                        )
                        self._emit("run_error", error=result.error)
                        break
                    if runtime._requires_memory_search and read_only_calls and not runtime.memory_search_completed:
                        messages.append({
                            "role": "developer",
                            "content": (
                                "You have inspected the live workload. Search MemoryHub "
                                "now with a concise query based on the observed symptoms "
                                "before considering any Kubernetes change."
                            ),
                        })
                        force_memory_search = True
                    else:
                        force_memory_search = False
                else:
                    result.status = "timed_out"
        except Exception as exc:
            result.error = self._format_exception(exc)
            if result.status != "timed_out":
                result.status = "failed"
            self._emit("run_error", error=result.error)
        result.elapsed_seconds = round(time.monotonic() - started, 3)
        return result

    @staticmethod
    def _format_exception(exc: BaseException) -> str:
        """Preserve the useful leaf error from asyncio task groups."""

        leaves: list[str] = []

        def visit(error: BaseException) -> None:
            nested = getattr(error, "exceptions", None)
            if nested:
                for child in nested:
                    visit(child)
                return
            message = str(error).strip()
            rendered = f"{type(error).__name__}: {message}" if message else type(error).__name__
            if rendered not in leaves:
                leaves.append(rendered)

        visit(exc)
        if leaves:
            return "Multiple errors: " + " | ".join(leaves)
        return f"{type(exc).__name__}: {exc}"

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

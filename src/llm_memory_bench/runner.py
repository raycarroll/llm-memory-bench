from __future__ import annotations

import time
from dataclasses import dataclass, field

from rich.console import Console

from .config import RunConfig, get_provider
from .dataset import Conversation, Dataset
from .providers.base import ToolCall
from .systems import get_system

console = Console()


@dataclass
class TurnResult:
    turn_index: int
    role: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0


@dataclass
class ConversationResult:
    conversation_id: str
    turn_results: list[TurnResult] = field(default_factory=list)

    @property
    def total_tool_calls(self) -> int:
        return sum(len(tr.tool_calls) for tr in self.turn_results)

    @property
    def total_tokens(self) -> int:
        return sum(tr.input_tokens + tr.output_tokens for tr in self.turn_results)


@dataclass
class RunResult:
    config: RunConfig
    conversation_results: list[ConversationResult] = field(default_factory=list)
    total_time_ms: float = 0


async def run_conversation(
    conversation: Conversation,
    provider,
    system_prompt: str,
    tools: list[dict],
    format_tool_result,
) -> ConversationResult:
    from rich.live import Live

    result = ConversationResult(conversation_id=conversation.id)
    messages: list[dict] = []
    user_turns = sum(1 for t in conversation.turns if t.role == "user")
    user_turn_num = 0

    with Live("", console=console, refresh_per_second=4, transient=True) as live:
        for i, turn in enumerate(conversation.turns):
            messages.append({"role": turn.role, "content": turn.content})

            if turn.role != "user":
                result.turn_results.append(
                    TurnResult(turn_index=i, role=turn.role)
                )
                continue

            user_turn_num += 1
            live.update(f"    [dim]turn {user_turn_num}/{user_turns}...[/dim]")

            start = time.monotonic()
            response = await provider.generate(
                messages=messages,
                tools=tools,
                system=system_prompt,
            )
            latency = (time.monotonic() - start) * 1000

            tool_str = f" +{len(response.tool_calls)} tools" if response.tool_calls else ""
            live.update(
                f"    [dim]turn {user_turn_num}/{user_turns} ({latency:.0f}ms{tool_str})[/dim]"
            )

            turn_result = TurnResult(
                turn_index=i,
                role=turn.role,
                tool_calls=response.tool_calls,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                latency_ms=latency,
            )
            result.turn_results.append(turn_result)

            if response.tool_calls:
                tool_msgs = provider.build_tool_result_messages(
                    response, format_tool_result=format_tool_result
                )
                messages.extend(tool_msgs)
            else:
                messages.append({"role": "assistant", "content": response.text})

    total_calls = result.total_tool_calls
    console.print(
        f"    done — {user_turns} turns, {total_calls} tool calls, "
        f"{result.total_tokens} tokens",
        style="dim",
    )
    return result


async def run_benchmark(dataset: Dataset, config: RunConfig) -> RunResult:
    provider = get_provider(config)
    memory_system = get_system(config.system)
    system_prompt = memory_system.system_prompt()
    tools = memory_system.tool_definitions()

    console.print("Verifying provider credentials...", style="dim")
    await provider.preflight()
    console.print("Provider OK.", style="dim")

    conversations = dataset.conversations
    if config.max_conversations:
        conversations = conversations[: config.max_conversations]

    run_result = RunResult(config=config)
    start = time.monotonic()

    for i, conversation in enumerate(conversations):
        console.print(
            f"  [{i+1}/{len(conversations)}] {conversation.id} "
            f"({len(conversation.turns)} turns)",
            style="dim",
        )
        conv_result = await run_conversation(
            conversation,
            provider,
            system_prompt,
            tools,
            format_tool_result=memory_system.format_tool_result,
        )
        run_result.conversation_results.append(conv_result)

    run_result.total_time_ms = (time.monotonic() - start) * 1000
    return run_result

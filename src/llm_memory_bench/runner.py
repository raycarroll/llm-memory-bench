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


async def run_conversation_sequential(
    conversation: Conversation,
    provider,
    system_prompt: str,
    tools: list[dict],
    format_tool_result,
    memory_state: dict,  # Accumulated memory across conversations
    memory_system,
) -> ConversationResult:
    """Run a conversation with access to accumulated memory state."""
    from rich.live import Live

    result = ConversationResult(conversation_id=conversation.id)
    messages: list[dict] = []
    user_turns = sum(1 for t in conversation.turns if t.role == "user")
    user_turn_num = 0

    with Live("", console=console, refresh_per_second=4, transient=True) as live:
        for i, turn in enumerate(conversation.turns):
            # For user turns, include current memory state in context
            if turn.role == "user" and memory_state:
                memory_context = _format_memory_context(memory_state, memory_system)
                if memory_context:
                    # Prepend memory context to user message
                    content_with_memory = f"{memory_context}\n\n{turn.content}"
                else:
                    content_with_memory = turn.content
                messages.append({"role": turn.role, "content": content_with_memory})
            else:
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

            # Update memory state from tool calls
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    _update_memory_state(memory_state, tool_call, memory_system)

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


def _format_memory_context(memory_state: dict, memory_system) -> str:
    """Format accumulated memory state for inclusion in user messages."""
    if not memory_state:
        return ""

    # For KV systems: show stored key-value pairs
    system_name = memory_system.__class__.__name__.lower()
    if "kv" in system_name or "key" in system_name:
        if not memory_state:
            return ""
        items = [f"- {k}: {v}" for k, v in sorted(memory_state.items())]
        return f"[Current Memory]\n" + "\n".join(items[:50])  # Limit to avoid context overflow

    # For other systems: generic format
    if memory_state:
        items = [f"- {v}" for v in list(memory_state.values())[:50]]
        return f"[Stored Facts]\n" + "\n".join(items)

    return ""


def _update_memory_state(memory_state: dict, tool_call: ToolCall, memory_system):
    """Update memory state based on tool call."""
    # Handle KV store operations
    if tool_call.name in ("core_memory_add", "archival_memory_add"):
        key = tool_call.arguments.get("key")
        value = tool_call.arguments.get("value")
        if key and value:
            memory_state[key] = value

    elif tool_call.name == "core_memory_replace":
        key = tool_call.arguments.get("key")
        value = tool_call.arguments.get("value")
        if key and value:
            memory_state[key] = value

    elif tool_call.name in ("core_memory_remove", "archival_memory_remove"):
        key = tool_call.arguments.get("key")
        if key and key in memory_state:
            del memory_state[key]

    # For simple memory systems
    elif tool_call.name == "store_fact":
        fact = tool_call.arguments.get("fact")
        if fact:
            # Use fact as both key and value for simple systems
            memory_state[fact] = fact

    else:
        fact = memory_system.extract_stored_fact(tool_call)
        if fact:
            memory_state[fact] = fact


async def run_benchmark(dataset: Dataset, config: RunConfig) -> RunResult:
    # Validate max_conversations compatibility with dataset type EARLY
    from .dataset import GroundTruthType
    if config.max_conversations:
        if (dataset.ground_truth_type == GroundTruthType.CUMULATIVE
            and dataset.cumulative_ground_truth is not None):
            raise ValueError(
                f"max_conversations={config.max_conversations} is incompatible with "
                f"dataset-level cumulative ground truth. The ground truth assumes ALL "
                f"{len(dataset.conversations)} conversations are processed sequentially. "
                f"Either remove --max-conversations or use a per-turn/per-conversation dataset."
            )

    provider = get_provider(config)

    system_kwargs = {}
    if config.scenario:
        system_kwargs["scenario"] = config.scenario
    memory_system = get_system(config.system, **system_kwargs)

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

    # Detect if we need sequential mode (dataset-level cumulative ground truth)
    from .dataset import GroundTruthType
    is_sequential = (
        dataset.ground_truth_type == GroundTruthType.CUMULATIVE
        and dataset.cumulative_ground_truth is not None
    )

    if is_sequential:
        console.print("  [bold cyan]Sequential mode:[/bold cyan] Memory persists across conversations", style="dim")
        memory_state = {}  # Accumulate memory across all conversations

        for i, conversation in enumerate(conversations):
            console.print(
                f"  [{i+1}/{len(conversations)}] {conversation.id} "
                f"({len(conversation.turns)} turns) [dim]| {len(memory_state)} facts in memory[/dim]",
                style="dim",
            )
            conv_result = await run_conversation_sequential(
                conversation,
                provider,
                system_prompt,
                tools,
                format_tool_result=memory_system.format_tool_result,
                memory_state=memory_state,
                memory_system=memory_system,
            )
            run_result.conversation_results.append(conv_result)
    else:
        # Independent mode: each conversation is isolated
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

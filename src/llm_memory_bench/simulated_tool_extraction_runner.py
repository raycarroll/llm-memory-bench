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
    response_text: str = ""  # Assistant's text response
    metadata: dict = field(default_factory=dict)  # Provider metadata (e.g., actual_model)
    errors: list[str] = field(default_factory=list)  # API errors, warnings, or exceptions
    warnings: list[str] = field(default_factory=list)  # Non-fatal warnings


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
    stop_reason: str | None = None
    preflight_input_tokens: int = 0
    preflight_output_tokens: int = 0
    actual_model: str | None = None  # Model actually used by provider (may differ from config)


@dataclass
class UsageBudget:
    model: str
    max_total_tokens: int | None = None
    max_cost_usd: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    stopped: bool = False

    def record(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        total = self.input_tokens + self.output_tokens
        if self.max_total_tokens is not None and total >= self.max_total_tokens:
            self.stopped = True
        if self.max_cost_usd is not None:
            from .pricing import estimate_cost
            cost = estimate_cost(self.model, self.input_tokens, self.output_tokens)
            if cost is not None and cost >= self.max_cost_usd:
                self.stopped = True


async def run_batch_conversation(
    conversation: Conversation,
    provider,
    system_prompt: str,
    budget: UsageBudget | None = None,
) -> ConversationResult:
    """Run batch extraction (AlpsBench-style).

    Sends entire conversation as JSON in single API call.
    Model returns JSON (not tool calls).
    """
    import json
    from rich.live import Live

    result = ConversationResult(conversation_id=conversation.id)

    # Build dialogue JSON (AlpsBench format)
    dialogue = []
    for turn in conversation.turns:
        dialogue.append({
            "role": turn.role,
            "text": turn.content
        })

    dialogue_json = json.dumps({"dialogue": dialogue}, ensure_ascii=False)

    with Live("", console=console, refresh_per_second=4, transient=True) as live:
        live.update("[dim]batch extraction...[/dim]")

        start = time.monotonic()
        response = await provider.generate(
            messages=[{"role": "user", "content": dialogue_json}],
            tools=[],  # No tools - just JSON response
            system=system_prompt,
        )
        latency = (time.monotonic() - start) * 1000

        live.update(f"[dim]batch extraction ({latency:.0f}ms)[/dim]")

        # Parse JSON response to extract memories
        errors = []
        warnings = []
        tool_calls = []
        memory_items = []

        try:
            # Remove markdown code blocks if present
            text = response.text.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                if lines[0].lower() in ["```", "```json"]:
                    text = "\n".join(lines[1:-1]) if lines[-1] == "```" else "\n".join(lines[1:])

            memories_data = json.loads(text)
            if not isinstance(memories_data, dict):
                raise ValueError(f"Expected dict, got {type(memories_data).__name__}")

            memory_items = memories_data.get("memory_items", [])

            # Convert each memory item to a separate tool call for evaluation
            # This matches our normal evaluation flow (one fact per tool call)
            from .providers.base import ToolCall
            tool_calls = []
            for item in memory_items:
                if isinstance(item, dict):
                    # Create a tool call for each memory item
                    # Use the "value" field as the stored fact
                    tool_calls.append(
                        ToolCall(
                            name="batch_memory_item",
                            arguments=item  # Store full item (type, label, value, etc.)
                        )
                    )
        except (json.JSONDecodeError, ValueError, AttributeError) as e:
            error_msg = f"JSON parsing failed: {e}"
            errors.append(error_msg)
            console.print(f"[yellow]Warning: Failed to parse JSON response for {conversation.id}[/yellow]")
            console.print(f"[yellow]  Error: {e}[/yellow]")
            console.print(f"[yellow]  Response preview: {response.text[:200]}...[/yellow]")

            # Save raw response for debugging
            import os
            debug_dir = "results/debug"
            os.makedirs(debug_dir, exist_ok=True)
            debug_path = f"{debug_dir}/failed_json_{conversation.id}.txt"
            with open(debug_path, "w") as f:
                f.write(response.text)
            console.print(f"[yellow]  Saved raw response to {debug_path}[/yellow]")

        # Record as a single turn at the end of conversation
        turn_result = TurnResult(
            turn_index=len(conversation.turns) - 1,
            role="batch",
            tool_calls=tool_calls,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            latency_ms=latency,
            response_text=response.text,
            errors=errors,
            warnings=warnings,
        )
        result.turn_results.append(turn_result)

        if budget:
            budget.record(response.input_tokens, response.output_tokens)

    console.print(
        f"    done — {len(memory_items)} memories extracted, "
        f"{response.input_tokens + response.output_tokens} tokens",
        style="dim",
    )
    return result


async def run_conversation(
    conversation: Conversation,
    provider,
    system_prompt: str,
    tools: list[dict],
    format_tool_result,
    budget: UsageBudget | None = None,
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
                response_text=response.text,
            )
            result.turn_results.append(turn_result)
            if budget:
                budget.record(response.input_tokens, response.output_tokens)

            if response.tool_calls:
                tool_msgs = provider.build_tool_result_messages(
                    response, format_tool_result=format_tool_result
                )
                messages.extend(tool_msgs)
            else:
                messages.append({"role": "assistant", "content": response.text})

            if budget and budget.stopped:
                break

    total_calls = result.total_tool_calls
    console.print(
        f"    done — {user_turns} turns, {total_calls} tool calls, "
        f"{result.total_tokens} tokens",
        style="dim",
    )
    return result


async def run_final_sweep(
    conversation: Conversation,
    provider,
    sweep_prompt: str,
    tools: list[dict],
    format_tool_result,
    budget: UsageBudget | None = None,
) -> list[ToolCall]:
    """Run final extraction sweep after conversation completes.

    Args:
        conversation: The completed conversation
        provider: LLM provider
        sweep_prompt: Extraction-focused system prompt
        tools: Memory tools
        format_tool_result: Tool result formatter
        budget: Optional token budget

    Returns:
        List of tool calls from the sweep
    """
    import json

    # Build conversation text for review
    dialogue = []
    for turn in conversation.turns:
        dialogue.append({
            "role": turn.role,
            "text": turn.content
        })

    conversation_text = json.dumps({"dialogue": dialogue}, ensure_ascii=False)

    console.print("    [dim]final sweep...[/dim]", style="dim")

    start = time.monotonic()
    response = await provider.generate(
        messages=[{"role": "user", "content": conversation_text}],
        tools=tools,
        system=sweep_prompt,
    )
    latency = (time.monotonic() - start) * 1000

    if budget:
        budget.record(response.input_tokens, response.output_tokens)

    tool_calls = response.tool_calls or []
    console.print(
        f"    sweep done — {len(tool_calls)} additional facts, "
        f"{response.input_tokens + response.output_tokens} tokens ({latency:.0f}ms)",
        style="dim",
    )

    return tool_calls


async def run_conversation_sequential(
    conversation: Conversation,
    provider,
    system_prompt: str,
    tools: list[dict],
    format_tool_result,
    memory_state: dict,  # Accumulated memory across conversations
    memory_system,
    budget: UsageBudget | None = None,
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
                response_text=response.text,
            )
            result.turn_results.append(turn_result)
            if budget:
                budget.record(response.input_tokens, response.output_tokens)

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

            if budget and budget.stopped:
                break

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
    # Detect if we need sequential mode (dataset-level cumulative ground truth)
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

    conversations = dataset.conversations
    if config.max_conversations:
        conversations = conversations[: config.max_conversations]

    budget = UsageBudget(
        model=config.model,
        max_total_tokens=config.max_total_tokens,
        max_cost_usd=config.max_cost_usd,
    )
    console.print("Verifying provider credentials...", style="dim")
    preflight_response = await provider.preflight()
    budget.record(preflight_response.input_tokens, preflight_response.output_tokens)

    # Log actual model used (may differ from requested if alias/fallback)
    actual_model = preflight_response.metadata.get("actual_model")
    if actual_model and actual_model != config.model:
        console.print(
            f"[yellow]Warning: Provider is using model '{actual_model}' "
            f"(requested '{config.model}')[/yellow]"
        )
    elif actual_model:
        console.print(f"Using model: {actual_model}", style="dim")

    console.print("Provider OK.", style="dim")

    run_result = RunResult(config=config)
    run_result.preflight_input_tokens = preflight_response.input_tokens
    run_result.preflight_output_tokens = preflight_response.output_tokens
    run_result.actual_model = preflight_response.metadata.get("actual_model")
    start = time.monotonic()

    if budget.stopped:
        run_result.stop_reason = "Budget reached during provider preflight"
        return run_result

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
                budget=budget,
            )
            run_result.conversation_results.append(conv_result)
            if budget.stopped:
                break
    else:
        # Independent mode: each conversation is isolated
        for i, conversation in enumerate(conversations):
            console.print(
                f"  [{i+1}/{len(conversations)}] {conversation.id} "
                f"({len(conversation.turns)} turns)"
                f"{' [dim][batch mode][/dim]' if config.batch else ''}",
                style="dim",
            )

            if config.batch:
                # Batch extraction mode (AlpsBench-style)
                conv_result = await run_batch_conversation(
                    conversation,
                    provider,
                    system_prompt,
                    budget=budget,
                )
            else:
                # Turn-by-turn mode (standard)
                conv_result = await run_conversation(
                    conversation,
                    provider,
                    system_prompt,
                    tools,
                    format_tool_result=memory_system.format_tool_result,
                    budget=budget,
                )

                # Check if system requires final extraction sweep
                if memory_system.requires_final_sweep():
                    sweep_tool_calls = await run_final_sweep(
                        conversation,
                        provider,
                        sweep_prompt=memory_system.final_sweep_prompt(),
                        tools=tools,
                        format_tool_result=memory_system.format_tool_result,
                        budget=budget,
                    )

                    # Add sweep tool calls to the last turn result
                    if sweep_tool_calls and conv_result.turn_results:
                        # Create a synthetic turn result for the sweep
                        sweep_turn = TurnResult(
                            turn_index=len(conversation.turns),
                            role="sweep",
                            tool_calls=sweep_tool_calls,
                            input_tokens=0,  # Already tracked in budget
                            output_tokens=0,
                            latency_ms=0,
                        )
                        conv_result.turn_results.append(sweep_turn)

            run_result.conversation_results.append(conv_result)
            if budget.stopped:
                break

    run_result.total_time_ms = (time.monotonic() - start) * 1000
    if budget.stopped:
        limits = []
        if config.max_total_tokens is not None:
            limits.append(f"{config.max_total_tokens:,} tokens")
        if config.max_cost_usd is not None:
            limits.append(f"${config.max_cost_usd:.4f}")
        run_result.stop_reason = f"Budget reached ({' or '.join(limits)})"
    return run_result

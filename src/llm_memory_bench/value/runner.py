from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..config import ValueRunConfig, get_provider
from ..providers.base import ToolCall
from ..systems import get_system
from ..systems.base import MemorySystem
from .dataset import ValueDataset, ValueScenario


@dataclass
class TurnRecord:
    role: str  # "agent" or "user_sim"
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    integration_events: list[dict] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0


@dataclass
class TrialResult:
    scenario_id: str
    mode: str  # "baseline" | "memory_prompt" | "memory_tool"
    turns: list[TurnRecord] = field(default_factory=list)
    final_response: str = ""
    memory_tokens: int = 0
    system_prompt: str = ""  # NEW: Store the actual system prompt used
    goal_status: str = "incomplete"
    completion_turn: int | None = None
    completion_validations: list[dict] = field(default_factory=list)
    stop_reason: str = ""

    @property
    def agent_turns(self) -> list[TurnRecord]:
        return [t for t in self.turns if t.role == "agent"]

    @property
    def agent_input_tokens(self) -> int:
        return sum(t.input_tokens for t in self.agent_turns)

    @property
    def agent_output_tokens(self) -> int:
        return sum(t.output_tokens for t in self.agent_turns)

    @property
    def total_agent_tokens(self) -> int:
        return self.agent_input_tokens + self.agent_output_tokens

    @property
    def turn_count(self) -> int:
        return len(self.agent_turns)

    @property
    def questions_asked(self) -> int:
        count = 0
        for t in self.agent_turns:
            last_sentence = t.content.rstrip().rsplit(".", 1)[-1].strip()
            if "?" in last_sentence:
                count += 1
        return count


@dataclass
class ScenarioResult:
    scenario_id: str
    trials: dict[str, TrialResult] = field(default_factory=dict)


@dataclass
class ValueRunResult:
    config: ValueRunConfig
    scenario_results: list[ScenarioResult] = field(default_factory=list)
    total_time_ms: float = 0
    metadata: dict = field(default_factory=dict)


USER_SIMULATOR_PROMPT = """\
You are role-playing as a user in a conversation with an AI assistant. \
Stay in character based on the persona below. Answer the assistant's \
questions naturally and concisely using ONLY information from your persona. \
Do not volunteer information the assistant hasn't asked about.

When the assistant has given you a satisfactory, complete response that \
addresses your original request and all success criteria below are met, \
respond with a brief thanks and indicate you're satisfied. If any criterion \
is not met, do not say thanks or claim success; instead answer the assistant's \
question or state what is still missing.

Your private persona context (known to you, but not automatically visible to \
the assistant):
{persona}

Known facts:
{user_facts}

Your original request was: {task_prompt}

Success criteria:
{outcomes}
"""


def _is_terminal(agent_text: str, turn_number: int, max_turns: int) -> bool:
    """Check if agent's response is a complete answer (vs asking for more info)."""
    if turn_number >= max_turns - 1:
        return True
    if len(agent_text.strip()) < 50:
        return False

    text_lower = agent_text.lower()

    # Check if agent is asking for information (questions or requests)
    asking_signals = [
        "?",  # Explicit question
        "let me know",
        "can you tell me",
        "could you tell me",
        "can you provide",
        "could you provide",
        "if you tell me",
        "if you let me know",
        "if you can share",
        "what is your",
        "what are your",
        "where are you",
        "do you have",
        "are you",
        "which",
        "please provide",
        "please share",
        "any preferences",
        "your location",
        "tell me more",
        "more information",
    ]

    if any(signal in text_lower for signal in asking_signals):
        return False  # Agent is asking for more info, continue conversation

    # Also check if response is very generic/hedging (likely incomplete)
    hedge_signals = [
        "i recommend finding",
        "you might want to",
        "you could try",
        "it depends on",
        "would be helpful to know",
    ]

    if any(signal in text_lower for signal in hedge_signals):
        return False  # Generic response, likely needs follow-up

    # If no asking signals, treat as complete response
    return True


def _is_user_satisfied(user_text: str) -> bool:
    lower = user_text.lower()
    satisfaction_signals = [
        "thanks",
        "thank you",
        "that works",
        "that's exactly",
        "that's perfect",
        "perfect",
        "great, thanks",
        "exactly what i needed",
        "appreciate it",
    ]
    return any(signal in lower for signal in satisfaction_signals)


def _should_stop_after_agent(
    agent_text: str,
    turn_number: int,
    max_turns: int,
    goal_mode: bool,
) -> bool:
    """Decide whether a trial may stop after an agent response.

    Goal-mode trials must be driven by outcome validation. Natural-language
    completion heuristics are useful for conversational runs, but can stop a
    baseline before it has actually achieved the goal.
    """
    if turn_number >= max_turns - 1:
        return True
    return not goal_mode and _is_terminal(agent_text, turn_number, max_turns)


SEARCH_TOOL_DEFINITION = {
    "name": "search_web",
    "description": (
        "Search for current information about businesses, services, or facts "
        "you don't have direct knowledge of. Use this when you need real-time "
        "data like restaurant names, shop locations, current prices, or recent information."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query (e.g., 'coffee shops downtown Chongqing')",
            },
            "search_type": {
                "type": "string",
                "enum": ["places", "general"],
                "description": "Type of search - 'places' for businesses/locations, 'general' for facts",
            },
        },
        "required": ["query"],
    },
}


def _simulate_search_result(query: str, search_type: str = "general") -> dict:
    """Simulate search results to demonstrate tool use (prevents hallucination)."""

    query_lower = query.lower()

    # Simulate location-based searches
    if "coffee" in query_lower and "chongqing" in query_lower:
        return {
            "results": [
                {"name": "Seesaw Coffee", "address": "Jiefangbei, Yuzhong District", "rating": "4.5"},
                {"name": "Manner Coffee", "address": "Guanyinqiao, Jiangbei District", "rating": "4.3"},
                {"name": "Mellower Coffee", "address": "Nanping, Nan'an District", "rating": "4.4"},
            ],
            "source": "Simulated search (demo purposes - would use Google Places API)",
        }
    elif "coffee" in query_lower and ("portland" in query_lower or "downtown" in query_lower):
        return {
            "results": [
                {"name": "Stumptown Coffee Roasters", "address": "Downtown", "rating": "4.6"},
                {"name": "Coava Coffee", "address": "SE Portland", "rating": "4.5"},
                {"name": "Heart Coffee", "address": "Downtown", "rating": "4.7"},
            ],
            "source": "Simulated search (demo purposes)",
        }
    else:
        # Generic response for other searches
        return {
            "results": "No specific results available - this is a simulated search tool",
            "suggestion": f"In production, would search for: {query}",
            "source": "Simulated search (demo purposes)",
        }


async def run_value_trial(
    scenario: ValueScenario,
    agent_provider,
    user_provider,
    memory_system: MemorySystem,
    mode: str,
    max_turns: int = 10,
    enable_search: bool = True,  # NEW: Control search tool availability
    completion_provider=None,
) -> TrialResult:
    result = TrialResult(scenario_id=scenario.id, mode=mode)

    base_prompt = memory_system.system_prompt()
    tools: list[dict] = []
    memories_as_dicts = [m.model_dump() for m in scenario.memory_bank]

    # Add search tool to all trials (both baseline and memory)
    if enable_search:
        tools.append(SEARCH_TOOL_DEFINITION)

    if mode == "memory_prompt":
        system_prompt = memory_system.inject_memories(base_prompt, memories_as_dicts)
        prompt_delta = len(system_prompt) - len(base_prompt)
        result.memory_tokens = max(1, prompt_delta // 4)
    elif mode == "memory_tool":
        system_prompt = base_prompt
        tools.extend(memory_system.recall_tool_definitions())
    else:
        system_prompt = base_prompt

    # Store the system prompt for manual review
    result.system_prompt = system_prompt

    user_sim_system = USER_SIMULATOR_PROMPT.format(
        persona=scenario.persona_prompt,
        user_facts="\n".join(
            f"- {memory.fact}"
            for memory in scenario.memory_bank
        ) or "None provided",
        task_prompt=scenario.task_prompt,
        outcomes="\n".join(
            f"- {outcome.id}: {outcome.statement}"
            for outcome in scenario.effective_outcomes()
        ),
    )

    messages: list[dict] = [{"role": "user", "content": scenario.task_prompt}]

    # Record the initial user prompt as turn 0
    result.turns.append(
        TurnRecord(
            role="user",
            content=scenario.task_prompt,
            tool_calls=[],
            input_tokens=0,  # Not charged separately
            output_tokens=0,
            latency_ms=0,
        )
    )

    for turn_num in range(max_turns):
        start = time.monotonic()
        response = await agent_provider.generate(
            messages=messages,
            tools=tools,
            system=system_prompt,
        )
        latency = (time.monotonic() - start) * 1000

        agent_text = response.text

        # Handle tool calls (both search and memory recall)
        if response.tool_calls:
            # Handle search tool calls
            search_calls = [tc for tc in response.tool_calls if tc.name == "search_web"]
            if search_calls:
                # Simulate search results
                search_results = []
                for tc in search_calls:
                    result_data = _simulate_search_result(
                        tc.arguments.get("query", ""),
                        tc.arguments.get("search_type", "general")
                    )
                    search_results.append(result_data)

                # Build tool result messages
                tool_msgs = agent_provider.build_tool_result_messages(
                    response,
                    format_tool_result=lambda tc: _simulate_search_result(
                        tc.arguments.get("query", ""),
                        tc.arguments.get("search_type", "general")
                    )
                )
                messages.extend(tool_msgs)

                # Get agent's follow-up response using search results
                followup_start = time.monotonic()
                followup = await agent_provider.generate(
                    messages=messages,
                    tools=tools,
                    system=system_prompt,
                )
                followup_latency = (time.monotonic() - followup_start) * 1000

                agent_text = followup.text
                response_tokens_in = response.input_tokens + followup.input_tokens
                response_tokens_out = response.output_tokens + followup.output_tokens
                latency += followup_latency

                result.turns.append(
                    TurnRecord(
                        role="agent",
                        content=agent_text,
                        tool_calls=response.tool_calls + followup.tool_calls,
                        input_tokens=response_tokens_in,
                        output_tokens=response_tokens_out,
                        latency_ms=latency,
                    )
                )

                # Completion and ordinary terminal checks happen below.

            # Handle memory recall tool calls (only in memory_tool mode)
            elif mode == "memory_tool":
                recall_names = {t["name"] for t in memory_system.recall_tool_definitions()}
                recall_calls = [tc for tc in response.tool_calls if tc.name in recall_names]
                if recall_calls:
                    recall_result = memory_system.format_recall_result(memories_as_dicts)
                tool_msgs = agent_provider.build_tool_result_messages(
                    response,
                    format_tool_result=lambda _tc: recall_result,
                )
                messages.extend(tool_msgs)

                import json
                result.memory_tokens = max(1, len(json.dumps(recall_result)) // 4)

                followup_start = time.monotonic()
                followup = await agent_provider.generate(
                    messages=messages,
                    tools=tools,
                    system=system_prompt,
                )
                followup_latency = (time.monotonic() - followup_start) * 1000

                agent_text = followup.text
                response_tokens_in = response.input_tokens + followup.input_tokens
                response_tokens_out = response.output_tokens + followup.output_tokens
                latency += followup_latency

                result.turns.append(
                    TurnRecord(
                        role="agent",
                        content=agent_text,
                        tool_calls=response.tool_calls + followup.tool_calls,
                        input_tokens=response_tokens_in,
                        output_tokens=response_tokens_out,
                        latency_ms=latency,
                    )
                )
            else:
                result.turns.append(
                    TurnRecord(
                        role="agent",
                        content=agent_text,
                        tool_calls=response.tool_calls,
                        input_tokens=response.input_tokens,
                        output_tokens=response.output_tokens,
                        latency_ms=latency,
                    )
                )
        else:
            result.turns.append(
                TurnRecord(
                    role="agent",
                    content=agent_text,
                    tool_calls=response.tool_calls,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    latency_ms=latency,
                )
            )

        goal_mode = completion_provider is not None
        if goal_mode:
            from .evaluator import validate_outcomes

            validations: list[dict] = []
            for validator in scenario.effective_validators():
                validations.extend(
                    await validate_outcomes(
                        scenario.task_prompt,
                        agent_text,
                        scenario.effective_outcomes(),
                        validator,
                        completion_provider,
                        memory_facts=(
                            [memory.fact for memory in scenario.memory_bank]
                            if mode != "baseline" else []
                        ),
                        conversation=[
                            {"role": turn.role, "content": turn.content}
                            for turn in result.turns
                        ],
                    )
                )
            result.completion_validations = validations
            if validations and all(v.get("status") == "passed" for v in validations):
                result.goal_status = "complete"
                result.completion_turn = turn_num + 1
                result.final_response = agent_text
                result.stop_reason = "outcomes_passed"
                break

        if _should_stop_after_agent(
            agent_text,
            turn_num,
            max_turns,
            goal_mode=goal_mode,
        ):
            result.final_response = agent_text
            result.stop_reason = "max_turns"
            break

        messages.append({"role": "assistant", "content": agent_text})

        user_start = time.monotonic()
        user_response = await user_provider.generate(
            messages=[
                {"role": "user", "content": f"The assistant said:\n\n{agent_text}\n\nRespond in character."}
            ],
            tools=[],
            system=user_sim_system,
        )
        user_latency = (time.monotonic() - user_start) * 1000

        user_text = user_response.text
        result.turns.append(
            TurnRecord(
                role="user_sim",
                content=user_text,
                input_tokens=user_response.input_tokens,
                output_tokens=user_response.output_tokens,
                latency_ms=user_latency,
            )
        )

        if _is_user_satisfied(user_text) and not goal_mode:
            result.final_response = agent_text
            result.stop_reason = "user_satisfied"
            break

        if _is_user_satisfied(user_text) and goal_mode:
            # The user simulator can still be over-optimistic. Do not feed a
            # satisfaction loop back into the agent; the outcome validator is
            # authoritative for goal completion.
            result.final_response = agent_text
            result.stop_reason = "user_satisfied_before_goal"
            break

        messages.append({"role": "user", "content": user_text})

    if not result.final_response and result.agent_turns:
        result.final_response = result.agent_turns[-1].content

    return result


async def run_value_benchmark(
    dataset: ValueDataset, config: ValueRunConfig
) -> ValueRunResult:
    agent_provider = get_provider(config)

    from ..config import RunConfig

    user_sim_config = RunConfig(
        provider=config.user_simulator_provider,
        model=config.user_simulator_model,
        api_key_env=config.api_key_env,
    )
    user_provider = get_provider(user_sim_config)

    judge_config = RunConfig(
        provider=config.judge_provider,
        model=config.judge_model,
        api_key_env=config.api_key_env,
    )
    completion_provider = (
        get_provider(judge_config) if config.judge_during_run else None
    )

    memory_system = get_system(config.system)

    scenarios = dataset.scenarios
    if config.max_scenarios:
        scenarios = scenarios[: config.max_scenarios]

    modes = []
    if config.injection_mode == "both":
        modes = ["baseline", "memory_prompt", "memory_tool"]
    elif config.injection_mode == "prompt":
        modes = ["baseline", "memory_prompt"]
    elif config.injection_mode == "tool":
        modes = ["baseline", "memory_tool"]
    else:
        modes = ["baseline", config.injection_mode]

    run_result = ValueRunResult(config=config)
    start = time.monotonic()

    for scenario in scenarios:
        scenario_result = ScenarioResult(scenario_id=scenario.id)

        for mode in modes:
            trial = await run_value_trial(
                scenario=scenario,
                agent_provider=agent_provider,
                user_provider=user_provider,
                memory_system=memory_system,
                mode=mode,
                max_turns=config.max_turns,
                enable_search=True,  # Enable search tool for grounding
                completion_provider=completion_provider,
            )
            scenario_result.trials[mode] = trial

        run_result.scenario_results.append(scenario_result)

    run_result.total_time_ms = (time.monotonic() - start) * 1000
    return run_result

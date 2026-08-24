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


USER_SIMULATOR_PROMPT = """\
You are role-playing as a user in a conversation with an AI assistant. \
Stay in character based on the persona below. Answer the assistant's \
questions naturally and concisely using ONLY information from your persona. \
Do not volunteer information the assistant hasn't asked about.

When the assistant has given you a satisfactory, complete response that \
addresses your original request, respond with a brief thanks and indicate \
you're satisfied (e.g., "Thanks, that's exactly what I needed!" or \
"Perfect, that works for me.").

Your persona:
{persona}

Your original request was: {task_prompt}"""


def _is_terminal(agent_text: str, turn_number: int, max_turns: int) -> bool:
    if turn_number >= max_turns - 1:
        return True
    if len(agent_text.strip()) < 50:
        return False
    last_line = agent_text.rstrip().split("\n")[-1]
    if "?" not in last_line:
        return True
    return False


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


async def run_value_trial(
    scenario: ValueScenario,
    agent_provider,
    user_provider,
    memory_system: MemorySystem,
    mode: str,
    max_turns: int = 10,
) -> TrialResult:
    result = TrialResult(scenario_id=scenario.id, mode=mode)

    base_prompt = memory_system.system_prompt()
    tools: list[dict] = []
    memories_as_dicts = [m.model_dump() for m in scenario.memory_bank]

    if mode == "memory_prompt":
        system_prompt = memory_system.inject_memories(base_prompt, memories_as_dicts)
        prompt_delta = len(system_prompt) - len(base_prompt)
        result.memory_tokens = max(1, prompt_delta // 4)
    elif mode == "memory_tool":
        system_prompt = base_prompt
        tools = memory_system.recall_tool_definitions()
    else:
        system_prompt = base_prompt

    user_sim_system = USER_SIMULATOR_PROMPT.format(
        persona=scenario.persona_prompt,
        task_prompt=scenario.task_prompt,
    )

    messages: list[dict] = [{"role": "user", "content": scenario.task_prompt}]

    for turn_num in range(max_turns):
        start = time.monotonic()
        response = await agent_provider.generate(
            messages=messages,
            tools=tools,
            system=system_prompt,
        )
        latency = (time.monotonic() - start) * 1000

        agent_text = response.text

        if response.tool_calls and mode == "memory_tool":
            recall_calls = [tc for tc in response.tool_calls if tc.name == "recall_memory"]
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

        if _is_terminal(agent_text, turn_num, max_turns):
            result.final_response = agent_text
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

        if _is_user_satisfied(user_text):
            result.final_response = agent_text
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
            )
            scenario_result.trials[mode] = trial

        run_result.scenario_results.append(scenario_result)

    run_result.total_time_ms = (time.monotonic() - start) * 1000
    return run_result

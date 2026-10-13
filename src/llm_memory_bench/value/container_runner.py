"""Value benchmark execution through a real containerised agent.

This module deliberately does not reuse the API-level memory adapters.  The
baseline runs Claude Code without a memory integration; the memory runs seed
the external provider and then execute Claude Code with the materialized
integration profile.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..bench import (
    RunConfig as BenchRunConfig,
    build_host_image,
    drive_conversation,
    install_integration,
    load_integration,
    materialize_integration,
    resolved_system_environment,
    start_container,
    stop_container,
    validate_integration_environment,
)
from ..config import RunConfig, ValueRunConfig, get_provider
from ..providers.base import ToolCall
from ..seeding import (
    SeedError,
    SeedMemory,
    SeedPlan,
    cleanup_benchmark_data,
    seed_benchmark_data,
)
from .dataset import ValueDataset, ValueScenario
from .evaluator import validate_outcomes
from .runner import (
    USER_SIMULATOR_PROMPT,
    ScenarioResult,
    TrialResult,
    TurnRecord,
    ValueRunResult,
    _is_terminal,
    _is_user_satisfied,
    _should_stop_after_agent,
)


def _scenario_seed_plan(scenario: ValueScenario, provider: str) -> SeedPlan:
    """Convert a value scenario's memory bank into a benchmark seed plan."""

    memories = tuple(
        SeedMemory(
            content=entry.fact,
            # Value scenarios are isolated run fixtures.  A user-scoped seed
            # would be visible to every later benchmark using the same
            # identity, even when project_id is supplied to the API.
            scope="project",
            metadata={"category": entry.category, "source": "value-scenario"},
            # The memory bank is an explicit benchmark fixture. MemoryHub's
            # duplicate gate is global enough that an identical fact can be
            # rejected even under a fresh project_id; force materialization
            # keeps the fixture present in this benchmark project.
            force=True,
        )
        for entry in scenario.memory_bank
    )
    canonical = json.dumps(
        {
            "scenario": scenario.id,
            "provider": provider,
            "memories": [
                {"content": memory.content, "scope": memory.scope}
                for memory in memories
            ],
        },
        sort_keys=True,
    )
    return SeedPlan(
        path=Path(f"<value-scenario:{scenario.id}>"),
        api_version="llm-memory-bench.dev/v1",
        seed_id=f"value-{scenario.id}",
        version="1.0.0",
        provider=provider,
        mode="declarative",
        memories=memories,
        fixture_hash=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    )


def _new_project_id(scenario_id: str) -> str:
    safe_id = "".join(char if char.isalnum() else "-" for char in scenario_id)
    return f"llm-bench-value-{safe_id}-{uuid.uuid4().hex[:10]}"


def _parse_agent_response(
    raw: str,
) -> tuple[str, str | None, dict[str, int], list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        preview = raw[:1000].replace("\n", "\\n")
        raise RuntimeError(
            f"Container driver returned invalid JSON: {exc}; stdout={preview!r}"
        ) from exc

    responses = payload.get("responses")
    if not isinstance(responses, list) or not responses:
        raise RuntimeError(f"Container driver returned no response: {raw[:500]}")
    response = responses[-1]
    if not isinstance(response, dict):
        raise RuntimeError("Container driver returned an invalid response record")
    if response.get("error"):
        raise RuntimeError(str(response["error"]))

    usage = response.get("usage", {})
    if not isinstance(usage, dict):
        usage = {}
    input_tokens = usage.get("input_tokens", usage.get("input", 0))
    output_tokens = usage.get("output_tokens", usage.get("output", 0))
    tool_calls = response.get("tool_calls", [])
    if not isinstance(tool_calls, list):
        tool_calls = []
    integration_events = response.get("integration_events", [])
    if not isinstance(integration_events, list):
        integration_events = []
    return (
        str(response.get("response", "")),
        response.get("session_id") or payload.get("session_id"),
        {
            "input_tokens": int(input_tokens) if isinstance(input_tokens, (int, float)) else 0,
            "output_tokens": int(output_tokens) if isinstance(output_tokens, (int, float)) else 0,
        },
        [call for call in tool_calls if isinstance(call, dict)],
        [event for event in integration_events if isinstance(event, dict)],
    )


def _baseline_config(config: BenchRunConfig) -> BenchRunConfig:
    """Remove provider integration credentials from the no-memory trial."""

    filtered_env = {
        key: value
        for key, value in config.system_env.items()
        if not key.startswith("MEMORYHUB_")
    }
    return replace(
        config,
        system_name="baseline",
        system_env=filtered_env,
        integration_profile=None,
        integration_platform=None,
    )


async def _run_container_trial(
    scenario: ValueScenario,
    config: BenchRunConfig,
    container_id: str,
    user_provider,
    mode: str,
    max_turns: int,
    profile_mode: bool,
    completion_provider,
    memory_tokens: int = 0,
) -> TrialResult:
    result = TrialResult(scenario_id=scenario.id, mode=mode)
    user_sim_system = USER_SIMULATOR_PROMPT.format(
        persona=scenario.persona_prompt,
        user_facts="\n".join(f"- {memory.fact}" for memory in scenario.memory_bank)
        or "None provided",
        task_prompt=scenario.task_prompt,
        outcomes="\n".join(
            f"- {outcome.id}: {outcome.statement}"
            for outcome in scenario.effective_outcomes()
        ),
    )
    result.memory_tokens = memory_tokens
    result.turns.append(TurnRecord(role="user", content=scenario.task_prompt))

    session_id: str | None = None
    prompt = scenario.task_prompt
    goal_mode = completion_provider is not None
    seen_event_ids: set[str] = set()

    for turn_number in range(max_turns):
        raw = drive_conversation(
            container_id,
            _single_user_turn(scenario.id, prompt),
            model=config.model,
            profile_mode=profile_mode,
            session_id=session_id,
        )
        (
            agent_text,
            session_id,
            usage,
            raw_tool_calls,
            raw_integration_events,
        ) = _parse_agent_response(raw)
        integration_events = []
        for event in raw_integration_events:
            event_id = str(event.get("id", ""))
            event_key = event_id or json.dumps(event, sort_keys=True)
            if event_key in seen_event_ids:
                continue
            seen_event_ids.add(event_key)
            integration_events.append(event)
        tool_calls = [
            ToolCall(
                name=str(call.get("name", "unknown")),
                arguments=(
                    call.get("arguments", {})
                    if isinstance(call.get("arguments", {}), dict)
                    else {}
                ),
            )
            for call in raw_tool_calls
        ]
        result.turns.append(
            TurnRecord(
                role="agent",
                content=agent_text,
                tool_calls=tool_calls,
                integration_events=integration_events,
                input_tokens=usage["input_tokens"],
                output_tokens=usage["output_tokens"],
            )
        )

        if completion_provider is not None:
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
                            if mode != "baseline"
                            else []
                        ),
                        conversation=[
                            {"role": turn.role, "content": turn.content}
                            for turn in result.turns
                        ],
                    )
                )
            result.completion_validations = validations
            if validations and all(item.get("status") == "passed" for item in validations):
                result.goal_status = "complete"
                result.completion_turn = turn_number + 1
                result.final_response = agent_text
                result.stop_reason = "outcomes_passed"
                break

        if _should_stop_after_agent(
            agent_text,
            turn_number,
            max_turns,
            goal_mode=goal_mode,
        ):
            result.final_response = agent_text
            result.stop_reason = "max_turns"
            break

        user_response = await user_provider.generate(
            messages=[
                {
                    "role": "user",
                    "content": f"The assistant said:\n\n{agent_text}\n\nRespond in character.",
                }
            ],
            tools=[],
            system=user_sim_system,
        )
        user_text = user_response.text
        result.turns.append(
            TurnRecord(
                role="user_sim",
                content=user_text,
                input_tokens=user_response.input_tokens,
                output_tokens=user_response.output_tokens,
                latency_ms=0,
            )
        )
        if _is_user_satisfied(user_text) and not goal_mode:
            result.final_response = agent_text
            result.stop_reason = "user_satisfied"
            break
        if _is_user_satisfied(user_text) and goal_mode:
            result.final_response = agent_text
            result.stop_reason = "user_satisfied_before_goal"
            break
        prompt = user_text

    if not result.final_response and result.agent_turns:
        result.final_response = result.agent_turns[-1].content
    return result


def _single_user_turn(scenario_id: str, prompt: str):
    from ..dataset import Conversation, Turn

    return Conversation(
        id=f"{scenario_id}-value-turn",
        turns=[Turn(role="user", content=prompt)],
    )


async def run_container_value_benchmark(
    dataset: ValueDataset,
    container_config: BenchRunConfig,
    value_config: ValueRunConfig,
) -> ValueRunResult:
    """Run paired value trials against the real agent and integration."""

    profile = load_integration(container_config)
    if profile is None:
        raise RuntimeError("Container value runs require integration.profile")
    profile_provider = str(profile.memory_system.get("name", "")).lower()
    if profile_provider != "memoryhub":
        raise RuntimeError(
            "The first container value runner supports the MemoryHub seed adapter only"
        )

    validate_integration_environment(container_config, profile)
    build_host_image(container_config)

    user_sim_config = RunConfig(
        provider=value_config.user_simulator_provider,
        model=value_config.user_simulator_model,
        api_key_env=value_config.api_key_env,
    )
    user_provider = get_provider(user_sim_config)
    judge_config = RunConfig(
        provider=value_config.judge_provider,
        model=value_config.judge_model,
        api_key_env=value_config.api_key_env,
    )
    completion_provider = (
        get_provider(judge_config) if value_config.judge_during_run else None
    )

    scenarios = dataset.scenarios
    if value_config.max_scenarios:
        scenarios = scenarios[: value_config.max_scenarios]

    run_result = ValueRunResult(config=value_config)
    run_result.metadata = {
        "execution": "container",
        "host": {
            "name": container_config.host_name,
            "version": container_config.host_version,
        },
        "model": container_config.model,
        "integration_profile": profile.profile_id,
        "scenarios": {},
    }
    start = time.monotonic()
    for scenario in scenarios:
        scenario_result = ScenarioResult(scenario_id=scenario.id)

        baseline_container = start_container(
            _baseline_config(container_config),
            profile=None,
            legacy_install=False,
        )
        try:
            baseline_config = _baseline_config(container_config)
            scenario_result.trials["baseline"] = await _run_container_trial(
                scenario,
                baseline_config,
                baseline_container,
                user_provider,
                "baseline",
                value_config.max_turns,
                False,
                completion_provider,
            )
        finally:
            stop_container(baseline_container, keep=container_config.keep_container)

        project_id = _new_project_id(scenario.id)
        workspace_tmp = __import__("tempfile").TemporaryDirectory(
            prefix="llm-bench-value-integration-"
        )
        memory_container: str | None = None
        seeded = False
        seed_result = None
        seed_environment = resolved_system_environment(container_config)
        try:
            effective_hash = materialize_integration(
                container_config,
                profile,
                workspace_tmp.name,
                project_id,
            )
            seed_plan = _scenario_seed_plan(scenario, profile_provider)
            try:
                # The agent runs in Docker, while benchmark seeding runs on
                # the host. Docker Desktop's host.docker.internal name is not
                # resolvable from the host process itself.
                if seed_environment.get("MEMORYHUB_SEED_URL"):
                    seed_environment["MEMORYHUB_URL"] = seed_environment[
                        "MEMORYHUB_SEED_URL"
                    ]
                # The seed adapter is synchronous but uses asyncio.run() for
                # MemoryHub's async client. Run it in a worker thread because
                # this benchmark function already owns an event loop.
                seed_result = await asyncio.to_thread(
                    seed_benchmark_data,
                    seed_plan,
                    project_id,
                    seed_environment,
                )
                seeded = True
            except SeedError as exc:
                raise RuntimeError(f"Value benchmark seed failed: {exc}") from exc

            memory_container = start_container(
                container_config,
                profile=profile,
                project_id=project_id,
            )
            try:
                install_integration(memory_container, workspace_tmp.name, profile)
                memory_config = container_config
                memory_tokens = max(
                    1,
                    sum(len(memory.content) for memory in seed_plan.memories) // 4,
                )
                trial = await _run_container_trial(
                    scenario,
                    memory_config,
                    memory_container,
                    user_provider,
                    "memory_integration",
                    value_config.max_turns,
                    True,
                    completion_provider,
                    memory_tokens=memory_tokens,
                )
                scenario_result.trials["memory_integration"] = trial
                run_result.metadata["scenarios"][scenario.id] = {
                    "project_id": project_id,
                    "effective_integration_hash": effective_hash,
                    "seed": {
                        "plan": seed_plan.plan_id,
                        "fixture_hash": seed_plan.fixture_hash,
                        "count": len(seed_plan.memories),
                    },
                }
            finally:
                if memory_container is not None:
                    stop_container(
                        memory_container,
                        keep=container_config.keep_container,
                    )
                if seeded:
                    try:
                        await asyncio.to_thread(
                            cleanup_benchmark_data,
                            profile_provider,
                            project_id,
                            seed_environment,
                            seed_result,
                        )
                    except SeedError as exc:
                        raise RuntimeError(
                            f"Value benchmark cleanup failed: {exc}"
                        ) from exc
        finally:
            workspace_tmp.cleanup()

        run_result.scenario_results.append(scenario_result)

    run_result.total_time_ms = (time.monotonic() - start) * 1000
    return run_result

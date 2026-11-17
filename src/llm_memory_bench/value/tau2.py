"""Run the real τ²-bench environment as a paired memory-value experiment.

The τ² environment and evaluator remain authoritative.  This module only
owns the experiment arm, memory prompt, telemetry, pairing, and result shape.
The optional ``tau2-bench`` dependency is imported lazily so the normal
benchmark suite does not require τ² to be installed.
"""

from __future__ import annotations

import time
import types
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from ..systems import get_system
from ..providers.base import ToolCall as HarnessToolCall
from .dataset import MemoryEntry


class Tau2UnavailableError(RuntimeError):
    """Raised when a τ² run is requested without the optional dependency."""


TAU2_MEMORYHUB_PROMPT_VERSION = "v3_ben_capture_only_2026-11"


class Tau2TaskMemory(BaseModel):
    """Memory fixture for one official τ² task ID."""

    task_id: str
    memory_bank: list[MemoryEntry] = Field(default_factory=list)


class Tau2MemoryPlan(BaseModel):
    """Mapping from official τ² task IDs to memories used by the memory arm."""

    domain: str
    tasks: list[Tau2TaskMemory]
    source_revision: str | None = None

    def for_task(self, task_id: str) -> Tau2TaskMemory:
        for task in self.tasks:
            if task.task_id == str(task_id):
                return task
        raise ValueError(
            f"No memory fixture for τ² task {task_id!r} in domain {self.domain!r}"
        )


@dataclass
class Tau2MemoryBridge:
    """Bridge a harness memory system into one τ² arm.

    τ² owns the domain environment and evaluator.  This small bridge lets a
    selected benchmark memory system expose its normal tool schema while the
    harness records the writes and returns the system's normal simulated
    response.  The captured snapshot is copied into the paired replay arm.
    """

    memory_system: Any
    mode: Literal["disabled", "capture_only", "replay"] = "replay"
    initial_memories: list[dict[str, Any]] = field(default_factory=list)
    captured_memories: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def tool_names(self) -> set[str]:
        if self.mode == "disabled":
            return set()
        return {
            definition["name"]
            for definition in self.memory_system.tool_definitions_for_mode(self.mode)
        }

    def tools(self) -> list[Any]:
        """Return τ²-compatible tool objects from the system definitions."""

        if self.mode == "disabled":
            return []

        @dataclass(frozen=True)
        class SchemaTool:
            name: str
            description: str
            input_schema: dict[str, Any]

            @property
            def openai_schema(self) -> dict[str, Any]:
                return {
                    "type": "function",
                    "function": {
                        "name": self.name,
                        "description": self.description,
                        "parameters": self.input_schema,
                    },
                }

        return [
            SchemaTool(
                name=definition["name"],
                description=definition.get("description", ""),
                input_schema=definition.get("input_schema", {}),
            )
            for definition in self.memory_system.tool_definitions_for_mode(self.mode)
        ]

    def handle(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        call = HarnessToolCall(name=name, arguments=arguments)
        validate = getattr(self.memory_system, "validate_tool_call_for_mode", None)
        if validate is not None:
            error = validate(call, self.mode)
            if error:
                self.events.append(
                    {"tool": name, "arguments": arguments, "error": error}
                )
                return {"status": "error", "error": error}
        fact = self.memory_system.extract_stored_fact(call)
        event = {"tool": name, "arguments": arguments, "stored_fact": fact or None}
        executor = getattr(self.memory_system, "execute_tool_call", None)
        if executor is not None:
            response = executor(call, self.mode)
            event["response"] = response
        else:
            response = self.memory_system.format_tool_result(call)
        self.events.append(event)
        if fact and response.get("status") != "error":
            entry = {
                "fact": fact,
                "category": "captured",
                "source": "baseline_conversation",
                "tool": name,
            }
            if entry not in self.captured_memories:
                self.captured_memories.append(entry)
        return response


def _attach_memory_bridge(environment: Any, bridge: Tau2MemoryBridge) -> None:
    """Route memory tool calls through the harness-to-τ² bridge."""

    original_use_tool = environment.use_tool

    def use_tool(self: Any, tool_name: str, **kwargs: Any) -> Any:
        if tool_name in bridge.tool_names:
            return bridge.handle(tool_name, kwargs)
        return original_use_tool(tool_name, **kwargs)

    environment.use_tool = types.MethodType(use_tool, environment)


@dataclass
class Tau2AgentTelemetry:
    """Harness-owned instrumentation for a single τ² agent arm."""

    llm_calls: int = 0
    tool_calls: int = 0
    input_chars: int = 0
    output_chars: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    usage_records: int = 0
    usage_unavailable: int = 0
    latency_ms: float = 0.0
    memory_chars: int = 0

    @property
    def estimated_input_tokens(self) -> int:
        return max(0, self.input_chars // 4)

    @property
    def estimated_output_tokens(self) -> int:
        return max(0, self.output_chars // 4)

    @property
    def token_usage_exact(self) -> bool:
        return self.usage_records > 0 and self.usage_unavailable == 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_calls,
            "input_chars": self.input_chars,
            "output_chars": self.output_chars,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "token_usage_exact": self.token_usage_exact,
            "token_usage_records": self.usage_records,
            "token_usage_unavailable": self.usage_unavailable,
            "estimated_input_tokens": self.estimated_input_tokens,
            "estimated_output_tokens": self.estimated_output_tokens,
            "estimated_total_tokens": self.estimated_input_tokens
            + self.estimated_output_tokens,
            "latency_ms": round(self.latency_ms, 2),
            "memory_chars": self.memory_chars,
            "token_usage_note": (
                "provider-reported LiteLLM usage"
                if self.token_usage_exact
                else "estimated as characters / 4 when provider usage is unavailable"
            ),
        }


@dataclass
class Tau2ArmResult:
    task_id: str
    mode: Literal["baseline", "memory_prompt"]
    reward: float | None
    telemetry: Tau2AgentTelemetry
    system_prompt: str
    simulation: dict[str, Any]
    initial_memories: list[dict[str, Any]] = field(default_factory=list)
    captured_memories: list[dict[str, Any]] = field(default_factory=list)
    memory_events: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "mode": self.mode,
            "reward": self.reward,
            "telemetry": self.telemetry.as_dict(),
            "system_prompt": self.system_prompt,
            "simulation": self.simulation,
            "initial_memories": self.initial_memories,
            "captured_memories": self.captured_memories,
            "memory_events": self.memory_events,
        }


def load_tau2_memory_plan(path: Path | str) -> Tau2MemoryPlan:
    """Load a task-to-memory mapping without importing τ²."""

    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        return Tau2MemoryPlan.model_validate(yaml.safe_load(handle))


def _require_tau2() -> dict[str, Any]:
    try:
        from tau2.agent.base_agent import HalfDuplexAgent
        from tau2.data_model.message import (
            AssistantMessage,
            Message,
            MultiToolMessage,
            SystemMessage,
        )
        from tau2.evaluator.evaluator import EvaluationType
        from tau2.orchestrator.orchestrator import Orchestrator
        from tau2.runner import build_environment, build_user, get_tasks, run_simulation
        from tau2.utils.llm_utils import generate
    except ImportError as exc:
        raise Tau2UnavailableError(
            "τ²-bench could not be imported. Install its pinned upstream "
            f"package and dependencies ({exc})."
        ) from exc

    return {
        "HalfDuplexAgent": HalfDuplexAgent,
        "AssistantMessage": AssistantMessage,
        "Message": Message,
        "MultiToolMessage": MultiToolMessage,
        "SystemMessage": SystemMessage,
        "EvaluationType": EvaluationType,
        "Orchestrator": Orchestrator,
        "build_environment": build_environment,
        "build_user": build_user,
        "get_tasks": get_tasks,
        "run_simulation": run_simulation,
        "generate": generate,
    }


def _message_chars(message: Any) -> int:
    content = getattr(message, "content", None)
    if not content:
        return 0
    return len(str(content))


def _make_agent(tau2: dict[str, Any], telemetry: Tau2AgentTelemetry, *,
                tools: list[Any], domain_policy: str, model: str,
                memory_system: Any, memory_facts: list[dict[str, Any]],
                memory_mode: Literal["capture_only", "replay"] = "replay"):
    """Build a τ² HalfDuplexAgent with an optional harness memory prompt."""

    HalfDuplexAgent = tau2["HalfDuplexAgent"]
    AssistantMessage = tau2["AssistantMessage"]
    Message = tau2["Message"]
    MultiToolMessage = tau2["MultiToolMessage"]
    SystemMessage = tau2["SystemMessage"]
    generate = tau2["generate"]

    base_prompt = (
        "You are a helpful customer-service agent.\n\n"
        f"## Domain policy\n{domain_policy}\n\n"
        "Follow the policy strictly and use the provided domain tools when needed."
    )
    if memory_system is not None:
        memory_prompt = memory_system.system_prompt_for_mode(memory_mode).strip()
        if memory_prompt:
            base_prompt += f"\n\n## Memory integration\n{memory_prompt}"
    if memory_facts:
        prompt = memory_system.inject_memories(base_prompt, memory_facts)
        telemetry.memory_chars = max(0, len(prompt) - len(base_prompt))
    else:
        prompt = base_prompt

    class HarnessAgent(HalfDuplexAgent[list[Message]]):
        def __init__(self):
            super().__init__(tools=tools, domain_policy=domain_policy)
            self._system_messages = [SystemMessage(role="system", content=prompt)]

        def get_init_state(
            self, message_history: list[Message] | None = None
        ) -> list[Message]:
            return list(message_history) if message_history else []

        def generate_next_message(
            self, message: Message, state: list[Message]
        ) -> tuple[AssistantMessage, list[Message]]:
            # The orchestrator wraps multiple tool results in a
            # MultiToolMessage.  LiteLLM's τ² message converter expects each
            # ToolMessage separately, so flatten the wrapper before sending
            # the next request.  Appending the wrapper itself silently drops
            # all results and leaves orphaned assistant tool-call IDs in the
            # next OpenAI request.
            if isinstance(message, MultiToolMessage):
                state.extend(message.tool_messages)
            else:
                state.append(message)
            started = time.monotonic()
            response = generate(
                model=model,
                tools=self.tools,
                messages=self._system_messages + state,
            )
            telemetry.llm_calls += 1
            telemetry.input_chars += sum(_message_chars(item) for item in state)
            telemetry.output_chars += _message_chars(response)
            telemetry.tool_calls += len(getattr(response, "tool_calls", []) or [])
            usage = getattr(response, "usage", None)
            if (
                isinstance(usage, dict)
                and isinstance(usage.get("prompt_tokens"), int)
                and isinstance(usage.get("completion_tokens"), int)
            ):
                telemetry.input_tokens += usage["prompt_tokens"]
                telemetry.output_tokens += usage["completion_tokens"]
                telemetry.usage_records += 1
            else:
                telemetry.usage_unavailable += 1
            telemetry.latency_ms += (time.monotonic() - started) * 1000
            state.append(response)
            return response, state

    return HarnessAgent(), prompt


def _run_tau2_arm(
    tau2: dict[str, Any],
    *,
    domain: str,
    task: Any,
    mode: Literal["baseline", "memory_prompt"],
    memory_system: Any,
    memory_facts: list[dict[str, Any]],
    memory_bridge: Tau2MemoryBridge,
    agent_model: str,
    user_model: str,
    seed: int,
    max_steps: int,
) -> Tau2ArmResult:
    environment = tau2["build_environment"](domain)
    _attach_memory_bridge(environment, memory_bridge)
    telemetry = Tau2AgentTelemetry()
    agent_tools = environment.get_tools() + memory_bridge.tools()
    agent, system_prompt = _make_agent(
        tau2,
        telemetry,
        tools=agent_tools,
        domain_policy=environment.get_policy(),
        model=agent_model,
        memory_system=memory_system,
        memory_facts=memory_facts if mode == "memory_prompt" else [],
        memory_mode=memory_bridge.mode,
    )
    user = tau2["build_user"](
        "user_simulator",
        environment,
        task,
        llm=user_model,
    )
    orchestrator = tau2["Orchestrator"](
        domain=domain,
        agent=agent,
        user=user,
        environment=environment,
        task=task,
        max_steps=max_steps,
        max_errors=5,
        seed=seed,
    )
    simulation = tau2["run_simulation"](
        orchestrator,
        evaluation_type=tau2["EvaluationType"].ALL,
    )
    simulation_data = simulation.model_dump(mode="json")
    reward_info = simulation_data.get("reward_info") or {}
    return Tau2ArmResult(
        task_id=str(task.id),
        mode=mode,
        reward=reward_info.get("reward"),
        telemetry=telemetry,
        system_prompt=system_prompt,
        simulation=simulation_data,
        initial_memories=(
            list(memory_bridge.initial_memories)
            if memory_bridge.initial_memories
            else list(memory_facts)
        ),
        captured_memories=list(memory_bridge.captured_memories),
        memory_events=list(memory_bridge.events),
    )


def run_tau2_value_benchmark(
    plan: Tau2MemoryPlan,
    *,
    task_ids: list[str],
    agent_model: str = "openai/gpt-4.1",
    user_model: str = "openai/gpt-4.1",
    memory_system: str = "simple",
    memory_setup_mode: Literal["seeded", "capture_first"] = "capture_first",
    seed: int = 300,
    max_steps: int = 20,
    memoryhub_live: bool = False,
    memoryhub_cleanup: bool = True,
) -> dict[str, Any]:
    """Run matched real-τ² baseline and memory-prompt arms."""

    tau2 = _require_tau2()
    if memory_setup_mode not in {"seeded", "capture_first"}:
        raise ValueError(
            "memory_setup_mode must be 'seeded' or 'capture_first'"
        )
    if plan.domain not in {"airline", "retail", "telecom"}:
        raise ValueError(f"Unsupported τ² domain: {plan.domain}")
    if not task_ids:
        raise ValueError("At least one τ² task ID is required")

    tasks = tau2["get_tasks"](plan.domain, task_ids=[str(task_id) for task_id in task_ids])
    tasks_by_id = {str(task.id): task for task in tasks}
    missing = [str(task_id) for task_id in task_ids if str(task_id) not in tasks_by_id]
    if missing:
        raise ValueError(f"τ² tasks not found in {plan.domain}: {', '.join(missing)}")

    memory_kwargs = {}
    if memory_system in {"memoryhub", "memoryhub_sweep"}:
        memory_kwargs["prompt_version"] = TAU2_MEMORYHUB_PROMPT_VERSION
    memory = get_system(memory_system, **memory_kwargs)
    if memoryhub_live:
        if memory_system not in {"memoryhub", "memoryhub_sweep"}:
            raise ValueError(
                "memoryhub_live requires --memory-system memoryhub or "
                "memoryhub_sweep"
            )
        from .memoryhub_live import MemoryHubLiveAdapter

        MemoryHubLiveAdapter.ensure_available()
    scenarios: list[dict[str, Any]] = []
    started = time.monotonic()
    for task_id in task_ids:
        task = tasks_by_id[str(task_id)]
        fixture = plan.for_task(str(task_id))
        fixture_memory = [entry.model_dump(mode="json") for entry in fixture.memory_bank]
        baseline_live = None
        replay_live = None
        if memoryhub_live:
            from .memoryhub_live import MemoryHubLiveAdapter

            project_prefix = (
                f"llm-bench-tau2-{plan.domain}-{task_id}-"
                f"{uuid.uuid4().hex[:8]}"
            )
            replay_live = MemoryHubLiveAdapter(
                project_id=f"{project_prefix}-replay",
                prompt_version=TAU2_MEMORYHUB_PROMPT_VERSION,
            )
            if memory_setup_mode == "capture_first":
                baseline_live = MemoryHubLiveAdapter(
                    project_id=f"{project_prefix}-baseline",
                    prompt_version=TAU2_MEMORYHUB_PROMPT_VERSION,
                )
            baseline_system = baseline_live
        else:
            baseline_system = memory
        if memory_setup_mode == "seeded":
            baseline_bridge = Tau2MemoryBridge(memory, mode="disabled")
            baseline_system = None
        else:
            baseline_bridge = Tau2MemoryBridge(baseline_system, mode="capture_only")
        try:
            baseline = _run_tau2_arm(
                tau2,
                domain=plan.domain,
                task=task,
                mode="baseline",
                memory_system=baseline_system,
                memory_facts=[],
                memory_bridge=baseline_bridge,
                agent_model=agent_model,
                user_model=user_model,
                seed=seed,
                max_steps=max_steps,
            )
            replay_system = replay_live or memory
            if memory_setup_mode == "seeded":
                handoff_memories = fixture_memory
            else:
                handoff_memories = list(baseline_bridge.captured_memories)
            replay_bridge = Tau2MemoryBridge(
                replay_system,
                mode="replay",
                initial_memories=list(handoff_memories),
            )
            if replay_live is not None:
                replay_bridge.initial_memories = replay_live.seed_memories(
                    list(handoff_memories)
                )
            memory_arm = _run_tau2_arm(
                tau2,
                domain=plan.domain,
                task=task,
                mode="memory_prompt",
                memory_system=replay_system,
                memory_facts=list(handoff_memories),
                memory_bridge=replay_bridge,
                agent_model=agent_model,
                user_model=user_model,
                seed=seed,
                max_steps=max_steps,
            )
        finally:
            if memoryhub_live and memoryhub_cleanup:
                for adapter in (replay_live, baseline_live):
                    if adapter is not None:
                        adapter.cleanup()
        scenarios.append(
            {
                "task_id": str(task_id),
                # Kept for compatibility and provenance only.  These fixture
                # facts are deliberately not preloaded into either arm.
                "memory_bank": fixture_memory,
                "memory_source": memory_setup_mode,
                "fixture_memory_bank_used": memory_setup_mode == "seeded",
                "memory_backend": "memoryhub_live" if memoryhub_live else "harness_simulated",
                "memoryhub_projects": (
                    {
                        "baseline": baseline_live.project_id,
                        "replay": replay_live.project_id,
                        "cleaned_up": memoryhub_cleanup,
                    }
                    if memoryhub_live and baseline_live and replay_live
                    else None
                ),
                "baseline": baseline.as_dict(),
                "memory_prompt": memory_arm.as_dict(),
                "reward_delta": (
                    memory_arm.reward - baseline.reward
                    if memory_arm.reward is not None and baseline.reward is not None
                    else None
                ),
            }
        )

    return {
        "execution": "tau2",
        "benchmark": {
            "name": "sierra-research/tau2-bench",
            "source_revision": plan.source_revision,
            "domain": plan.domain,
            "task_ids": [str(task_id) for task_id in task_ids],
        },
        "config": {
            "agent_model": agent_model,
            "user_model": user_model,
            "memory_system": memory_system,
            "memory_system_info": memory.version_info(),
            "memory_flow": (
                "scenario_seed_to_memory_arm"
                if memory_setup_mode == "seeded"
                else "cold_start_baseline_capture_to_replay"
            ),
            "memory_setup_mode": memory_setup_mode,
            "memoryhub_live": memoryhub_live,
            "memoryhub_cleanup": memoryhub_cleanup if memoryhub_live else None,
            "seed": seed,
            "max_steps": max_steps,
        },
        "scenarios": scenarios,
        "total_time_ms": round((time.monotonic() - started) * 1000, 2),
    }

"""Process-isolated CL-Bench worker used by :mod:`value.clbench`."""

from __future__ import annotations

import dataclasses
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, create_model


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {key: _jsonable(item) for key, item in dataclasses.asdict(value).items()}
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


class MemoryWrite(BaseModel):
    fact: str = Field(description="One durable fact learned from the interaction.")
    category: str = Field(default="general", description="Fact category.")


def _prepare_clbench_imports(root: Path) -> None:
    """Prefer the official checkout over this repository's ``src`` namespace."""

    root = root.resolve()
    sys.path.insert(0, str(root))
    package = sys.modules.get("src")
    if package is not None:
        package_path = str(root / "src")
        search_path = getattr(package, "__path__", None)
        if search_path is not None and package_path not in search_path:
            package.__path__ = [package_path, *list(search_path)]


def _envelope_schema(action_schema: type[BaseModel]) -> type[BaseModel]:
    return create_model(
        "ClBenchMemoryEnvelope",
        action=(action_schema, ...),
        memory_writes=(list[MemoryWrite], Field(default_factory=list)),
    )


def _memory_arguments(memory_system: Any, write: MemoryWrite) -> tuple[str, dict[str, Any]]:
    definitions = memory_system.tool_definitions_for_mode("capture_only")
    if not definitions:
        raise ValueError("Selected memory system does not expose a storage tool")
    definition = definitions[0]
    properties = definition.get("input_schema", {}).get("properties", {})
    arguments: dict[str, Any] = {}
    if "action" in properties:
        arguments["action"] = "write"
    if "content" in properties:
        arguments["content"] = write.fact
    if "fact" in properties:
        arguments["fact"] = write.fact
    if "category" in properties:
        arguments["category"] = write.category
    return definition["name"], arguments


class MemoryCLBenchSystem:
    """CL-Bench system backed by a shared MemorySystem.

    The CL-Bench response schema remains authoritative for the task action.
    Memory writes are an auxiliary structured field that is converted through
    the selected MemorySystem's normal extraction/validation contract.
    """

    def __init__(
        self,
        *,
        model: str,
        memory_system: Any | None,
        memory_mode: str,
        memories: list[dict[str, Any]] | None = None,
    ) -> None:
        from src.interface import ContinualLearningSystem  # type: ignore

        if not isinstance(self, ContinualLearningSystem):
            # The actual base class is mixed in below once CL-Bench is on sys.path.
            pass
        from src.systems.utils import ProviderTurnClient  # type: ignore

        self.model = model
        self.memory_system = memory_system
        self.memory_mode = memory_mode
        self.memories = list(memories or [])
        self.events: list[dict[str, Any]] = []
        self.messages: list[dict[str, str]] = []
        self._provider = ProviderTurnClient(model=model, system_prompt="")
        self._usage_events: list[Any] = []

    @property
    def name(self) -> str:
        return "llm_memory_bench"

    def reset(self) -> None:
        # CL-Bench uses reset_between_instances for the stateless baseline. The
        # durable memory store deliberately survives that reset.
        self.messages = []
        self._provider.reset()

    def record_usage_event(self, event: Any) -> None:
        self._usage_events.append(event)

    def consume_usage_events(self) -> list[Any]:
        events = list(self._usage_events)
        self._usage_events.clear()
        return events

    def _system_prompt(self) -> str:
        if self.memory_system is None:
            return ""
        prompt_mode = "capture_only" if self.memory_mode == "capture_only" else "default"
        prompt = self.memory_system.system_prompt_for_mode(prompt_mode)
        if self.memory_mode == "replay":
            prompt = self.memory_system.inject_memories(prompt, self.memories)
        return (
            prompt
            + "\n\nCL-Bench memory protocol: keep the task action in the required "
            "action field. When the interaction reveals durable, reusable "
            "knowledge, add it to memory_writes. Include user preferences, "
            "constraints, task/domain facts, and decisions when they are likely "
            "to improve a later task. Do not copy transient tool output or guesses."
        )

    def respond(self, query: Any) -> Any:
        from src.interface import Response  # type: ignore

        self.messages.append({"role": "user", "content": query.prompt})
        schema = (
            _envelope_schema(query.response_schema)
            if self.memory_system is not None and self.memory_mode in {"capture_only", "replay"}
            else query.response_schema
        )
        self._provider.system_prompt = self._system_prompt()
        result = self._provider.respond_structured(
            messages=self.messages,
            response_schema=schema,
        )
        self._usage_events.extend(result.usage_events)
        parsed = result.action
        memory_writes: list[dict[str, Any]] = []
        if self.memory_system is not None and hasattr(parsed, "memory_writes"):
            for write in parsed.memory_writes:
                name, arguments = _memory_arguments(self.memory_system, write)
                from llm_memory_bench.providers.base import ToolCall

                call = ToolCall(name=name, arguments=arguments)
                validate = getattr(self.memory_system, "validate_tool_call_for_mode", None)
                error = validate(call, "capture_only") if validate else None
                if error:
                    self.events.append({"tool": name, "arguments": arguments, "error": error})
                    continue
                fact = self.memory_system.extract_stored_fact(call)
                if not fact:
                    continue
                memory_entry = {
                    "fact": fact,
                    "category": write.category,
                    "source": "clbench_interaction",
                    "tool": name,
                }
                if memory_entry not in self.memories:
                    self.memories.append(memory_entry)
                memory_writes.append(memory_entry)
                self.events.append(
                    {
                        "tool": name,
                        "arguments": arguments,
                        "stored_fact": fact,
                        "response": self.memory_system.format_tool_result(call),
                    }
                )
            action = parsed.action
        else:
            action = parsed
        self.messages.append({"role": "assistant", "content": action.model_dump_json()})
        return Response(
            action=action,
            metadata={"memory_writes": memory_writes, "memory_count": len(self.memories)},
        )

    def observe(self, observation: Any, next_query: Any | None = None) -> None:
        if observation.content:
            self.messages.append({"role": "user", "content": f"FEEDBACK:\n{observation.content}"})

    def get_run_artifacts(self) -> dict[str, Any]:
        return {
            "artifact_type": "llm_memory_bench_clbench",
            "memory_mode": self.memory_mode,
            "memories": list(self.memories),
            "memory_events": list(self.events),
            "model": self.model,
            "memory_system": None
            if self.memory_system is None
            else self.memory_system.version_info(),
        }


def _task_params(request: dict[str, Any], task_class: type) -> dict[str, Any]:
    import inspect

    supplied = {
        "schedule": request.get("schedule"),
        "variant": request.get("variant"),
        "num_instances": request.get("max_instances"),
        "seed": request.get("seed", 42),
    }
    accepted = inspect.signature(task_class.__init__).parameters
    return {key: value for key, value in supplied.items() if key in accepted and value is not None}


def _run_one(request: dict[str, Any], *, mode: str, memories: list[dict[str, Any]]) -> dict[str, Any]:
    from src.interface import run_task  # type: ignore
    from src.registry import get_task_class  # type: ignore

    task_class = get_task_class(request["task"])
    task = task_class(**_task_params(request, task_class))
    memory_system = None
    if mode in {"capture_only", "replay"}:
        from llm_memory_bench.systems import get_system

        memory_system = get_system(
            request.get("memory_system", "simple"),
            prompt_version=request.get("memory_prompt_version"),
        )
    system = MemoryCLBenchSystem(
        model=request["model"],
        memory_system=memory_system,
        memory_mode=mode,
        memories=memories,
    )
    print(
        f"[clbench] phase={mode} task={request['task']} "
        f"instances={request.get('max_instances') or 'default'}",
        file=sys.stderr,
        flush=True,
    )
    result = run_task(
        task,
        system,
        show_progress=True,
        verbose_logging=True,
        reset_between_instances=True,
        phase=mode,
    )
    task_result = _jsonable(result)
    task_result["score"] = result.score
    return {
        "task_result": task_result,
        "system_artifacts": _jsonable(system.get_run_artifacts()),
    }


def main() -> None:
    request = json.loads(sys.stdin.read())
    root = Path(request["clbench_root"]).resolve()
    _prepare_clbench_imports(root)
    # LiteLLM and the upstream runner may emit diagnostics to stdout. Keep the
    # worker protocol on stdout strictly JSON so the parent adapter can parse it.
    with redirect_stdout(sys.stderr):
        mode = request.get("memory_setup_mode", "none")
        if mode == "none":
            baseline = _run_one(request, mode="disabled", memories=[])
            payload = {"baseline": baseline, "memory": None, "captured_memories": []}
        elif mode == "seeded":
            baseline = _run_one(request, mode="disabled", memories=[])
            memory = _run_one(
                request, mode="replay", memories=request.get("seed_memories", [])
            )
            payload = {
                "baseline": baseline,
                "memory": memory,
                "captured_memories": request.get("seed_memories", []),
            }
        elif mode == "capture_first":
            capture = _run_one(request, mode="capture_only", memories=[])
            captured = capture["system_artifacts"].get("memories", [])
            memory = _run_one(request, mode="replay", memories=captured)
            payload = {
                "baseline": capture,
                "memory": memory,
                "captured_memories": captured,
            }
        else:
            raise ValueError(f"Unsupported memory setup mode: {mode}")

        baseline_score = payload["baseline"]["task_result"].get("score", 0.0)
        memory_result = payload.get("memory")
        memory_score = (
            None
            if memory_result is None
            else memory_result["task_result"].get("score", 0.0)
        )
        payload.update(
            {
                "harness": "clbench",
                "task": request["task"],
                "schedule": request.get("schedule"),
                "variant": request.get("variant"),
                "memory_setup_mode": mode,
                "comparison": {
                    "baseline_score": baseline_score,
                    "memory_score": memory_score,
                    "gain": None
                    if memory_score is None
                    else memory_score - baseline_score,
                },
            }
        )
    print(json.dumps(payload))


if __name__ == "__main__":
    main()


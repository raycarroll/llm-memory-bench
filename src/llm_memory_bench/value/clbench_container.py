"""Run CL-Bench tasks through the real containerised agent.

The CL-Bench task and scoring loop stay in the benchmark process.  Only the
system implementation crosses the container boundary: each ``respond`` call
is translated to a JSON request for the agent, and the agent's real MCP
memory calls are handled by the materialised integration proxy.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import os
import sys
import tempfile
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..real_system_container_extraction_runner import (
    ContainerDriverWorker,
    RunConfig as BenchRunConfig,
    build_host_image,
    cleanup_integration,
    drive_conversation,
    install_integration,
    install_container_system_prompt,
    load_integration,
    materialize_integration,
    query_integration,
    resolved_system_environment,
    start_memoryhub_proxy,
    start_container,
    stop_container,
    validate_integration_environment,
    _exec,
)
from ..seeding import (
    SeedError,
    SeedMemory,
    SeedPlan,
    cleanup_benchmark_data,
    seed_benchmark_data,
)


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {key: _jsonable(item) for key, item in dataclasses.asdict(value).items()}
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _new_project_id(label: str) -> str:
    safe = "".join(char if char.isalnum() else "-" for char in label)
    return f"llm-bench-clbench-{safe}-{uuid.uuid4().hex[:10]}"


def _task_params(request: dict[str, Any], task_class: type) -> dict[str, Any]:
    supplied = {
        "schedule": request.get("schedule"),
        "variant": request.get("variant"),
        "num_instances": request.get("max_instances"),
        "seed": request.get("seed", 42),
    }
    accepted = inspect.signature(task_class.__init__).parameters
    return {key: value for key, value in supplied.items() if key in accepted and value is not None}


def _prepare_clbench_imports(root: Path) -> None:
    """Make the official checkout win over this repository's ``src`` namespace.

    ``llm_memory_bench`` is installed from a ``src`` layout too. Once that
    namespace has been imported, merely inserting the CL-Bench root on
    ``sys.path`` is not sufficient: Python keeps the existing namespace
    package's search path. Add the checkout's package directory explicitly.
    """

    root = root.resolve()
    sys.path.insert(0, str(root))
    package = sys.modules.get("src")
    if package is not None:
        package_path = str(root / "src")
        search_path = getattr(package, "__path__", None)
        if search_path is not None and package_path not in search_path:
            package.__path__ = [package_path, *list(search_path)]


def _extract_json(text: str) -> dict[str, Any]:
    """Parse an agent response while tolerating a markdown JSON fence."""

    candidates = [text.strip()]
    if "```" in text:
        for block in text.split("```")[1::2]:
            candidates.append(block.removeprefix("json").strip())
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        value = json.loads(text[start : end + 1])
        if isinstance(value, dict):
            return value
    raise RuntimeError(f"Container agent did not return a JSON object: {text[:1000]!r}")


class ContainerCLBenchSystem:
    """CL-Bench system backed by one live container agent session."""

    def __init__(
        self,
        *,
        container_id: str,
        config: BenchRunConfig,
        profile_mode: bool,
        container_prompt_path: str | None,
    ):
        from src.interface import ContinualLearningSystem  # type: ignore

        self._base = ContinualLearningSystem
        self.container_id = container_id
        self.config = config
        self.profile_mode = profile_mode
        self.container_prompt_path = container_prompt_path
        self._driver_worker: ContainerDriverWorker | None = None
        self.session_id: str | None = None
        self._pending_observation: Any | None = None
        self.events: list[dict[str, Any]] = []
        self._usage_events: list[Any] = []

    @property
    def name(self) -> str:
        return "llm_memory_bench_container"

    def reset(self) -> None:
        # A CL-Bench instance reset starts a fresh agent conversation. The
        # container and its backend project remain alive for the run.
        if self._driver_worker is not None:
            self._driver_worker.reset()
        self.session_id = None
        self._pending_observation = None

    def close(self) -> None:
        if self._driver_worker is not None:
            self._driver_worker.close()
            self._driver_worker = None

    def record_usage_event(self, event: Any) -> None:
        self._usage_events.append(self._normalize_usage_event(event))

    def consume_usage_events(self) -> list[Any]:
        events = list(self._usage_events)
        self._usage_events.clear()
        return events

    def _normalize_usage_event(self, usage: Any) -> Any:
        """Adapt Claude SDK usage names to the installed CL-Bench schema."""

        from src.usage import UsageEvent  # type: ignore

        if isinstance(usage, UsageEvent):
            return usage
        values = usage if isinstance(usage, dict) else {}
        input_tokens = values.get("input_tokens", values.get("prompt_tokens"))
        output_tokens = values.get("output_tokens", values.get("completion_tokens"))
        cached_input_tokens = values.get(
            "cached_input_tokens",
            values.get("cache_read_input_tokens"),
        )
        total_tokens = values.get("total_tokens")
        if total_tokens is None and (input_tokens is not None or output_tokens is not None):
            total_tokens = int(input_tokens or 0) + int(output_tokens or 0)
        return UsageEvent(
            call_type="agent_turn",
            model=self.config.model,
            provider="anthropic",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cached_input_tokens=cached_input_tokens,
            cache_creation_input_tokens=values.get("cache_creation_input_tokens"),
            raw_usage=dict(values),
        )

    def respond(self, query: Any) -> Any:
        from src.interface import Response  # type: ignore

        schema = query.response_schema
        # The task prompt remains unmodified.  The response schema is passed
        # separately to the Claude Agent SDK's structured-output channel.
        instruction = query.prompt
        if query.feedback is not None and query.feedback.content:
            instruction += f"\n\nFeedback from the task:\n{query.feedback.content}"
        if self._pending_observation is not None:
            observation = self._pending_observation
            content = getattr(observation, "content", str(observation))
            if content:
                instruction += f"\n\nFeedback from the task:\n{content}"
            self._pending_observation = None

        if self._driver_worker is None:
            self._driver_worker = ContainerDriverWorker(
                container_id=self.container_id,
                model=self.config.model,
                profile_mode=self.profile_mode,
                system_prompt_file=self.container_prompt_path,
                system_prompt_mode=self.config.system_prompt_mode,
            )
        response_record = self._driver_worker.respond(
            instruction,
            schema.model_json_schema(),
        )
        if response_record.get("error"):
            raise RuntimeError(str(response_record["error"]))
        previous_session_id = self.session_id
        self.session_id, session_continuity = _require_session_continuity(
            previous_session_id,
            response_record.get("session_id"),
            response_record.get("session_id"),
        )
        structured_output = response_record.get("structured_output")
        if not isinstance(structured_output, dict):
            raise RuntimeError(
                "Container driver returned no structured CL-Bench response; "
                "the agent SDK output schema was not applied"
            )
        action = schema.model_validate(structured_output)
        tool_calls = response_record.get("tool_calls", [])
        tool_results = response_record.get("tool_results", [])
        integration_events = response_record.get("integration_events", [])
        self.events.append(
            {
                "instance_id": query.instance_id,
                "session_id": self.session_id,
                "session_continuity": session_continuity,
                "session_continuity_verified": True,
                "tool_calls": tool_calls if isinstance(tool_calls, list) else [],
                "tool_results": tool_results if isinstance(tool_results, list) else [],
                "integration_events": (
                    integration_events if isinstance(integration_events, list) else []
                ),
            }
        )
        usage = response_record.get("usage", {})
        if isinstance(usage, dict):
            self._usage_events.append(self._normalize_usage_event(usage))
        return Response(
            action=action,
            metadata={
                "container_session_id": self.session_id,
                "session_continuity": session_continuity,
                "session_continuity_verified": True,
            },
        )

    def observe(self, observation: Any, next_query: Any | None = None) -> None:
        del next_query
        self._pending_observation = observation

    def get_run_artifacts(self) -> dict[str, Any]:
        return {
            "artifact_type": "llm_memory_bench_clbench_container",
            "memory_transport": "agent_mcp_to_benchmark_proxy_to_memoryhub_mcp",
            "proxy_arm": self.config.system_env.get("MEMORYHUB_BENCH_ARM"),
            "events": list(self.events),
        }


def _require_session_continuity(
    previous_session_id: str | None,
    response_session_id: Any,
    payload_session_id: Any,
) -> tuple[str, str]:
    """Require a Claude session ID and verify resumed turns stay in-session."""

    response_id = response_session_id if isinstance(response_session_id, str) else None
    payload_id = payload_session_id if isinstance(payload_session_id, str) else None
    if response_id and payload_id and response_id != payload_id:
        raise RuntimeError(
            "Claude driver returned conflicting session IDs in its response"
        )
    current_session_id = response_id or payload_id
    if not current_session_id:
        if previous_session_id:
            raise RuntimeError(
                "Claude session continuity lost: resumed turn returned no session ID"
            )
        raise RuntimeError("Claude driver returned no session ID for a new session")
    if previous_session_id is not None and current_session_id != previous_session_id:
        raise RuntimeError(
            "Claude session continuity lost: resumed turn returned a different "
            "session ID"
        )
    return current_session_id, "resumed" if previous_session_id else "new"


def _task_result_payload(result: Any) -> dict[str, Any]:
    payload = _jsonable(result)
    if isinstance(payload, dict):
        payload["score"] = getattr(result, "score", payload.get("score", 0.0))
    return payload


def _proxy_audit(container_id: str) -> list[dict[str, Any]]:
    result = _exec(
        container_id,
        ["bash", "-lc", "cat /run/llm-memory-bench/memoryhub-proxy-audit.jsonl 2>/dev/null || true"],
        timeout=10,
    )
    records: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            records.append(item)
    return records


def _run_phase_in_checkout(
    *,
    root: Path,
    request: dict[str, Any],
    container_id: str,
    config: BenchRunConfig,
    profile_mode: bool,
    phase: str,
    container_prompt_path: str | None,
) -> dict[str, Any]:
    _prepare_clbench_imports(root)
    from src.interface import run_task  # type: ignore
    from src.registry import get_task_class  # type: ignore

    task_class = get_task_class(request["task"])
    task = task_class(**_task_params(request, task_class))
    system = ContainerCLBenchSystem(
        container_id=container_id,
        config=config,
        profile_mode=profile_mode,
        container_prompt_path=container_prompt_path,
    )
    print(
        f"[clbench-container] phase={phase} task={request['task']} "
        f"instances={request.get('max_instances') or 'default'}",
        file=sys.stderr,
        flush=True,
    )
    try:
        result = run_task(
            task,
            system,
            show_progress=True,
            verbose_logging=True,
            reset_between_instances=True,
            phase=phase,
        )
        artifacts = _jsonable(system.get_run_artifacts())
        artifacts["proxy_audit"] = _proxy_audit(container_id)
        return {
            "task_result": _task_result_payload(result),
            "system_artifacts": artifacts,
        }
    finally:
        system.close()


def _run_phase(
    *,
    root: Path,
    request: dict[str, Any],
    container_id: str,
    config: BenchRunConfig,
    profile_mode: bool,
    phase: str,
    container_prompt_path: str | None,
) -> dict[str, Any]:
    """Run one phase from the CL-Bench checkout directory.

    Several upstream tasks intentionally use paths such as
    ``data/database_exploration/products.db`` relative to the checkout.
    """

    previous_cwd = os.getcwd()
    os.chdir(root)
    try:
        return _run_phase_in_checkout(
            root=root,
            request=request,
            container_id=container_id,
            config=config,
            profile_mode=profile_mode,
            phase=phase,
            container_prompt_path=container_prompt_path,
        )
    finally:
        os.chdir(previous_cwd)


def _captured_seed_plan(request: dict[str, Any], memories: list[dict[str, Any]]) -> SeedPlan:
    seed_memories = tuple(
        SeedMemory(
            content=str(item.get("content") or item.get("fact") or item.get("text")),
            scope="project",
            metadata={"category": item.get("category", "captured"), "source": "clbench_capture"},
            force=True,
        )
        for item in memories
        if isinstance(item, dict) and (item.get("content") or item.get("fact") or item.get("text"))
    )
    canonical = json.dumps([memory.content for memory in seed_memories], sort_keys=True)
    import hashlib

    return SeedPlan(
        path=Path(f"<clbench-captured:{request['task']}>") ,
        api_version="llm-memory-bench.dev/v1",
        seed_id=f"clbench-capture-{request['task']}",
        version="1.0.0",
        provider="memoryhub",
        mode="declarative",
        memories=seed_memories,
        fixture_hash=hashlib.sha256(canonical.encode()).hexdigest(),
    )


def _write_prompt_file(prompt_system: Any, mode: str) -> Path:
    """Materialize the prompt variant selected for a CL-Bench execution arm."""

    prompt_file_handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".txt",
        prefix=f"llm-bench-clbench-{mode}-prompt-",
        delete=False,
    )
    prompt_file_handle.write(prompt_system.system_prompt_for_mode(mode))
    prompt_file_handle.close()
    return Path(prompt_file_handle.name)


def run_clbench_container_value_benchmark(
    *,
    clbench_root: Path,
    request: dict[str, Any],
    container_config: BenchRunConfig,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Run CL-Bench in capture/replay mode using the real MCP integration."""

    profile = load_integration(container_config)
    if profile is None:
        raise RuntimeError("CL-Bench container runs require integration.profile")
    validate_integration_environment(container_config, profile)
    build_host_image(container_config)

    mode = request.get("memory_setup_mode", "none")
    if mode not in {"none", "seeded", "capture_first"}:
        raise ValueError(f"Unsupported memory setup mode: {mode}")

    root = clbench_root.resolve()
    prompt_system: Any | None = None
    prompt_version = request.get("memory_prompt_version")
    if prompt_version:
        from ..systems import get_system

        prompt_system = get_system(
            request.get("memory_system", "memoryhub"),
            prompt_version=prompt_version,
        )
        prompt_file = _write_prompt_file(prompt_system, "default")
        container_config = replace(container_config, system_prompt_file=str(prompt_file))
    baseline: dict[str, Any]
    captured: list[dict[str, Any]] = []
    start = time.monotonic()
    proxy = start_memoryhub_proxy(container_config, profile) if mode != "none" else None

    if mode == "capture_first":
        capture_config = container_config
        if prompt_system is not None:
            capture_prompt_file = _write_prompt_file(prompt_system, "capture_only")
            capture_config = replace(
                container_config,
                system_prompt_file=str(capture_prompt_file),
            )
        project_id = _new_project_id("capture")
        workspace = tempfile.TemporaryDirectory(prefix="llm-bench-clbench-capture-")
        container_id: str | None = None
        try:
            materialize_integration(
                capture_config, profile, workspace.name, project_id,
                proxy_url=proxy.url,
            )
            container_id = start_container(
                capture_config,
                profile=profile,
                project_id=project_id,
                proxy_arm="capture",
                proxy_audit_path="/run/llm-memory-bench/memoryhub-proxy-audit.jsonl",
                proxy_url=proxy.url,
            )
            install_integration(container_id, workspace.name, profile)
            container_prompt_path = install_container_system_prompt(container_id, capture_config)
            baseline = _run_phase(
                root=root,
                request=request,
                container_id=container_id,
                config=capture_config,
                profile_mode=True,
                phase="capture_only",
                container_prompt_path=container_prompt_path,
            )
            captured, _ = query_integration(container_id, profile, project_id)
        finally:
            if container_id is not None:
                cleanup_integration(container_id, profile, project_id)
                stop_container(container_id, keep=container_config.keep_container)
            workspace.cleanup()
        seed_plan = _captured_seed_plan(request, captured)
    else:
        # A no-memory baseline is still run through the real host container.
        # The seeded arm uses the same agent image and normal prompt, with the
        # proxy in replay mode and the declared fixture in its project.
        baseline_config = container_config
        container_id = start_container(
            baseline_config,
            profile=None,
            legacy_install=False,
        )
        try:
            container_prompt_path = install_container_system_prompt(container_id, container_config)
            baseline = _run_phase(
                root=root,
                request=request,
                container_id=container_id,
                config=baseline_config,
                profile_mode=False,
                phase="baseline",
                container_prompt_path=container_prompt_path,
            )
        finally:
            stop_container(container_id, keep=container_config.keep_container)
        if mode == "seeded":
            from .clbench import load_seed_memories

            declared = load_seed_memories(
                Path(request["seed_memories_path"]) if request.get("seed_memories_path") else None
            )
            seed_plan = _captured_seed_plan(request, declared)
        else:
            seed_plan = _captured_seed_plan(request, [])

    memory: dict[str, Any] | None = None
    if mode != "none":
        project_id = _new_project_id("replay")
        workspace = tempfile.TemporaryDirectory(prefix="llm-bench-clbench-replay-")
        container_id = None
        seeded = False
        seed_environment = resolved_system_environment(container_config)
        try:
            materialize_integration(
                container_config, profile, workspace.name, project_id,
                proxy_url=proxy.url,
            )
            if seed_environment.get("MEMORYHUB_SEED_URL"):
                seed_environment["MEMORYHUB_URL"] = seed_environment["MEMORYHUB_SEED_URL"]
            try:
                seed_result = seed_benchmark_data(seed_plan, project_id, seed_environment)
                seeded = True
            except SeedError as exc:
                raise RuntimeError(f"CL-Bench replay seed failed: {exc}") from exc
            container_id = start_container(
                container_config,
                profile=profile,
                project_id=project_id,
                proxy_arm="replay",
                proxy_audit_path="/run/llm-memory-bench/memoryhub-proxy-audit.jsonl",
                proxy_url=proxy.url,
            )
            install_integration(container_id, workspace.name, profile)
            container_prompt_path = install_container_system_prompt(container_id, container_config)
            memory = _run_phase(
                root=root,
                request=request,
                container_id=container_id,
                config=container_config,
                profile_mode=True,
                phase="replay",
                container_prompt_path=container_prompt_path,
            )
        finally:
            if container_id is not None:
                stop_container(container_id, keep=container_config.keep_container)
            if seeded:
                cleanup_benchmark_data(profile.memory_system["name"], project_id, seed_environment, seed_result)
            workspace.cleanup()

    baseline_score = baseline["task_result"].get("score", 0.0)
    memory_score = None if memory is None else memory["task_result"].get("score", 0.0)
    result = {
        "harness": "clbench",
        "execution": "container",
        "task": request["task"],
        "schedule": request.get("schedule"),
        "variant": request.get("variant"),
        "memory_setup_mode": mode,
        "baseline": baseline,
        "memory": memory,
        "captured_memories": captured,
        "comparison": {
            "baseline_score": baseline_score,
            "memory_score": memory_score,
            "gain": None if memory_score is None else memory_score - baseline_score,
        },
        "elapsed_ms": round((time.monotonic() - start) * 1000, 3),
    }
    if prompt_file is not None:
        prompt_file.unlink(missing_ok=True)
    if proxy is not None:
        proxy.stop()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result

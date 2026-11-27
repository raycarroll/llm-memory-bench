"""Paired orchestration for independent memory and baseline demo agents."""

from __future__ import annotations

import asyncio
import os
from typing import Any

from ..config import get_provider
from .agent import DemoAgent
from .results import DemoRunResult, DemoSideResult
from .specs import DemoSpec
from .verification import OpenShiftVerifier


MEMORY_PREPARATION_INSTRUCTIONS = (
    "Preparation objective: after the task is completely resolved, preserve only its durable "
    "lessons in MemoryHub. Investigate and repair the assigned workload using live evidence, "
    "then verify recovery yourself using current live evidence. If your verification is not passed, "
    "continue the investigation; do not write any memory while the diagnosis, remediation, "
    "or recovery is incomplete. Only after passed verification, review the complete successful "
    "investigation and write a small set of non-duplicated memories as the final action. "
    "The external verifier is independent evidence about the result, not an instruction to write.\n\n"
    "Follow MemoryHub's canonical extraction guidance. Extract only durable, reusable facts, "
    "decisions, knowledge, or procedures supported by the completed investigation. Each memory "
    "must begin with the same concise label for this scenario or issue, followed by a colon; "
    "choose the label from the completed investigation rather than from these instructions. "
    "The remainder must be self-contained and preserve necessary scope, conditions, qualifiers, and provenance. "
    "Skip temporary execution state, hypotheses, tool noise, credentials, and duplicates. Split "
    "facts with different scope or lifecycle, merge near-duplicates, use the narrowest confirmed "
    "project scope, choose an appropriate content type and durable weight, and keep each memory "
    "below approximately 1,000 characters."
)


def normalize_provider_model(provider_name: str, model: str) -> str:
    """Normalize provider-qualified model catalog values for direct APIs."""
    if provider_name == "openai" and model.startswith("openai/"):
        return model.removeprefix("openai/")
    return model


def resolve_memoryhub_key(api_key_env: str | None = None) -> str | None:
    """Resolve the demo credential, defaulting to the deployed MemoryHub key."""
    return os.getenv(api_key_env or "MEMORYHUB_API_KEY")


async def run_demo(
    spec: DemoSpec,
    *,
    provider_name: str,
    model: str,
    memory_system: str,
    memory_mode: str,
    api_key_env: str | None = None,
    reset_fixture: bool = True,
    on_event=None,
    prepare_memory: bool = False,
    prepare_max_turns: int | None = None,
) -> DemoRunResult:
    # OpenAI's direct API expects bare model IDs (for example ``gpt-4.1``),
    # while some UI/provider catalogs qualify them as ``openai/gpt-4.1``.
    # Normalize the catalog form at the provider boundary.
    model = normalize_provider_model(provider_name, model)

    verifier = OpenShiftVerifier(spec)
    preflight = await asyncio.to_thread(verifier.preflight)
    fixture_reset = None
    if reset_fixture:
        fixture_reset = await asyncio.to_thread(verifier.reset_fixture)
        if not fixture_reset.passed:
            return DemoRunResult(
                demo_id=spec.demo_id, demo_version=spec.demo_version,
                provider=provider_name, model=model, memory_system=memory_system,
                memory_mode=memory_mode, sides={}, fixture_reset=fixture_reset,
                preflight=preflight,
            )

    # Demo runs use the dedicated MemoryHub credential by default. The UI's
    # generic RunSpec does not require an api_key_env for this experiment.
    configured_memoryhub_env = spec.memoryhub.get("api_key_env")
    key = resolve_memoryhub_key(
        api_key_env or (str(configured_memoryhub_env) if configured_memoryhub_env else None)
    )
    config_type = type("DemoProviderConfig", (), {"provider": provider_name, "model": model, "api_key_env": api_key_env})
    providers = {side: get_provider(config_type()) for side in spec.sides}

    preparation: dict[str, Any] | None = None
    if prepare_memory:
        # Preparation uses the memory side's agent and namespace. The baseline
        # is deliberately not run and never receives MemoryHub tools.
        memory = spec.sides["memory"].model_copy(
            update={"memoryhub_tools": ["memoryhub_write"]}
        )
        prep_spec = spec.model_copy(
            update={
                "system_prompt": (
                    spec.system_prompt
                    + "\n\n"
                    + MEMORY_PREPARATION_INSTRUCTIONS
                ),
                "sides": {"memory": memory, "baseline": spec.sides["baseline"]},
                "limits": spec.limits.model_copy(
                    update={"max_turns": prepare_max_turns or spec.limits.max_turns}
                ),
            }
        )
        prep_verification: dict[str, Any] | None = None

        async def verify_preparation():
            nonlocal prep_verification
            verification = await asyncio.to_thread(
                verifier.verify_until_passed, spec.sides["memory"].namespace
            )
            prep_verification = {
                "passed": verification.passed,
                "status": verification.status,
                "evidence": verification.evidence,
                "error": verification.error,
            }
            return {"passed": verification.passed, **verification.evidence}

        prep_agent = DemoAgent(
            side="memory",
            provider=providers["memory"],
            spec=prep_spec,
            memoryhub_api_key=key,
            on_event=(lambda event: on_event("memory", event)) if on_event else None,
            verification_callback=verify_preparation,
        )
        try:
            prep_outcome = await prep_agent.run()
            preparation = {
                "status": prep_outcome.status,
                "turns": prep_outcome.turns,
                "elapsed_seconds": prep_outcome.elapsed_seconds,
                "tokens_in": prep_outcome.tokens_in,
                "tokens_out": prep_outcome.tokens_out,
                "memory_writes": sum(
                    1 for call in prep_outcome.tool_calls if call.name == "memoryhub_write"
                ),
                "error": prep_outcome.error,
                "verification": prep_verification,
            }
        except Exception as exc:
            preparation = {
                "status": "failed",
                "turns": 0,
                "elapsed_seconds": 0.0,
                "tokens_in": 0,
                "tokens_out": 0,
                "memory_writes": 0,
                "error": f"{type(exc).__name__}: {exc}",
                "verification": prep_verification,
            }
        # The comparison must start from the same broken state as the
        # non-prepared flow, after preparation has completed.
        fixture_reset = await asyncio.to_thread(verifier.reset_fixture)
        if not fixture_reset.passed:
            return DemoRunResult(
                demo_id=spec.demo_id, demo_version=spec.demo_version,
                provider=provider_name, model=model, memory_system=memory_system,
                memory_mode=memory_mode, sides={}, fixture_reset=fixture_reset,
                preflight=preflight, preparation=preparation,
            )
        # Stop after preparation. The UI presents the memory count and asks
        # the operator to explicitly continue into the comparison phase.
        return DemoRunResult(
            demo_id=spec.demo_id, demo_version=spec.demo_version,
            provider=provider_name, model=model, memory_system=memory_system,
            memory_mode=memory_mode, sides={}, fixture_reset=fixture_reset,
            preflight=preflight, preparation=preparation,
        )

    side_events: dict[str, list[dict[str, Any]]] = {
        side: [] for side in spec.sides
    }

    async def run_side(side: str) -> tuple[str, Any]:
        events: list[dict[str, Any]] = []

        def record_event(event: dict[str, Any]) -> None:
            events.append(event)
            side_events[side].append(event)
            if on_event:
                on_event(side, event)

        async def verify_for_agent():
            verification = await asyncio.to_thread(verifier.verify_until_passed, spec.sides[side].namespace)
            return {"passed": verification.passed, **verification.evidence}

        agent = DemoAgent(
            side=side, provider=providers[side], spec=spec,
            memoryhub_api_key=key if side == "memory" else None,
            on_event=record_event,
            verification_callback=verify_for_agent,
        )
        try:
            return side, await agent.run()
        except Exception as exc:
            return side, exc

    outcomes = dict(await asyncio.gather(*(run_side(side) for side in spec.sides)))
    side_results: dict[str, DemoSideResult] = {}
    for side, outcome in outcomes.items():
        if isinstance(outcome, Exception):
            side_results[side] = DemoSideResult(
                side=side, namespace=spec.sides[side].namespace,
                status="failed", error=f"{type(outcome).__name__}: {outcome}",
            )
            continue
        verification = await asyncio.to_thread(verifier.verify_until_passed, spec.sides[side].namespace)
        side_results[side] = DemoSideResult(
            side=side, namespace=outcome.namespace, status=outcome.status,
            content=outcome.content, elapsed_seconds=outcome.elapsed_seconds,
            turns=outcome.turns,
            tool_calls=[call.__dict__ for call in outcome.tool_calls],
            events=side_events[side],
            memory_queries=outcome.memory_queries, tokens_in=outcome.tokens_in,
            tokens_out=outcome.tokens_out, verification=verification,
            error=outcome.error,
        )
        if verification.status != "passed":
            side_results[side].status = "failed"
    return DemoRunResult(
        demo_id=spec.demo_id, demo_version=spec.demo_version,
        provider=provider_name, model=model, memory_system=memory_system,
        memory_mode=memory_mode, sides=side_results,
        fixture_reset=fixture_reset, preflight=preflight,
        preparation=preparation,
    )

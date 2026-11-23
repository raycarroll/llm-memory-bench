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
    key = resolve_memoryhub_key(api_key_env)
    config_type = type("DemoProviderConfig", (), {"provider": provider_name, "model": model, "api_key_env": api_key_env})
    providers = {side: get_provider(config_type()) for side in spec.sides}

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
            verification = await asyncio.to_thread(verifier.verify, spec.sides[side].namespace)
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
        verification = await asyncio.to_thread(verifier.verify, spec.sides[side].namespace)
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
    )

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

    key = os.getenv(api_key_env) if api_key_env else None
    config_type = type("DemoProviderConfig", (), {"provider": provider_name, "model": model, "api_key_env": api_key_env})
    providers = {side: get_provider(config_type()) for side in spec.sides}

    async def run_side(side: str) -> tuple[str, Any]:
        agent = DemoAgent(
            side=side, provider=providers[side], spec=spec,
            memoryhub_api_key=key if side == "memory" else None,
            on_event=(lambda event: on_event(side, event)) if on_event else None,
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
            content=outcome.content, turns=outcome.turns,
            tool_calls=[call.__dict__ for call in outcome.tool_calls],
            memory_queries=outcome.memory_queries, tokens_in=outcome.tokens_in,
            tokens_out=outcome.tokens_out, verification=verification,
            error=outcome.error,
        )
    return DemoRunResult(
        demo_id=spec.demo_id, demo_version=spec.demo_version,
        provider=provider_name, model=model, memory_system=memory_system,
        memory_mode=memory_mode, sides=side_results,
        fixture_reset=fixture_reset, preflight=preflight,
    )

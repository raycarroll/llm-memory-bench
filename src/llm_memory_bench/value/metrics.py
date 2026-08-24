from __future__ import annotations

from .runner import TrialResult


def compute_value_metrics(
    baseline: TrialResult,
    memory: TrialResult,
    quality_baseline: float,
    quality_memory: float,
) -> dict:
    baseline_tokens = baseline.total_agent_tokens
    memory_tokens = memory.total_agent_tokens
    memory_overhead = memory.memory_tokens

    net_token_savings = baseline_tokens - memory_tokens
    effective_savings = net_token_savings - memory_overhead

    token_efficiency_ratio = (
        baseline_tokens / memory_tokens if memory_tokens > 0 else float("inf")
    )

    baseline_turns = baseline.turn_count
    memory_turns = memory.turn_count
    turn_compression = (
        baseline_turns / memory_turns if memory_turns > 0 else float("inf")
    )

    quality_delta = quality_memory - quality_baseline

    value_per_token = (
        quality_delta / memory_overhead if memory_overhead > 0 else 0.0
    )

    return {
        "baseline_agent_tokens": baseline_tokens,
        "memory_agent_tokens": memory_tokens,
        "memory_overhead_tokens": memory_overhead,
        "net_token_savings": net_token_savings,
        "effective_token_savings": effective_savings,
        "token_efficiency_ratio": round(token_efficiency_ratio, 4),
        "baseline_turns": baseline_turns,
        "memory_turns": memory_turns,
        "turn_compression": round(turn_compression, 4),
        "baseline_questions_asked": baseline.questions_asked,
        "memory_questions_asked": memory.questions_asked,
        "quality_baseline": round(quality_baseline, 4),
        "quality_memory": round(quality_memory, 4),
        "quality_delta": round(quality_delta, 4),
        "value_per_token": round(value_per_token, 6),
    }


def aggregate_metrics(per_scenario: list[dict]) -> dict:
    if not per_scenario:
        return {}

    n = len(per_scenario)

    def avg(key: str) -> float:
        vals = [m[key] for m in per_scenario if key in m]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    def total(key: str) -> int | float:
        return sum(m.get(key, 0) for m in per_scenario)

    positive_delta = sum(1 for m in per_scenario if m.get("quality_delta", 0) > 0)
    negative_delta = sum(1 for m in per_scenario if m.get("quality_delta", 0) < 0)
    net_positive_savings = sum(
        1 for m in per_scenario if m.get("effective_token_savings", 0) > 0
    )

    return {
        "scenario_count": n,
        "avg_token_efficiency_ratio": avg("token_efficiency_ratio"),
        "avg_turn_compression": avg("turn_compression"),
        "avg_quality_delta": avg("quality_delta"),
        "avg_value_per_token": avg("value_per_token"),
        "total_baseline_tokens": total("baseline_agent_tokens"),
        "total_memory_tokens": total("memory_agent_tokens"),
        "total_memory_overhead": total("memory_overhead_tokens"),
        "total_effective_savings": total("effective_token_savings"),
        "scenarios_quality_improved": positive_delta,
        "scenarios_quality_degraded": negative_delta,
        "scenarios_tokens_saved": net_positive_savings,
    }

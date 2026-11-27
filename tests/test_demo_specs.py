import pytest
from pydantic import ValidationError

from llm_memory_bench.demo import (
    DemoLimits,
    DemoRunResult,
    DemoSideResult,
    DemoSideSpec,
    DemoSpec,
    VerificationResult,
)


def valid_spec(**overrides):
    payload = {
        "demo_id": "example@1.0.0",
        "demo_version": "1.0.0",
        "prompt": "Repair the workload and verify it.",
        "memoryhub_project_id": "example-project",
        "sides": {
            "memory": {
                "namespace": "demo-memory",
                "kubernetes_endpoint_env": "K8S_MEMORY_MCP_URL",
                "kubernetes_tools": ["pods_get", "resources_create_or_update"],
                "memoryhub_tools": ["memoryhub_search"],
            },
            "baseline": {
                "namespace": "demo-baseline",
                "kubernetes_endpoint_env": "K8S_BASELINE_MCP_URL",
                "kubernetes_tools": ["pods_get", "resources_create_or_update"],
            },
        },
        "read_only_tools": ["pods_get"],
        "change_making_tools": ["resources_create_or_update"],
        "change_resources": ["networkpolicy/example"],
    }
    payload.update(overrides)
    return payload


def test_demo_spec_requires_exactly_two_named_sides():
    with pytest.raises(ValidationError, match="exactly memory and baseline"):
        DemoSpec(**valid_spec(sides={"memory": valid_spec()["sides"]["memory"]}))


def test_demo_spec_rejects_unclassified_kubernetes_tools():
    with pytest.raises(ValidationError, match="unclassified"):
        DemoSpec(**valid_spec(sides={
            **valid_spec()["sides"],
            "memory": {
                **valid_spec()["sides"]["memory"],
                "kubernetes_tools": ["unknown_tool"],
            },
        }))


def test_demo_spec_rejects_memory_tools_without_project():
    with pytest.raises(ValidationError, match="memoryhub_project_id"):
        DemoSpec(**valid_spec(memoryhub_project_id=None))


def test_demo_spec_rejects_baseline_memory_tools():
    sides = valid_spec()["sides"]
    sides["baseline"]["memoryhub_tools"] = ["memoryhub_search"]
    with pytest.raises(ValidationError, match="baseline side"):
        DemoSpec(**valid_spec(sides=sides))


def test_demo_spec_rejects_overlapping_tool_classes():
    with pytest.raises(ValidationError, match="both read-only and change-making"):
        DemoSpec(**valid_spec(change_making_tools=["pods_get"]))


def test_demo_limits_are_positive_and_serializable():
    limits = DemoLimits(max_turns=3, timeout_seconds=10)
    assert limits.model_dump() == {
        "max_turns": 3,
        "max_tool_calls": 36,
        "timeout_seconds": 10.0,
        "tool_timeout_seconds": 45.0,
    }


def test_verification_result_distinguishes_unavailable_from_failure():
    unavailable = VerificationResult(status="unavailable", passed=False, error="timeout")
    failed = VerificationResult(status="failed", passed=False, evidence={"ready": False})

    assert unavailable.status != failed.status
    assert unavailable.passed is False
    assert failed.evidence["ready"] is False


def test_paired_result_requires_both_sides_to_pass():
    sides = {
        name: DemoSideResult(side=name, namespace=f"demo-{name}", status="completed")
        for name in ("memory", "baseline")
    }
    result = DemoRunResult(
        demo_id="example@1.0.0",
        demo_version="1.0.0",
        provider="openai",
        model="gpt-4o",
        memory_system="memoryhub",
        memory_mode="both",
        sides=sides,
    )
    assert result.independent_passed() is False

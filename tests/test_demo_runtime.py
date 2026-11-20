from types import SimpleNamespace

import pytest

from llm_memory_bench.demo import DemoToolRuntime, load_demo_spec


SPEC = load_demo_spec("scenarios/openshift/network-policy.yaml")


def runtime(side="memory"):
    return DemoToolRuntime(side=side, spec=SPEC, memoryhub_api_key="test")


def manifest(namespace="memory-demo-memory", name="frontend-egress", api_version="networking.k8s.io/v1"):
    return f"""apiVersion: {api_version}
kind: NetworkPolicy
metadata:
  name: {name}
  namespace: {namespace}
spec: {{}}
"""


def test_memory_side_hides_mutations_before_search():
    demo = runtime()
    demo._tools = {
        "pods_get": {"name": "pods_get", "parameters": {}},
        "resources_create_or_update": {"name": "resources_create_or_update", "parameters": {}},
        "memoryhub_search": {"name": "memoryhub_search", "parameters": {}},
    }

    assert [tool["name"] for tool in demo.tools_for_turn()] == ["pods_get", "memoryhub_search"]


def test_search_unlocks_memory_side_mutations():
    demo = runtime()
    demo._tools = {
        "pods_get": {"name": "pods_get", "parameters": {}},
        "resources_create_or_update": {"name": "resources_create_or_update", "parameters": {}},
        "memoryhub_search": {"name": "memoryhub_search", "parameters": {}},
    }
    demo.memory_search_completed = True

    assert {tool["name"] for tool in demo.tools_for_turn()} == {
        "pods_get", "resources_create_or_update", "memoryhub_search"
    }


def test_change_validation_requires_exact_namespace_resource_and_api():
    demo = runtime()
    allowed, reason = demo._validate_change(
        "resources_create_or_update", {"resource": manifest()}
    )
    assert allowed is True
    assert reason == ""

    for kwargs, expected in [
        ({"resource": manifest(namespace="other")}, "namespace"),
        ({"resource": manifest(name="deployment/frontend")}, "not writable"),
        ({"resource": manifest(api_version="v1")}, "networking.k8s.io/v1"),
    ]:
        allowed, reason = demo._validate_change("resources_create_or_update", kwargs)
        assert allowed is False
        assert expected in reason


def test_baseline_never_requires_memory_search():
    demo = runtime("baseline")
    demo._tools = {"resources_create_or_update": {"name": "resources_create_or_update", "parameters": {}}}

    assert len(demo.tools_for_turn()) == 1


def test_namespace_enforcement_fills_or_rejects_discovered_namespace_argument():
    demo = runtime()
    demo._tools = {"pods_get": {"name": "pods_get", "parameters": {"properties": {"namespace": {}}}}}

    assert demo._enforce_namespace("pods_get", {})["namespace"] == "memory-demo-memory"
    with pytest.raises(ValueError, match="restricted to namespace"):
        demo._enforce_namespace("pods_get", {"namespace": "other"})

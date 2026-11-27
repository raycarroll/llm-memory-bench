import pytest

from llm_memory_bench.demo import OpenShiftVerifier, load_demo_spec


SPEC = load_demo_spec("scenarios/openshift/network-policy.yaml")


def test_selector_matching_uses_actual_labels():
    assert OpenShiftVerifier._selector_matches({"matchLabels": {"app": "frontend"}}, {"app": "frontend"})
    assert not OpenShiftVerifier._selector_matches({"matchLabels": {"app": "frontend"}}, {"app": "frontend-v1"})


def test_oc_requires_explicit_context(monkeypatch):
    verifier = OpenShiftVerifier(SPEC)
    verifier.context = None
    monkeypatch.delenv("OPENSHIFT_CONTEXT", raising=False)

    with pytest.raises(RuntimeError, match="OPENSHIFT_CONTEXT"):
        verifier._oc(["whoami"])


def test_policy_comparison_ignores_rule_order():
    first = {"ports": [{"protocol": "TCP", "port": 8080}], "to": [{"podSelector": {"matchLabels": {"app": "dependency"}}}]}
    second = {"to": [{"podSelector": {"matchLabels": {"app": "dependency"}}}], "ports": [{"port": 8080, "protocol": "TCP"}]}
    assert OpenShiftVerifier._same(first, second)


def test_fixture_broken_requires_expected_negative_evidence():
    verifier = OpenShiftVerifier(SPEC)
    result = type("Result", (), {"evidence": {
        "service_has_dependency_endpoint": True,
        "policy_selects_dependency": True,
        "policy_allows_client": False,
        "egress_policy_allows_dns": True,
        "egress_policy_allows_dependency_port": False,
    }})()
    assert verifier._fixture_is_broken(result)

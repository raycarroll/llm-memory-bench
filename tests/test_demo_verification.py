from types import SimpleNamespace

import pytest

from llm_memory_bench.demo import OpenShiftVerifier, load_demo_spec


SPEC = load_demo_spec("scenarios/openshift/network-policy.yaml")


def test_selector_matching_uses_actual_labels():
    assert OpenShiftVerifier._selector_matches({"matchLabels": {"app": "frontend"}}, {"app": "frontend"})
    assert not OpenShiftVerifier._selector_matches({"matchLabels": {"app": "frontend"}}, {"app": "frontend-v1"})


def test_oc_uses_in_cluster_service_account_when_context_is_missing(monkeypatch, tmp_path):
    verifier = OpenShiftVerifier(SPEC)
    verifier.context = None
    monkeypatch.delenv("OPENSHIFT_CONTEXT", raising=False)
    monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.0.0.1")
    monkeypatch.setenv("KUBERNETES_SERVICE_PORT_HTTPS", "6443")
    (tmp_path / "token").write_text("service-account-token")
    (tmp_path / "ca.crt").write_text("ca")
    verifier.service_account_dir = tmp_path
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)

    assert verifier._oc(["whoami"]) == "ok"
    assert captured["command"] == [
        "oc",
        "--server",
        "https://10.0.0.1:6443",
        "--token",
        "service-account-token",
        "--certificate-authority",
        str(tmp_path / "ca.crt"),
        "whoami",
    ]


def test_oc_requires_context_or_in_cluster_credentials(monkeypatch, tmp_path):
    verifier = OpenShiftVerifier(SPEC)
    verifier.context = None
    verifier.service_account_dir = tmp_path
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)

    with pytest.raises(RuntimeError, match="kubeconfig.*service-account"):
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

"""Independent direct-oc fixture reset and OpenShift verification."""

from __future__ import annotations

import json
import subprocess
import time
from typing import Any

from .results import VerificationResult
from .specs import DemoSpec


class OpenShiftVerifier:
    """Verify live demo state without using either MCP server."""

    def __init__(self, spec: DemoSpec, *, oc_timeout_seconds: float = 20.0):
        self.spec = spec
        self.timeout = oc_timeout_seconds

    def preflight(self) -> dict[str, Any]:
        return {"server": self._oc(["whoami", "--show-server"]).strip()}

    def verify(self, namespace: str, *, include_logs: bool = True) -> VerificationResult:
        config = self.spec.verification
        try:
            frontend = self._active_pod(namespace, config["frontend_pod_selector"])
            dependency = self._active_pod(namespace, config["dependency_pod_selector"])
            service = self._oc_json(["-n", namespace, "get", "endpoints", config["service"]])
            ingress = self._oc_json(["-n", namespace, "get", "networkpolicy", config["ingress_policy"]])
            egress = self._oc_json(["-n", namespace, "get", "networkpolicy", config["egress_policy"]])
        except Exception as exc:
            return VerificationResult(
                status="unavailable", passed=False, error=f"{type(exc).__name__}: {exc}"
            )

        frontend_labels = frontend.get("metadata", {}).get("labels", {})
        dependency_labels = dependency.get("metadata", {}).get("labels", {})
        dependency_ip = dependency.get("status", {}).get("podIP")
        endpoints = [
            address.get("ip")
            for subset in service.get("subsets", [])
            for address in subset.get("addresses", [])
            if address.get("ip")
        ]
        ingress_rules = ingress.get("spec", {}).get("ingress", [])
        egress_rules = egress.get("spec", {}).get("egress", [])
        port = config["dependency_port"]
        protocol = config.get("dependency_protocol", "TCP")
        ingress_allows_frontend = any(
            self._selector_matches(source.get("podSelector", {}), frontend_labels)
            and any(p.get("protocol", "TCP") == protocol and p.get("port") == port for p in rule.get("ports", []))
            for rule in ingress_rules
            for source in rule.get("from", [])
        )
        dns_count = sum(self._same(rule, config["dns_rule"]) for rule in egress_rules)
        dependency_rule = {"to": [{"podSelector": {"matchLabels": {"app": "dependency"}}}], "ports": [{"protocol": protocol, "port": port}]}
        dependency_count = sum(self._same(rule, dependency_rule) for rule in egress_rules)

        latest_log = ""
        if include_logs and frontend.get("metadata", {}).get("name"):
            latest_log = self._oc([
                "-n", namespace, "logs", frontend["metadata"]["name"],
                "-c", config["frontend_container"],
                f"--tail={self.spec.verification.get('log_tail_lines', 100)}",
            ])
        log_lines = [
            line for line in latest_log.splitlines()
            if line.startswith(tuple(config.get("dependency_log_prefixes", [])))
        ]
        latest_check = log_lines[-1] if log_lines else ""
        dns_resolved = bool(config["dns_resolved_log_marker"] in latest_check)
        succeeded = latest_check.startswith(config["dependency_success_log_prefix"])
        frontend_ready = self._ready(frontend)
        dependency_ready = self._ready(dependency)
        evidence = {
            "frontend_ready": frontend_ready,
            "dependency_ready": dependency_ready,
            "frontend_phase": frontend.get("status", {}).get("phase"),
            "frontend_restarts": self._restarts(frontend),
            "service_endpoint_count": len(endpoints),
            "service_has_dependency_endpoint": bool(dependency_ip and dependency_ip in endpoints),
            "policy_selects_dependency": self._selector_matches(ingress.get("spec", {}).get("podSelector", {}), dependency_labels),
            "policy_allows_client": ingress_allows_frontend,
            "egress_policy_selects_frontend": self._selector_matches(egress.get("spec", {}).get("podSelector", {}), frontend_labels),
            "egress_policy_allows_dns": dns_count == 1,
            "egress_policy_allows_dependency_port": dependency_count == 1,
            "dns_resolved": dns_resolved,
            "dependency_check_succeeded": succeeded,
            "latest_dependency_log": latest_check,
        }
        passed = all([
            evidence["frontend_ready"], evidence["dependency_ready"],
            evidence["service_has_dependency_endpoint"], evidence["policy_selects_dependency"],
            evidence["policy_allows_client"], evidence["egress_policy_selects_frontend"],
            evidence["egress_policy_allows_dns"], evidence["egress_policy_allows_dependency_port"],
            evidence["dns_resolved"], evidence["dependency_check_succeeded"],
        ])
        return VerificationResult(status="passed" if passed else "failed", passed=passed, evidence=evidence)

    def reset_fixture(self, *, poll_seconds: float = 2.0, wait_seconds: float = 45.0) -> VerificationResult:
        try:
            for side in self.spec.sides.values():
                for item in self.spec.fixture.get("resources", []):
                    resource, name = item["resource"].split("/", 1)
                    self._oc([
                        "-n", side.namespace, "patch", resource, name,
                        "--type", item.get("patch_type", "merge"),
                        "-p", json.dumps(item["patch"]),
                    ])
            deadline = time.monotonic() + wait_seconds
            while time.monotonic() < deadline:
                states = [self.verify(side.namespace, include_logs=False) for side in self.spec.sides.values()]
                if all(self._fixture_is_broken(state) for state in states):
                    return VerificationResult(status="passed", passed=True, evidence={"sides": [s.evidence for s in states]})
                time.sleep(poll_seconds)
            return VerificationResult(status="failed", passed=False, error="fixture did not reach the expected broken state")
        except Exception as exc:
            return VerificationResult(status="unavailable", passed=False, error=f"{type(exc).__name__}: {exc}")

    def _active_pod(self, namespace: str, selector: str) -> dict[str, Any]:
        payload = self._oc_json(["-n", namespace, "get", "pods", "-l", selector])
        pods = [pod for pod in payload.get("items", []) if pod.get("metadata", {}).get("deletionTimestamp") is None]
        if not pods:
            raise RuntimeError(f"no active pod matched {selector!r} in {namespace}")
        return pods[0]

    def _oc(self, args: list[str]) -> str:
        completed = subprocess.run(["oc", *args], capture_output=True, text=True, timeout=self.timeout, check=False)
        if completed.returncode:
            raise RuntimeError(completed.stderr.strip() or f"oc exited {completed.returncode}")
        return completed.stdout

    def _oc_json(self, args: list[str]) -> dict[str, Any]:
        return json.loads(self._oc([*args, "-o", "json"]))

    @staticmethod
    def _ready(pod: dict[str, Any]) -> bool:
        return any(c.get("type") == "Ready" and c.get("status") == "True" for c in pod.get("status", {}).get("conditions", []))

    @staticmethod
    def _restarts(pod: dict[str, Any]) -> int:
        return sum(int(c.get("restartCount", 0)) for c in pod.get("status", {}).get("containerStatuses", []))

    @staticmethod
    def _selector_matches(selector: dict[str, Any], labels: dict[str, str]) -> bool:
        return all(labels.get(k) == v for k, v in selector.get("matchLabels", {}).items())

    @classmethod
    def _same(cls, actual: Any, expected: Any) -> bool:
        if isinstance(actual, dict) and isinstance(expected, dict):
            return actual.keys() == expected.keys() and all(cls._same(actual[k], expected[k]) for k in actual)
        if isinstance(actual, list) and isinstance(expected, list):
            return sorted((json.dumps(v, sort_keys=True) for v in actual)) == sorted(json.dumps(v, sort_keys=True) for v in expected)
        return actual == expected

    def _fixture_is_broken(self, result: VerificationResult) -> bool:
        evidence = result.evidence
        return (
            evidence.get("service_has_dependency_endpoint") is True
            and evidence.get("policy_selects_dependency") is True
            and evidence.get("policy_allows_client") is False
            and evidence.get("egress_policy_allows_dns") is True
            and evidence.get("egress_policy_allows_dependency_port") is False
        )

from pathlib import Path

import pytest

from llm_memory_bench.demo import load_demo_spec


CONFIG = Path("scenarios/openshift/network-policy.yaml")


def test_network_policy_demo_config_loads():
    spec = load_demo_spec(CONFIG)

    assert spec.demo_id == "network-policy@1.0.0"
    assert spec.sides["memory"].namespace == "demo-memory"
    assert spec.sides["baseline"].namespace == "demo-baseline"
    assert spec.memory_search_required_before_change is True
    assert spec.verification["dependency_port"] == 8080


def test_network_policy_config_has_disjoint_tool_classes():
    spec = load_demo_spec(CONFIG)

    assert set(spec.read_only_tools).isdisjoint(spec.change_making_tools)
    assert "resources_create_or_update" in spec.change_making_tools
    assert spec.sides["baseline"].memoryhub_tools == []


def test_loader_rejects_non_mapping(tmp_path):
    path = tmp_path / "invalid.yaml"
    path.write_text("- not-a-demo-mapping\n")

    with pytest.raises(ValueError, match="one YAML mapping"):
        load_demo_spec(path)


def test_loader_reports_missing_file(tmp_path):
    with pytest.raises(ValueError, match="Unable to read"):
        load_demo_spec(tmp_path / "missing.yaml")

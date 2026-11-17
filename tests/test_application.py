import asyncio
import json
from pathlib import Path

from llm_memory_bench.application import (
    EvaluationSpec,
    ExperimentOutput,
    ExtractionExperiment,
    RunSpec,
    Tau2Experiment,
    ValueExperiment,
)
from llm_memory_bench.application.run_service import RunService, _jsonable
from llm_memory_bench.value.container_runner import _parse_agent_response


def test_value_run_spec_maps_to_existing_config():
    spec = RunSpec(
        dataset_path="scenarios/test-value.yaml",
        provider="litellm",
        model="test-model",
        system="simple",
        mode="both",
        max_scenarios=2,
    )

    config = spec.value_config()

    assert config.injection_mode == "both"
    assert config.max_scenarios == 2
    assert config.user_simulator_provider == "litellm"
    assert config.judge_model == "test-model"


def test_value_run_spec_maps_memory_setup_mode():
    spec = RunSpec(
        dataset_path="scenarios/test-value.yaml",
        execution="container",
        container_config_path="configs/claude-memoryhub-remote.yaml",
        memory_setup_mode="capture_first",
    )

    assert spec.value_config().memory_setup_mode == "capture_first"
    assert spec.container_value_config("claude-test").memory_setup_mode == "capture_first"


def test_container_value_run_spec_maps_to_claude_and_remote_judging():
    spec = RunSpec(
        execution="container",
        container_config_path="configs/claude-memoryhub-remote.yaml",
        dataset_path="scenarios/test-value.yaml",
        judge_provider="openai",
        judge_model="gpt-4.1-mini",
        user_simulator_provider="openai",
        user_simulator_model="gpt-4o-mini",
    )

    config = spec.container_value_config(
        "claude-sonnet-4@20250514",
    )

    assert config.provider == "anthropic"
    assert config.model == "claude-sonnet-4@20250514"
    assert config.system == "memoryhub"
    assert config.injection_mode == "integration"
    assert config.judge_model == "gpt-4.1-mini"
    assert config.user_simulator_model == "gpt-4o-mini"


def test_run_service_uses_stable_run_id_shape(tmp_path):
    service = RunService(tmp_path)
    spec = RunSpec(dataset_path="scenarios/test-value.yaml")

    run_id = service._run_id(spec)

    assert run_id.endswith("_value_openai_gpt-4o_simple")


def test_experiment_implementations_have_distinct_kinds(tmp_path):
    assert ExtractionExperiment().kind == "extraction"
    assert ValueExperiment().kind == "value"
    assert Tau2Experiment(tmp_path).kind == "tau2"


def test_run_service_dispatches_to_experiment_implementation(tmp_path, monkeypatch):
    service = RunService(tmp_path)
    spec = RunSpec(kind="extraction", dataset_path="datasets/test_sequential.yaml")
    calls = []

    async def fake_extraction_run(received_spec, run_id):
        calls.append((received_spec, run_id))
        return ExperimentOutput({"run": {"conversation_results": []}})

    monkeypatch.setattr(service.extraction, "run", fake_extraction_run)
    asyncio.run(service.execute(spec))

    assert calls[0][0] is spec
    assert calls[0][1].endswith("_extraction_openai_gpt-4o_simple")


def test_jsonable_serializes_run_spec():
    spec = RunSpec(dataset_path="scenarios/test-value.yaml")

    result = _jsonable(spec)

    assert result["kind"] == "value"
    assert result["dataset_path"] == "scenarios/test-value.yaml"


def test_evaluation_spec_defaults_to_openai_hybrid():
    spec = EvaluationSpec(result_paths=["results/run.json"])

    assert spec.matcher == "hybrid"
    assert spec.judge_provider == "openai"
    assert spec.judge_model == "gpt-4o"


def test_run_service_converts_native_stored_memories_to_storage_calls():
    run_data = {
        "config": {"system": "memoryhub", "provider": "claude-code-container", "model": "test"},
        "conversations": [{
            "conversation_id": "conversation-1",
            "stored": [{"content": "User prefers early flights."}],
            "turns": [{
                "turn_index": 0,
                "role": "assistant",
                "tool_calls": [{
                    "name": "mcp__memoryhub__memory",
                    "arguments": {"action": "search", "query": "flight"},
                }],
            }],
        }],
    }

    result = RunService._run_result_from_json(run_data)
    calls = result.conversation_results[0].turn_results[0].tool_calls

    assert calls[-1].name == "memory"
    assert calls[-1].arguments["action"] == "write"
    assert calls[-1].arguments["content"] == "User prefers early flights."


def test_run_service_writes_result_raw_output_and_log_artifacts(tmp_path, monkeypatch):
    service = RunService(tmp_path)
    spec = RunSpec(dataset_path="scenarios/test-value.yaml")

    async def fake_value_run(_spec, _run_id):
        return ExperimentOutput(
            {"run": {"scenario_results": [{"scenario_id": "goal-1"}]}}
        )

    monkeypatch.setattr(service.value, "run", fake_value_run)
    payload = asyncio.run(service.execute(spec))

    for key in ("output_path", "raw_output_path", "log_path"):
        assert Path(payload[key]).exists()

    output = json.loads((tmp_path / f"{payload['run_id']}.json").read_text())
    raw = json.loads((tmp_path / f"{payload['run_id']}.raw.json").read_text())
    assert output["result"] == raw["result"]
    assert "scenario_results=1" in (tmp_path / f"{payload['run_id']}.log").read_text()


def test_parse_agent_response_preserves_tool_calls():
    raw = json.dumps({
        "responses": [{
            "response": "I found the preference.",
            "session_id": "session-1",
            "usage": {"input_tokens": 12, "output_tokens": 8},
            "tool_calls": [{
                "name": "memoryhub_search",
                "arguments": {"query": "coffee"},
            }],
        }],
    })

    text, session_id, usage, tool_calls, integration_events = _parse_agent_response(raw)

    assert text == "I found the preference."
    assert session_id == "session-1"
    assert usage == {"input_tokens": 12, "output_tokens": 8}
    assert tool_calls == [{
        "name": "memoryhub_search",
        "arguments": {"query": "coffee"},
    }]
    assert integration_events == []

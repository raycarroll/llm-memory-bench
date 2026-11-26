from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_memory_bench.application import RunSpec
from llm_memory_bench.dataset import Conversation, Turn
from llm_memory_bench.real_system_container_extraction_runner import (
    drive_conversation,
)
from llm_memory_bench.value.clbench import (
    load_seed_memories,
    run_clbench_value_benchmark,
)
from llm_memory_bench.value.clbench_container import (
    _require_session_continuity,
    _write_prompt_file,
)


def test_clbench_run_spec_defaults_to_none_memory_control() -> None:
    spec = RunSpec(dataset_path=".cl-bench", value_source="clbench")

    assert spec.memory_setup_mode == "none"
    assert spec.clbench_task == "database_exploration"
    assert spec.clbench_schedule == "default"


def test_load_clbench_seed_memories_accepts_wrapped_yaml(tmp_path: Path) -> None:
    path = tmp_path / "memories.yaml"
    path.write_text(
        "memories:\n"
        "  - fact: The user prefers concise SQL answers.\n"
        "    category: preference\n"
    )

    assert load_seed_memories(path) == [
        {
            "fact": "The user prefers concise SQL answers.",
            "category": "preference",
        }
    ]


def test_clbench_adapter_uses_external_worker(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "cl-bench"
    (root / "src").mkdir(parents=True)
    (root / "src" / "interface.py").write_text("")
    interpreter = tmp_path / "python"
    interpreter.write_text("")

    class Completed:
        returncode = 0
        stdout = json.dumps({"harness": "clbench", "comparison": {"gain": 0.1}})
        stderr = ""

    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return Completed()

    monkeypatch.setattr("llm_memory_bench.value.clbench.subprocess.run", fake_run)
    result = run_clbench_value_benchmark(
        clbench_root=root,
        clbench_python=str(interpreter),
        task="database_exploration",
        schedule="default",
        variant=None,
        max_instances=2,
        seed=42,
        memory_setup_mode="capture_first",
        memory_system="simple",
        memory_prompt_version=None,
        seed_memories_path=None,
        model="openai/gpt-4.1",
    )

    assert result["harness"] == "clbench"
    assert calls[0][0][-2:] == ["-m", "llm_memory_bench.value.clbench_worker"]
    request = json.loads(calls[0][1]["input"])
    assert request["memory_setup_mode"] == "capture_first"
    assert request["max_instances"] == 2


def test_session_continuity_accepts_resumed_session() -> None:
    session_id, status = _require_session_continuity(
        "session-1",
        "session-1",
        "session-1",
    )

    assert session_id == "session-1"
    assert status == "resumed"


def test_session_continuity_rejects_missing_resumed_session() -> None:
    with pytest.raises(RuntimeError, match="continuity lost"):
        _require_session_continuity("session-1", None, None)


def test_session_continuity_rejects_changed_session() -> None:
    with pytest.raises(RuntimeError, match="continuity lost"):
        _require_session_continuity("session-1", "session-2", "session-2")


def test_session_continuity_rejects_conflicting_driver_ids() -> None:
    with pytest.raises(RuntimeError, match="conflicting session IDs"):
        _require_session_continuity(None, "session-1", "session-2")


def test_clbench_prompt_materialization_selects_requested_mode() -> None:
    class PromptSystem:
        def system_prompt_for_mode(self, mode: str) -> str:
            return f"prompt for {mode}"

    path = _write_prompt_file(PromptSystem(), "capture_only")
    try:
        assert path.read_text(encoding="utf-8") == "prompt for capture_only"
        assert "capture_only" in path.name
    finally:
        path.unlink()


def test_container_driver_receives_schema_outside_task_prompt(
    tmp_path: Path, monkeypatch
) -> None:
    captured: dict[str, object] = {}

    def fake_docker(*args, **kwargs):
        del kwargs
        if args and args[0] == "cp":
            captured["turns"] = json.loads(Path(args[1]).read_text(encoding="utf-8"))

    class Completed:
        returncode = 0
        stdout = json.dumps({"responses": []})
        stderr = ""

    def fake_exec(*args, **kwargs):
        del args, kwargs
        return Completed()

    monkeypatch.setattr(
        "llm_memory_bench.real_system_container_extraction_runner._docker",
        fake_docker,
    )
    monkeypatch.setattr(
        "llm_memory_bench.real_system_container_extraction_runner._exec",
        fake_exec,
    )

    schema = {
        "type": "object",
        "properties": {"action": {"type": "string"}},
        "required": ["action"],
    }
    drive_conversation(
        "container",
        Conversation(id="test", turns=[Turn(role="user", content="task text")]),
        response_schema=schema,
    )

    turn = captured["turns"][0]
    assert turn["content"] == "task text"
    assert turn["response_schema"] == schema


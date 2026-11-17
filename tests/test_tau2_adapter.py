from pathlib import Path

import pytest

from llm_memory_bench.value.tau2 import (
    Tau2AgentTelemetry,
    Tau2ArmResult,
    Tau2MemoryBridge,
    TAU2_MEMORYHUB_PROMPT_VERSION,
    Tau2MemoryPlan,
    Tau2UnavailableError,
    _make_agent,
    load_tau2_memory_plan,
)


def test_tau2_memory_plan_loads_and_indexes_task_ids(tmp_path: Path) -> None:
    path = tmp_path / "memory-plan.yaml"
    path.write_text(
        """\
domain: airline
source_revision: test-revision
tasks:
  - task_id: \"3\"
    memory_bank:
      - fact: The user prefers numeric answers.
        category: preference
"""
    )

    plan = load_tau2_memory_plan(path)

    assert plan.domain == "airline"
    assert plan.source_revision == "test-revision"
    assert plan.for_task("3").memory_bank[0].fact == "The user prefers numeric answers."


def test_tau2_memory_plan_rejects_missing_task() -> None:
    plan = Tau2MemoryPlan(domain="retail", tasks=[])

    with pytest.raises(ValueError, match="No memory fixture"):
        plan.for_task("missing")


def test_tau2_memory_bridge_can_disable_memory_tools() -> None:
    class FakeMemorySystem:
        def tool_definitions_for_mode(self, _mode):
            raise AssertionError("disabled bridges must not expose memory tools")

    bridge = Tau2MemoryBridge(FakeMemorySystem(), mode="disabled")

    assert bridge.tool_names == set()
    assert bridge.tools() == []


def test_tau2_runner_import_is_optional() -> None:
    # The repository test environment does not need tau2-bench installed just
    # to import the adapter and validate plans.
    assert issubclass(Tau2UnavailableError, RuntimeError)


def test_tau2_arm_result_preserves_effective_system_prompt() -> None:
    result = Tau2ArmResult(
        task_id="3",
        mode="memory_prompt",
        reward=1.0,
        telemetry=Tau2AgentTelemetry(memory_chars=42),
        system_prompt="system prompt with injected memory",
        simulation={},
    )

    assert result.as_dict()["system_prompt"] == "system prompt with injected memory"


def test_tau2_telemetry_prefers_provider_reported_tokens() -> None:
    telemetry = Tau2AgentTelemetry(
        input_chars=400,
        output_chars=200,
        input_tokens=123,
        output_tokens=45,
        usage_records=1,
    )

    data = telemetry.as_dict()

    assert data["token_usage_exact"] is True
    assert data["total_tokens"] == 168
    assert data["token_usage_note"] == "provider-reported LiteLLM usage"


def test_tau2_memory_capture_records_writes_without_preloading() -> None:
    class FakeMemorySystem:
        def accepted_tool_names(self):
            return {"save_memory"}

        def tool_definitions(self):
            return [
                {
                    "name": "save_memory",
                    "description": "Save a fact",
                    "input_schema": {"type": "object"},
                }
            ]

        def extract_stored_fact(self, tool_call):
            return tool_call.arguments.get("fact", "")

        def format_tool_result(self, tool_call):
            return {"status": "success"}

    capture = Tau2MemoryBridge(FakeMemorySystem())

    assert capture.initial_memories == []
    assert capture.handle("save_memory", {"fact": "User prefers aisle seats"}) == {
        "status": "success"
    }
    assert capture.captured_memories == [
        {
            "fact": "User prefers aisle seats",
            "category": "captured",
            "source": "baseline_conversation",
            "tool": "save_memory",
        }
    ]


def test_tau2_memoryhub_baseline_uses_ben_write_only_variant() -> None:
    from llm_memory_bench.systems import get_system

    memory = get_system("memoryhub", prompt_version=TAU2_MEMORYHUB_PROMPT_VERSION)
    bridge = Tau2MemoryBridge(memory, mode="capture_only")
    schema = memory.tool_definitions_for_mode("capture_only")[0]

    assert memory.prompt_version == TAU2_MEMORYHUB_PROMPT_VERSION
    assert "Do not search" in memory.system_prompt_for_mode("capture_only")
    assert schema["input_schema"]["properties"]["action"]["enum"] == ["write"]
    assert bridge.tool_names == {"memory"}
    assert memory.validate_tool_call_for_mode(
        type("Call", (), {"name": "memory", "arguments": {"action": "search"}})(),
        "capture_only",
    )


def test_memoryhub_live_adapter_executes_scoped_writes() -> None:
    from llm_memory_bench.value.memoryhub_live import MemoryHubLiveAdapter

    class FakeClient:
        def __init__(self):
            self.writes = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def write(self, content, **kwargs):
            self.writes.append((content, kwargs))
            return {"memory": {"id": "mh-1", "content": content}}

    client = FakeClient()
    adapter = MemoryHubLiveAdapter(
        project_id="project-1",
        prompt_version=TAU2_MEMORYHUB_PROMPT_VERSION,
        client_factory=lambda: client,
    )

    response = Tau2MemoryBridge(adapter, mode="capture_only").handle(
        "memory", {"action": "write", "content": "User is Gold"}
    )

    assert response["memory"]["id"] == "mh-1"
    assert client.writes[0][1]["project_id"] == "project-1"


def test_tau2_agent_flattens_multi_tool_results_before_next_request() -> None:
    class FakeBaseAgent:
        @classmethod
        def __class_getitem__(cls, _item):
            return cls

        def __init__(self, tools, domain_policy):
            self.tools = tools
            self.domain_policy = domain_policy

    class FakeMessage:
        def __init__(self, content=None):
            self.content = content

    class FakeSystemMessage(FakeMessage):
        def __init__(self, content=None, **_kwargs):
            super().__init__(content)

    class FakeToolMessage(FakeMessage):
        pass

    class FakeMultiToolMessage:
        def __init__(self, tool_messages):
            self.tool_messages = tool_messages

    class FakeAssistantMessage(FakeMessage):
        def __init__(self, content=None, tool_calls=None):
            super().__init__(content)
            self.tool_calls = tool_calls
            self.usage = None

    requests = []

    def fake_generate(**kwargs):
        requests.append(kwargs["messages"])
        return FakeAssistantMessage(content="ok")

    tau2 = {
        "HalfDuplexAgent": FakeBaseAgent,
        "AssistantMessage": FakeAssistantMessage,
        "Message": FakeMessage,
        "MultiToolMessage": FakeMultiToolMessage,
        "SystemMessage": FakeSystemMessage,
        "generate": fake_generate,
    }
    telemetry = Tau2AgentTelemetry()
    agent, _ = _make_agent(
        tau2,
        telemetry,
        tools=[],
        domain_policy="policy",
        model="test-model",
        memory_system=None,
        memory_facts=[],
    )

    state = agent.get_init_state()
    agent.generate_next_message(FakeMessage("first"), state)
    results = [FakeToolMessage("result 1"), FakeToolMessage("result 2")]
    agent.generate_next_message(FakeMultiToolMessage(results), state)

    assert requests[1][-2:] == results

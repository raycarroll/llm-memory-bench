from __future__ import annotations

import pytest

from llm_memory_bench.providers.base import ProviderResponse, ToolCall
from llm_memory_bench.systems import get_system
from llm_memory_bench.value.dataset import ValueScenario
from llm_memory_bench.value.runner import run_value_trial


class FakeProvider:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    async def generate(self, *, messages, tools, system):
        self.requests.append({"messages": messages, "tools": tools, "system": system})
        return self.responses.pop(0)

    def build_tool_result_messages(self, response, format_tool_result=None):
        formatter = format_tool_result or (lambda _call: {"status": "success"})
        return [
            {
                "role": "tool",
                "content": formatter(call),
            }
            for call in response.tool_calls
        ]


@pytest.mark.asyncio
async def test_capture_first_returns_baseline_memories_for_handoff():
    scenario = ValueScenario.model_validate(
        {
            "id": "capture-test",
            "memory_bank": [{"fact": "User likes tea", "category": "preference"}],
            "user_facts": [{"fact": "User likes tea", "category": "preference"}],
            "task_prompt": "Suggest a drink.",
            "persona_prompt": "The user likes tea.",
            "outcomes": [],
        }
    )
    agent = FakeProvider(
        [
            ProviderResponse(
                text="",
                tool_calls=[
                    ToolCall(
                        name="add_memory",
                        arguments={
                            "fact": "User likes tea",
                            "category": "preference",
                        },
                    )
                ],
            ),
            ProviderResponse(
                text="I will use that preference to suggest a suitable drink for you.",
            ),
        ]
    )
    user = FakeProvider([])

    result = await run_value_trial(
        scenario,
        agent,
        user,
        get_system("simple"),
        "baseline",
        max_turns=2,
        initial_memories=[],
        capture_memories=True,
    )

    assert result.initial_memories == []
    assert result.captured_memories[0]["fact"] == "User likes tea"
    assert result.memory_events[0]["tool"] == "add_memory"
    assert any(tool["name"] == "add_memory" for tool in agent.requests[0]["tools"])

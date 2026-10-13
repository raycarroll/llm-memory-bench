from __future__ import annotations

from llm_memory_bench.value.container_runner import _scenario_seed_plan
from llm_memory_bench.value.dataset import ValueScenario


def test_value_scenario_seeds_are_project_scoped() -> None:
    scenario = ValueScenario.model_validate(
        {
            "id": "location-test",
            "memory_bank": [
                {"fact": "User lives in Portland, Oregon", "category": "personal"},
            ],
            "task_prompt": "Suggest dinner.",
            "persona_prompt": "You live in Portland, Oregon.",
            "outcomes": [],
        }
    )

    plan = _scenario_seed_plan(scenario, "memoryhub")

    assert [memory.scope for memory in plan.memories] == ["project"]
    assert [memory.force for memory in plan.memories] == [True]

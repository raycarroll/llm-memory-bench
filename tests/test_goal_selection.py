from pathlib import Path

from llm_memory_bench.application import RunSpec
from llm_memory_bench.application.run_service import RunService
from llm_memory_bench.value.dataset import load_value_dataset


def test_quick_demo_exposes_goal_definitions():
    dataset = load_value_dataset(Path("scenarios/quick-demo-goal.yaml"))

    assert dataset.scenarios
    assert all(scenario.outcomes for scenario in dataset.scenarios)
    assert all(scenario.memory_bank for scenario in dataset.scenarios)


def test_run_spec_carries_selected_goal():
    spec = RunSpec(
        dataset_path="scenarios/quick-demo-goal.yaml",
        goal_id="quick-coffee-location",
    )

    assert spec.goal_id == "quick-coffee-location"
    assert RunService._run_id(spec).endswith("_value_openai_gpt-4o_simple")

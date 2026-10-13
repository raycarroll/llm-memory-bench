from pathlib import Path

from llm_memory_bench.value.dataset import load_value_dataset


def test_curated_tau2_dataset_loads_as_goal_scenarios():
    dataset = load_value_dataset(Path("scenarios/tau2/airline-retail.yaml"))

    assert len(dataset.scenarios) == 10
    assert len({scenario.id for scenario in dataset.scenarios}) == 10
    assert all(scenario.task_prompt for scenario in dataset.scenarios)
    assert all(scenario.memory_bank for scenario in dataset.scenarios)
    assert all(scenario.outcomes for scenario in dataset.scenarios)
    assert all(scenario.validators for scenario in dataset.scenarios)

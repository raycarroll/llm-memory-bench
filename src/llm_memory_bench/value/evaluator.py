from __future__ import annotations

from dataclasses import dataclass, field

from ..config import RunConfig, ValueRunConfig, get_provider
from .dataset import ValueDataset, ValueScenario
from .metrics import aggregate_metrics, compute_value_metrics
from .runner import TrialResult, ValueRunResult


QUALITY_JUDGE_PROMPT = """\
You are evaluating an AI assistant's response to a user request.

User's request: {task_prompt}

Assistant's response:
{response}

Evaluation rubric:
{rubric}

Score the response from 0.0 to 1.0 based on the rubric. Consider:
- Did the response address the user's actual needs?
- Was the response personalized and specific (vs generic and hedging)?
- Did the assistant avoid asking unnecessary clarifying questions?
- Was the response complete and actionable?

Respond with ONLY a decimal number between 0.0 and 1.0."""


async def judge_quality(
    task_prompt: str,
    response: str,
    rubric: str,
    judge_provider,
) -> float:
    prompt = QUALITY_JUDGE_PROMPT.format(
        task_prompt=task_prompt,
        response=response,
        rubric=rubric,
    )

    result = await judge_provider.generate(
        messages=[{"role": "user", "content": prompt}],
        tools=[],
        system="You are a precise evaluator. Respond with exactly one decimal number.",
    )

    text = result.text.strip()
    try:
        score = float(text)
        return max(0.0, min(1.0, score))
    except ValueError:
        import re

        match = re.search(r"(\d+\.?\d*)", text)
        if match:
            return max(0.0, min(1.0, float(match.group(1))))
        return 0.5


@dataclass
class ScenarioEvaluation:
    scenario_id: str
    metrics_by_mode: dict[str, dict] = field(default_factory=dict)


@dataclass
class ValueEvaluationResult:
    scenario_evaluations: list[ScenarioEvaluation] = field(default_factory=list)

    def aggregate(self) -> dict[str, dict]:
        by_mode: dict[str, list[dict]] = {}
        for se in self.scenario_evaluations:
            for mode, metrics in se.metrics_by_mode.items():
                by_mode.setdefault(mode, []).append(metrics)

        return {mode: aggregate_metrics(items) for mode, items in by_mode.items()}


async def evaluate_value_run(
    dataset: ValueDataset,
    run_result: ValueRunResult,
    judge_config: RunConfig | None = None,
) -> ValueEvaluationResult:
    if judge_config is None:
        judge_config = RunConfig(
            provider=run_result.config.judge_provider,
            model=run_result.config.judge_model,
            api_key_env=run_result.config.api_key_env,
        )

    judge_provider = get_provider(judge_config)
    scenario_map = {s.id: s for s in dataset.scenarios}

    eval_result = ValueEvaluationResult()

    for scenario_result in run_result.scenario_results:
        scenario = scenario_map.get(scenario_result.scenario_id)
        if not scenario:
            continue

        baseline_trial = scenario_result.trials.get("baseline")
        if not baseline_trial:
            continue

        quality_baseline = await judge_quality(
            scenario.task_prompt,
            baseline_trial.final_response,
            scenario.quality_rubric,
            judge_provider,
        )

        scenario_eval = ScenarioEvaluation(scenario_id=scenario.id)

        for mode, trial in scenario_result.trials.items():
            if mode == "baseline":
                continue

            quality_memory = await judge_quality(
                scenario.task_prompt,
                trial.final_response,
                scenario.quality_rubric,
                judge_provider,
            )

            metrics = compute_value_metrics(
                baseline=baseline_trial,
                memory=trial,
                quality_baseline=quality_baseline,
                quality_memory=quality_memory,
            )
            scenario_eval.metrics_by_mode[mode] = metrics

        eval_result.scenario_evaluations.append(scenario_eval)

    return eval_result

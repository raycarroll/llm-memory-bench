from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from ..config import RunConfig, get_provider
from .dataset import Outcome, ValueDataset, ValueScenario, ValidatorSpec
from .metrics import aggregate_metrics, compute_value_metrics
from .runner import TrialResult, ValueRunResult


OUTCOME_JUDGE_PROMPT = """\
You are evaluating whether an AI assistant achieved specific outcomes for a user.

User's request:
{task_prompt}

Assistant's response:
{response}

Memory available to the assistant:
{memory_context}

Conversation context, including facts revealed by the user:
{conversation_context}

Outcomes to evaluate:
{outcomes}

For each outcome, return an object with:
- status: one of "passed", "failed", or "uncertain"
- score: a number from 0.0 to 1.0
- explanation: concise evidence from the response, including which available
  memory fact(s), if any, contributed to the outcome. Do not claim memory
  contributed if the response does not use it.
- evidence: short quotes or concrete observations supporting the judgement

Respond with ONLY valid JSON in this form:
{{"outcomes": {{"outcome_id": {{"status": "passed", "score": 1.0, "explanation": "..."}}}}}}
"""


def _parse_outcome_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
    try:
        parsed = json.loads(text)
        return parsed.get("outcomes", {}) if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


async def validate_outcomes(
    task_prompt: str,
    response: str,
    outcomes: list[Outcome],
    validator: ValidatorSpec,
    judge_provider,
    memory_facts: list[str] | None = None,
    conversation: list[dict] | None = None,
) -> list[dict]:
    """Run the initial LLM-judge validator and return extensible results."""
    if validator.type != "llm_judge":
        return [
            {
                "outcome_id": outcome.id,
                "status": "error",
                "score": None,
                "explanation": f"Unsupported validator type: {validator.type}",
                "evidence": {},
            }
            for outcome in outcomes
        ]

    selected_ids = set(validator.outcomes)
    selected = [o for o in outcomes if not selected_ids or o.id in selected_ids]
    outcome_text = "\n".join(f"- {o.id}: {o.statement}" for o in selected)
    memory_context = "\n".join(f"- {fact}" for fact in (memory_facts or [])) or "None"
    conversation_context = "\n".join(
        f"{turn.get('role', 'unknown')}: {turn.get('content', '')}"
        for turn in (conversation or [])
        if turn.get("content")
    ) or "No additional conversation context"
    result = await judge_provider.generate(
        messages=[
            {
                "role": "user",
                "content": OUTCOME_JUDGE_PROMPT.format(
                    task_prompt=task_prompt,
                    response=response,
                    memory_context=memory_context,
                    conversation_context=conversation_context,
                    outcomes=outcome_text,
                ),
            }
        ],
        tools=[],
        system="You are a precise evaluator. Return only valid JSON.",
    )
    judged = _parse_outcome_json(result.text)

    validations = []
    for outcome in selected:
        value = judged.get(outcome.id, {})
        score = value.get("score") if isinstance(value, dict) else None
        if isinstance(score, (int, float)):
            score = max(0.0, min(1.0, float(score)))
        else:
            score = None
        status = value.get("status", "uncertain") if isinstance(value, dict) else "uncertain"
        validations.append(
            {
                "outcome_id": outcome.id,
                "status": status,
                "score": score,
                "explanation": value.get("explanation", "") if isinstance(value, dict) else "",
                "evidence": value.get("evidence", {}) if isinstance(value, dict) else {},
            }
        )
    return validations


def _quality_score(validations: list[dict]) -> float:
    scores = [v["score"] for v in validations if isinstance(v.get("score"), (int, float))]
    return sum(scores) / len(scores) if scores else 0.5


@dataclass
class ScenarioEvaluation:
    scenario_id: str
    metrics_by_mode: dict[str, dict] = field(default_factory=dict)
    validations_by_mode: dict[str, list[dict]] = field(default_factory=dict)


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

        baseline_validations: list[dict] = []
        for validator in scenario.effective_validators():
            baseline_validations.extend(
                await validate_outcomes(
                    scenario.task_prompt,
                    baseline_trial.final_response,
                    scenario.effective_outcomes(),
                    validator,
                    judge_provider,
                    memory_facts=[],
                    conversation=[
                        {"role": turn.role, "content": turn.content}
                        for turn in baseline_trial.turns
                    ],
                )
            )
        quality_baseline = _quality_score(baseline_validations)

        scenario_eval = ScenarioEvaluation(scenario_id=scenario.id)
        scenario_eval.validations_by_mode["baseline"] = baseline_validations

        for mode, trial in scenario_result.trials.items():
            if mode == "baseline":
                continue

            memory_validations: list[dict] = []
            memory_facts = [memory.fact for memory in scenario.memory_bank]
            for validator in scenario.effective_validators():
                memory_validations.extend(
                    await validate_outcomes(
                        scenario.task_prompt,
                        trial.final_response,
                        scenario.effective_outcomes(),
                        validator,
                        judge_provider,
                        memory_facts=memory_facts,
                        conversation=[
                            {"role": turn.role, "content": turn.content}
                            for turn in trial.turns
                        ],
                    )
                )
            quality_memory = _quality_score(memory_validations)
            scenario_eval.validations_by_mode[mode] = memory_validations

            metrics = compute_value_metrics(
                baseline=baseline_trial,
                memory=trial,
                quality_baseline=quality_baseline,
                quality_memory=quality_memory,
            )
            scenario_eval.metrics_by_mode[mode] = metrics

        eval_result.scenario_evaluations.append(scenario_eval)

    return eval_result

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ..dataset import load_dataset
from ..bench import RunConfig as BenchRunConfig
from ..config import RunConfig
from ..evaluator import evaluate_run
from ..value.dataset import load_value_dataset
from ..value.container_runner import run_container_value_benchmark
from ..value.evaluator import evaluate_value_run
from ..value.runner import run_value_benchmark
from ..runner import run_benchmark
from .specs import RunSpec


def _jsonable(value: Any) -> Any:
    """Convert benchmark dataclasses and Pydantic models to JSON data."""
    if isinstance(value, BaseModel):
        return _jsonable(value.model_dump(mode="json"))
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


class RunService:
    """Run existing benchmark engines behind a UI/backend-neutral boundary."""

    def __init__(self, results_dir: Path | str = "results/ui"):
        self.results_dir = Path(results_dir)

    async def execute(self, spec: RunSpec) -> dict[str, Any]:
        run_id = self._run_id(spec)
        if spec.kind == "value":
            result = await self._execute_value(spec)
        else:
            result = await self._execute_extraction(spec)

        result_json = _jsonable(result)
        output_path = self.results_dir / f"{run_id}.json"
        raw_output_path = self.results_dir / f"{run_id}.raw.json"
        log_path = self.results_dir / f"{run_id}.log"
        payload = {
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "spec": spec.model_dump(mode="json"),
            "result": result_json,
            "output_path": str(output_path),
            "raw_output_path": str(raw_output_path),
            "log_path": str(log_path),
        }
        self.results_dir.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, default=str))
        raw_payload = {
            "run_id": run_id,
            "created_at": payload["created_at"],
            "spec": payload["spec"],
            "result": result_json,
        }
        raw_output_path.write_text(json.dumps(raw_payload, indent=2, default=str))
        log_path.write_text(self._build_log(payload))
        return payload

    @staticmethod
    def _build_log(payload: dict[str, Any]) -> str:
        """Write a small, stable execution log without secrets or prompts."""
        spec = payload.get("spec", {})
        result = payload.get("result", {})
        lines = [
            f"run_id={payload.get('run_id', '')}",
            f"created_at={payload.get('created_at', '')}",
            f"kind={spec.get('kind', '')}",
            f"execution={spec.get('execution', '')}",
            f"provider={spec.get('provider', '')}",
            f"model={spec.get('model', '')}",
            f"system={spec.get('system', '')}",
            f"dataset={spec.get('dataset_path', '')}",
        ]
        if result.get("run", {}).get("scenario_results") is not None:
            lines.append(
                f"scenario_results={len(result['run'].get('scenario_results', []))}"
            )
        if result.get("run", {}).get("conversation_results") is not None:
            lines.append(
                f"conversation_results={len(result['run'].get('conversation_results', []))}"
            )
        lines.extend([
            f"output_json={payload.get('output_path', '')}",
            f"raw_output_json={payload.get('raw_output_path', '')}",
        ])
        events = []
        for scenario in result.get("run", {}).get("scenario_results", []):
            for trial in scenario.get("trials", {}).values():
                for turn in trial.get("turns", []):
                    events.extend(turn.get("integration_events", []))
        if events:
            lines.append("")
            lines.append("integration_events:")
            for event in events:
                lines.append(json.dumps(event, sort_keys=True))
        return "\n".join(lines) + "\n"

    def execute_sync(self, spec: RunSpec) -> dict[str, Any]:
        """Synchronous entry point for Streamlit callbacks and CLI callers."""
        return asyncio.run(self.execute(spec))

    async def _execute_value(self, spec: RunSpec) -> dict[str, Any]:
        dataset = load_value_dataset(spec.dataset())
        if spec.goal_id:
            selected = [scenario for scenario in dataset.scenarios if scenario.id == spec.goal_id]
            if not selected:
                raise ValueError(f"Goal not found in dataset: {spec.goal_id}")
            dataset = dataset.model_copy(update={"scenarios": selected})
        if spec.execution == "container":
            return await self._execute_container_value(spec, dataset)

        config = spec.value_config()
        run_result = await run_value_benchmark(dataset, config)
        evaluation = await evaluate_value_run(dataset, run_result)
        return self._value_result_payload(dataset, run_result, evaluation)

    async def _execute_container_value(self, spec: RunSpec, dataset) -> dict[str, Any]:
        if not spec.container_config_path:
            raise ValueError(
                "container_config_path is required for container value runs"
            )

        container_config = BenchRunConfig.from_yaml(Path(spec.container_config_path))
        if not container_config.integration_profile:
            raise ValueError(
                "Container value runs require a config with integration.profile"
            )

        # The profile directory (for example ``remote-http-v1``) is not an
        # API-level system registry key. Keep the container profile name in
        # result metadata, but use the registered provider name internally.
        value_config = spec.container_value_config(container_config.model)
        run_result = await run_container_value_benchmark(
            dataset,
            container_config,
            value_config,
        )
        judge_config = RunConfig(
            provider=value_config.judge_provider,
            model=value_config.judge_model,
            api_key_env=value_config.api_key_env,
        )
        evaluation = await evaluate_value_run(dataset, run_result, judge_config)
        payload = self._value_result_payload(dataset, run_result, evaluation)
        payload["container"] = {
            "host": {
                "name": container_config.host_name,
                "version": container_config.host_version,
            },
            "model": container_config.model,
            "integration_profile": container_config.integration_profile,
            "config_path": str(Path(spec.container_config_path)),
        }
        return payload

    @staticmethod
    def _value_result_payload(dataset, run_result, evaluation) -> dict[str, Any]:
        return {
            "dataset_summary": dataset.summary(),
            "scenario_context": {
                scenario.id: {
                    "task_prompt": scenario.task_prompt,
                    "memory_bank": [
                        memory.model_dump(mode="json")
                        for memory in scenario.memory_bank
                    ],
                    "outcomes": [
                        outcome.model_dump(mode="json")
                        for outcome in scenario.effective_outcomes()
                    ],
                }
                for scenario in dataset.scenarios
            },
            "run": run_result,
            "evaluation": evaluation,
            "evaluation_aggregate": evaluation.aggregate(),
        }

    async def _execute_extraction(self, spec: RunSpec) -> dict[str, Any]:
        dataset = load_dataset(spec.dataset())
        config = spec.extraction_config()
        run_result = await run_benchmark(dataset, config)
        evaluation = await evaluate_run(dataset, run_result)
        return {
            "dataset_summary": dataset.summary(),
            "run": run_result,
            "evaluation": evaluation,
        }

    @staticmethod
    def _run_id(spec: RunSpec) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        model = spec.model.replace("/", "-").replace("@", "-")
        return f"{timestamp}_{spec.kind}_{spec.provider}_{model}_{spec.system}"

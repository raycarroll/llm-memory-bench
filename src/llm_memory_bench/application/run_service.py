from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .evaluation_service import EvaluationService
from .experiments import (
    DemoExperiment,
    ExtractionExperiment,
    Tau2Experiment,
    ValueExperiment,
)
from .specs import EvaluationSpec, RunSpec


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
        self.extraction = ExtractionExperiment()
        self.demo = DemoExperiment()
        self.value = ValueExperiment()
        self.tau2 = Tau2Experiment(self.results_dir)
        self.evaluation = EvaluationService(self.results_dir)

    async def execute(self, spec: RunSpec) -> dict[str, Any]:
        run_id = self._run_id(spec)
        if spec.kind == "extraction":
            output = await self.extraction.run(spec, run_id)
        elif spec.kind == "demo":
            output = await self.demo.run(spec, run_id)
        elif spec.kind == "value":
            output = await self.value.run(spec, run_id)
        elif spec.kind == "tau2":
            output = await self.tau2.run(spec, run_id)
        else:  # pragma: no cover - RunSpec validates the allowed values.
            raise ValueError(f"Unsupported experiment kind: {spec.kind}")

        result_json = _jsonable(output.result)
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
        if output.native_output_path is not None:
            payload["native_output_path"] = str(output.native_output_path)
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
            f"demo_config={spec.get('demo_config_path', '')}",
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

    async def evaluate(self, spec: EvaluationSpec) -> dict[str, Any]:
        """Compatibility entry point for saved-result evaluation."""
        return await self.evaluation.evaluate(spec)

    def evaluate_sync(self, spec: EvaluationSpec) -> dict[str, Any]:
        """Synchronous entry point for Streamlit evaluation callbacks."""
        return asyncio.run(self.evaluate(spec))

    @staticmethod
    def _run_result_from_json(run_data: dict[str, Any]) -> Any:
        """Compatibility alias for callers that used the old helper."""
        return EvaluationService._run_result_from_json(run_data)

    @staticmethod
    def _run_id(spec: RunSpec) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
        model = spec.model.replace("/", "-").replace("@", "-")
        return f"{timestamp}_{spec.kind}_{spec.provider}_{model}_{spec.system}"

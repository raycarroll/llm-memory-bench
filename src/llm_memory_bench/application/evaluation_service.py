"""Application coordinator for evaluating saved extraction results."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import RunConfig, get_provider
from ..dataset import load_dataset
from ..providers.base import ToolCall
from ..runner import ConversationResult, RunResult, TurnResult
from .evaluators import ExtractionEvaluator
from .specs import EvaluationSpec


class EvaluationService:
    """Load saved results and delegate evaluation to the right implementation."""

    def __init__(self, results_dir: Path | str = "results/ui"):
        self.results_dir = Path(results_dir)
        self.extraction = ExtractionEvaluator()

    async def evaluate(self, spec: EvaluationSpec) -> dict[str, Any]:
        evaluations = [
            await self._evaluate_result(Path(result_path), spec)
            for result_path in spec.result_paths
        ]

        created_at = datetime.now(timezone.utc).isoformat()
        evaluation_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
        output_dir = self.results_dir / "evaluations"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self._unique_output_path(
            output_dir / f"{evaluation_id}_{spec.matcher}.json"
        )
        payload = {
            "evaluation_id": evaluation_id,
            "created_at": created_at,
            "spec": spec.model_dump(mode="json"),
            "results": evaluations,
            "output_path": str(output_path),
        }
        output_path.write_text(json.dumps(payload, indent=2, default=str))
        return payload

    def evaluate_sync(self, spec: EvaluationSpec) -> dict[str, Any]:
        return asyncio.run(self.evaluate(spec))

    async def _evaluate_result(
        self,
        result_path: Path,
        spec: EvaluationSpec,
    ) -> dict[str, Any]:
        if not result_path.is_file():
            raise ValueError(f"Result file not found: {result_path}")

        source_data = json.loads(result_path.read_text(encoding="utf-8"))
        run_data = self._unwrap_saved_run(source_data)
        run_result = self._run_result_from_json(run_data)

        dataset_path = spec.dataset_path or run_data.get("dataset", {}).get("path")
        if not dataset_path:
            raise ValueError(
                f"No dataset path found for {result_path.name}; select one in the UI."
            )
        dataset = load_dataset(Path(dataset_path))
        matcher = self._build_matcher(spec)
        evaluation = await self.extraction.evaluate(
            dataset,
            run_result,
            matcher=matcher,
        )
        matcher_stats = {}
        for key in ("total_comparisons", "llm_escalations"):
            if hasattr(matcher, key):
                matcher_stats[key] = getattr(matcher, key)
        if hasattr(matcher, "escalation_rate"):
            matcher_stats["escalation_rate"] = matcher.escalation_rate()

        return {
            "source_run": str(result_path.resolve()),
            "dataset": dataset.summary(),
            "coverage": {
                "result_conversations": len(run_result.conversation_results),
                "dataset_conversations": len(dataset.conversations),
                "partial_dataset": len(run_result.conversation_results)
                < len(dataset.conversations),
            },
            "metrics": evaluation.metrics(dataset.ground_truth_type),
            "metrics_by_fact_type": evaluation.metrics_by_fact_type(),
            "matcher_stats": matcher_stats,
            "conversations": [
                conversation.to_dict()
                for conversation in evaluation.conversation_evaluations
            ],
        }

    @staticmethod
    def _build_matcher(spec: EvaluationSpec) -> Any:
        from ..matchers.alpsbench import AlpsBenchMatcher
        from ..matchers.embedding import EmbeddingMatcher
        from ..matchers.exact import ExactMatcher
        from ..matchers.hybrid import HybridMatcher
        from ..matchers.llm import LLMMatcher

        judge_config = RunConfig(
            provider=spec.judge_provider,
            model=spec.judge_model,
            api_key_env=spec.api_key_env,
        )
        if spec.matcher == "hybrid":
            return HybridMatcher(get_provider(judge_config))
        if spec.matcher == "llm":
            return LLMMatcher(get_provider(judge_config))
        if spec.matcher == "embedding":
            return EmbeddingMatcher()
        if spec.matcher == "exact":
            return ExactMatcher()
        return AlpsBenchMatcher()

    @staticmethod
    def _unwrap_saved_run(source_data: dict[str, Any]) -> dict[str, Any]:
        """Accept native result files and UI raw payloads."""
        if "config" in source_data and "conversations" in source_data:
            return source_data
        result = source_data.get("result", {})
        if isinstance(result, dict) and isinstance(result.get("run"), dict):
            return result["run"]
        raise ValueError(
            "The selected file is not a raw benchmark result with config and conversations."
        )

    @staticmethod
    def _run_result_from_json(run_data: dict[str, Any]) -> RunResult:
        run_result = RunResult(
            config=RunConfig(**run_data["config"]),
            total_time_ms=run_data.get("total_time_ms", 0),
        )
        for conversation in run_data.get("conversations", []):
            conversation_result = ConversationResult(
                conversation_id=conversation["conversation_id"]
            )
            for turn in conversation.get("turns", []):
                conversation_result.turn_results.append(
                    TurnResult(
                        turn_index=turn["turn_index"],
                        role=turn.get("role", "assistant"),
                        tool_calls=[
                            ToolCall(
                                name=tool_call["name"],
                                arguments=tool_call.get("arguments", {}),
                            )
                            for tool_call in turn.get("tool_calls", [])
                        ],
                        input_tokens=turn.get("input_tokens", 0),
                        output_tokens=turn.get("output_tokens", 0),
                        latency_ms=turn.get("latency_ms", 0),
                        response_text=turn.get("response_text", ""),
                    )
                )

            # Claude Code integrations often expose durable writes as the
            # result's ``stored`` artifact rather than assistant tool calls.
            system = str(run_data.get("config", {}).get("system", ""))
            storage_calls_present = False
            for turn in conversation_result.turn_results:
                for tool_call in turn.tool_calls:
                    if system.startswith("memoryhub") and tool_call.name in (
                        "memory",
                        "mcp__memoryhub__memory",
                    ) and tool_call.arguments.get("action") in ("write", "update"):
                        storage_calls_present = True
                    if system.startswith("hindsight") and tool_call.name in (
                        "retain",
                        "mcp__hindsight__retain",
                        "sync_retain",
                        "mcp__hindsight__sync_retain",
                        "hindsight_capture_initiative",
                        "hindsight_ingest_document",
                    ):
                        storage_calls_present = True

            if not storage_calls_present:
                stored_tool_calls = []
                for item in conversation.get("stored", []):
                    content = (
                        item.get("content")
                        or item.get("text")
                        or item.get("value")
                        or item.get("fact")
                    )
                    if not content:
                        continue
                    if system.startswith("memoryhub"):
                        stored_tool_calls.append(
                            ToolCall(
                                name="memory",
                                arguments={"action": "write", "content": content},
                            )
                        )
                    elif system.startswith("hindsight"):
                        stored_tool_calls.append(
                            ToolCall(
                                name="mcp__hindsight__retain",
                                arguments={"content": content},
                            )
                        )
                if stored_tool_calls:
                    if conversation_result.turn_results:
                        conversation_result.turn_results[-1].tool_calls.extend(
                            stored_tool_calls
                        )
                    else:
                        conversation_result.turn_results.append(
                            TurnResult(
                                turn_index=0,
                                role="assistant",
                                tool_calls=stored_tool_calls,
                            )
                        )
            run_result.conversation_results.append(conversation_result)
        return run_result

    @staticmethod
    def _unique_output_path(path: Path) -> Path:
        candidate = path
        suffix = 1
        while candidate.exists():
            candidate = path.with_name(f"{path.stem}_{suffix}{path.suffix}")
            suffix += 1
        return candidate

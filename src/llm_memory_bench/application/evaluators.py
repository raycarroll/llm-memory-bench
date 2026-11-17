"""Experiment-specific evaluation implementations.

The benchmark types do not share one evaluation algorithm.  These small
classes provide a common entry point without forcing the algorithms into one
generic evaluator.
"""

from __future__ import annotations

from typing import Any, Protocol

from ..config import RunConfig
from ..evaluator import evaluate_run
from ..value.evaluator import evaluate_value_run


class Evaluator(Protocol):
    """Common interface used by experiment implementations."""

    kind: str

    async def evaluate(
        self,
        dataset: Any,
        run_result: Any,
        **kwargs: Any,
    ) -> Any:
        """Evaluate one completed run for this experiment type."""


class ExtractionEvaluator:
    kind = "extraction"

    async def evaluate(
        self,
        dataset: Any,
        run_result: Any,
        *,
        matcher: Any | None = None,
        judge_config: RunConfig | None = None,
    ) -> Any:
        return await evaluate_run(
            dataset,
            run_result,
            matcher=matcher,
            judge_config=judge_config,
        )


class ValueEvaluator:
    kind = "value"

    async def evaluate(
        self,
        dataset: Any,
        run_result: Any,
        *,
        judge_config: RunConfig | None = None,
    ) -> Any:
        return await evaluate_value_run(dataset, run_result, judge_config)

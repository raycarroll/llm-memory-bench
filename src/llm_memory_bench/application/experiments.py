"""Concrete experiment implementations used by the application layer."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ..real_system_container_extraction_runner import RunConfig as BenchRunConfig
from ..real_system_container_extraction_runner import run_benchmark as run_container_benchmark
from ..config import RunConfig
from ..dataset import load_dataset
from ..simulated_tool_extraction_runner import run_benchmark
from ..value.real_system_container_value_runner import run_container_value_benchmark
from ..value.dataset import load_value_dataset
from ..value.tau2 import load_tau2_memory_plan, run_tau2_value_benchmark
from ..value.clbench import (
    default_clbench_python,
    run_clbench_value_benchmark,
)
from .evaluators import Evaluator, ExtractionEvaluator, ValueEvaluator
from .specs import RunSpec
from ..demo import load_demo_spec
from ..demo.driver import run_demo


@dataclass
class ExperimentOutput:
    """Result returned by an experiment, plus any native artifact it creates."""

    result: Any
    native_output_path: Path | None = None


class Experiment(Protocol):
    """Common entry point for a concrete benchmark type."""

    kind: str

    async def run(self, spec: RunSpec, run_id: str, on_event=None) -> ExperimentOutput:
        """Execute one experiment described by ``spec``."""


class DemoExperiment:
    kind = "demo"

    async def run(self, spec: RunSpec, run_id: str, on_event=None) -> ExperimentOutput:
        if not spec.demo_config_path:
            raise ValueError("demo_config_path is required for demo runs")
        demo = load_demo_spec(spec.demo_config_path)
        # The sidebar's demo limit must override the scenario default. Keep
        # the YAML as the source of all other limits and boundaries.
        demo = demo.model_copy(
            update={
                "limits": demo.limits.model_copy(update={"max_turns": spec.max_turns})
            }
        )
        result = await run_demo(
            demo,
            provider_name=spec.provider,
            model=spec.model,
            memory_system=spec.system,
            memory_mode=spec.mode,
            api_key_env=spec.api_key_env,
            on_event=on_event,
            prepare_memory=spec.demo_prepare_memory,
            prepare_max_turns=spec.demo_prepare_max_turns,
        )
        return ExperimentOutput({"demo": result})


class ExtractionExperiment:
    kind = "extraction"

    def __init__(self, evaluator: Evaluator | None = None):
        self.evaluator = evaluator or ExtractionEvaluator()

    async def run(self, spec: RunSpec, run_id: str) -> ExperimentOutput:
        if spec.execution == "container":
            return ExperimentOutput(await self._run_container(spec))

        dataset = load_dataset(spec.dataset())
        run_result = await run_benchmark(dataset, spec.extraction_config())
        return ExperimentOutput(
            {
                "dataset_summary": dataset.summary(),
                "run": run_result,
            }
        )

    async def _run_container(self, spec: RunSpec) -> dict[str, Any]:
        if not spec.container_config_path:
            raise ValueError(
                "container_config_path is required for container extraction runs"
            )

        dataset = load_dataset(spec.dataset())
        config = BenchRunConfig.from_yaml(Path(spec.container_config_path))
        # The UI owns dataset selection; the container config supplies the
        # agent, host, and memory integration surface.
        config.dataset_path = str(spec.dataset())
        config.max_conversations = spec.max_conversations
        run_result = await asyncio.to_thread(run_container_benchmark, config)
        return {
            "dataset_summary": dataset.summary(),
            "run": run_result,
            "container": {
                "host": {
                    "name": config.host_name,
                    "version": config.host_version,
                },
                "model": config.model,
                "integration_profile": config.integration_profile,
                "config_path": str(Path(spec.container_config_path)),
                "prompt_file": config.system_prompt_file,
                "prompt_mode": config.system_prompt_mode,
            },
            "container_summary": self._container_summary(run_result),
        }

    @staticmethod
    def _container_summary(run_result: Any) -> dict[str, Any]:
        conversations = run_result.conversation_results
        return {
            "conversations": len(conversations),
            "stored_items": sum(len(item.stored) for item in conversations),
            "expected_facts": sum(len(item.ground_truth) for item in conversations),
            "hook_storage_events": sum(
                len(item.hook_storage_events) for item in conversations
            ),
        }


class ValueExperiment:
    kind = "value"

    def __init__(self, evaluator: Evaluator | None = None):
        self.evaluator = evaluator or ValueEvaluator()

    async def run(self, spec: RunSpec, run_id: str) -> ExperimentOutput:
        if spec.value_source == "tau2" or spec.kind == "tau2":
            return await Tau2Experiment().run(spec, run_id)
        if spec.value_source == "clbench":
            if not spec.clbench_root:
                raise ValueError(
                    "clbench_root is required for CL-Bench value runs"
                )
            root = Path(spec.clbench_root).expanduser().resolve()
            interpreter = spec.clbench_python or default_clbench_python(root)
            native_output_path = self._clbench_output_path(spec, run_id)
            if spec.execution == "container":
                if not spec.container_config_path:
                    raise ValueError(
                        "container_config_path is required for CL-Bench container runs"
                    )
                from ..value.clbench_container import run_clbench_container_value_benchmark

                container_config = BenchRunConfig.from_yaml(
                    Path(spec.container_config_path)
                )
                request = {
                    "task": spec.clbench_task,
                    "schedule": spec.clbench_schedule,
                    "variant": spec.clbench_variant,
                    "max_instances": spec.clbench_max_instances,
                    "seed": spec.clbench_seed,
                    "memory_setup_mode": spec.memory_setup_mode,
                    "seed_memories_path": spec.clbench_seed_memories_path,
                    "memory_system": spec.clbench_memory_system,
                    "memory_prompt_version": spec.clbench_memory_prompt_version,
                }
                result = await asyncio.to_thread(
                    run_clbench_container_value_benchmark,
                    clbench_root=root,
                    request=request,
                    container_config=container_config,
                    output_path=native_output_path,
                )
                return ExperimentOutput(result, native_output_path)
            result = await asyncio.to_thread(
                run_clbench_value_benchmark,
                clbench_root=root,
                clbench_python=interpreter,
                task=spec.clbench_task,
                schedule=spec.clbench_schedule,
                variant=spec.clbench_variant,
                max_instances=spec.clbench_max_instances,
                seed=spec.clbench_seed,
                memory_setup_mode=spec.memory_setup_mode,
                memory_system=spec.clbench_memory_system,
                memory_prompt_version=spec.clbench_memory_prompt_version,
                seed_memories_path=(
                    Path(spec.clbench_seed_memories_path).expanduser()
                    if spec.clbench_seed_memories_path
                    else None
                ),
                model=spec.model,
                output_path=native_output_path,
            )
            return ExperimentOutput(
                {
                    "dataset_summary": {
                        "harness": "clbench",
                        "task": spec.clbench_task,
                        "schedule": spec.clbench_schedule,
                        "variant": spec.clbench_variant,
                    },
                    "run": result,
                    "evaluation": None,
                    "evaluation_aggregate": result.get("comparison", {}),
                },
                native_output_path,
            )

        dataset = load_value_dataset(spec.dataset())
        if spec.goal_id:
            selected = [
                scenario for scenario in dataset.scenarios if scenario.id == spec.goal_id
            ]
            if not selected:
                raise ValueError(f"Goal not found in dataset: {spec.goal_id}")
            dataset = dataset.model_copy(update={"scenarios": selected})

        if spec.execution == "container":
            result = await self._run_container(spec, dataset)
        else:
            config = spec.value_config()
            run_result = await self._run_api(dataset, config)
            evaluation = await self.evaluator.evaluate(dataset, run_result)
            result = self._result_payload(dataset, run_result, evaluation)
        return ExperimentOutput(result)

    @staticmethod
    async def _run_api(dataset: Any, config: Any) -> Any:
        from ..value.simulated_tool_value_runner import run_value_benchmark

        return await run_value_benchmark(dataset, config)

    @staticmethod
    def _clbench_output_path(spec: RunSpec, run_id: str) -> Path:
        del spec
        return Path("results/ui") / f"{run_id}.clbench.json"

    async def _run_container(self, spec: RunSpec, dataset: Any) -> dict[str, Any]:
        if not spec.container_config_path:
            raise ValueError("container_config_path is required for container value runs")

        container_config = BenchRunConfig.from_yaml(Path(spec.container_config_path))
        if not container_config.integration_profile:
            raise ValueError(
                "Container value runs require a config with integration.profile"
            )

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
        evaluation = await self.evaluator.evaluate(
            dataset,
            run_result,
            judge_config=judge_config,
        )
        payload = self._result_payload(dataset, run_result, evaluation)
        payload["container"] = {
            "host": {
                "name": container_config.host_name,
                "version": container_config.host_version,
            },
            "model": container_config.model,
            "integration_profile": container_config.integration_profile,
            "config_path": str(Path(spec.container_config_path)),
            "prompt_file": container_config.system_prompt_file,
            "prompt_mode": container_config.system_prompt_mode,
        }
        return payload

    @staticmethod
    def _result_payload(dataset: Any, run_result: Any, evaluation: Any) -> dict[str, Any]:
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


class Tau2Experiment:
    kind = "tau2"

    def __init__(self, results_dir: Path | str = "results/ui"):
        self.results_dir = Path(results_dir)

    async def run(self, spec: RunSpec, run_id: str) -> ExperimentOutput:
        if not spec.tau2_domain:
            raise ValueError("tau2_domain is required for τ² runs")
        if not spec.tau2_task_ids:
            raise ValueError("At least one τ² task ID is required")
        if not spec.tau2_memory_plan:
            raise ValueError("tau2_memory_plan is required for τ² runs")

        native_output_path = self.results_dir / f"{run_id}.tau2.json"
        if spec.execution == "container":
            from ..value.tau2_container import run_tau2_container

            native_output_path.parent.mkdir(parents=True, exist_ok=True)
            exit_code = await asyncio.to_thread(
                run_tau2_container,
                domain=spec.tau2_domain,
                task_ids=spec.tau2_task_ids,
                memory_plan=Path(spec.tau2_memory_plan),
                agent_model=spec.tau2_agent_model,
                user_model=spec.tau2_user_model,
                memory_system=spec.tau2_memory_system,
                memory_setup_mode=spec.memory_setup_mode,
                seed=spec.tau2_seed,
                max_steps=spec.tau2_max_steps,
                memoryhub_live=spec.tau2_memoryhub_live,
                memoryhub_cleanup=spec.tau2_memoryhub_cleanup,
                output=native_output_path,
            )
            if exit_code != 0:
                raise RuntimeError(f"τ² container exited with status {exit_code}")
        else:
            result = await asyncio.to_thread(
                run_tau2_value_benchmark,
                load_tau2_memory_plan(Path(spec.tau2_memory_plan)),
                task_ids=spec.tau2_task_ids,
                agent_model=spec.tau2_agent_model,
                user_model=spec.tau2_user_model,
                memory_system=spec.tau2_memory_system,
                memory_setup_mode=spec.memory_setup_mode,
                seed=spec.tau2_seed,
                max_steps=spec.tau2_max_steps,
                memoryhub_live=spec.tau2_memoryhub_live,
                memoryhub_cleanup=spec.tau2_memoryhub_cleanup,
            )
            native_output_path.parent.mkdir(parents=True, exist_ok=True)
            native_output_path.write_text(json.dumps(result, indent=2))
            return ExperimentOutput(result, native_output_path)

        if not native_output_path.exists():
            raise RuntimeError(
                f"τ² container completed without producing {native_output_path}"
            )
        return ExperimentOutput(
            json.loads(native_output_path.read_text()), native_output_path
        )

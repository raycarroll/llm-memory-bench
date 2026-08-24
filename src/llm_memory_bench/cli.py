from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from .config import RunConfig, ValueRunConfig, get_provider
from .dataset import load_dataset
from .evaluator import evaluate_run
from .matchers import list_matchers
from .matchers.base import FactMatcher
from .providers.base import ProviderError
from .runner import run_benchmark
from .systems import get_system, list_systems

# Per-million-token pricing. Input/output rates for common models.
_TOKEN_COSTS: dict[str, tuple[float, float]] = {
    # (input_per_M, output_per_M)
    "claude-sonnet-4-5@20250929": (3.0, 15.0),
    "claude-sonnet-4-5-20250929": (3.0, 15.0),
    "claude-sonnet-4@20250514": (3.0, 15.0),
    "claude-sonnet-4-20250514": (3.0, 15.0),
    "claude-opus-4@20250514": (15.0, 75.0),
    "claude-opus-4-20250514": (15.0, 75.0),
    "claude-haiku-3-5-20241022": (0.80, 4.0),
    "gpt-4o": (2.50, 10.0),
    "gpt-4o-mini": (0.15, 0.60),
}


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    rates = _TOKEN_COSTS.get(model)
    if rates is None:
        return None
    return (input_tokens * rates[0] + output_tokens * rates[1]) / 1_000_000


def _token_summary(run_result) -> dict:
    total_input = 0
    total_output = 0
    for cr in run_result.conversation_results:
        for tr in cr.turn_results:
            total_input += tr.input_tokens
            total_output += tr.output_tokens
    total = total_input + total_output
    cost = _estimate_cost(run_result.config.model, total_input, total_output)
    summary = {
        "input_tokens": total_input,
        "output_tokens": total_output,
        "total_tokens": total,
    }
    if cost is not None:
        summary["estimated_cost_usd"] = round(cost, 4)
    return summary


def _token_summary_from_data(data: dict) -> dict:
    """Compute token summary from a raw run JSON."""
    total_input = 0
    total_output = 0
    for conv in data.get("conversations", []):
        for turn in conv.get("turns", []):
            total_input += turn.get("input_tokens", 0)
            total_output += turn.get("output_tokens", 0)
    total = total_input + total_output
    model = data.get("config", {}).get("model", "")
    cost = _estimate_cost(model, total_input, total_output)
    summary = {
        "input_tokens": total_input,
        "output_tokens": total_output,
        "total_tokens": total,
    }
    if cost is not None:
        summary["estimated_cost_usd"] = round(cost, 4)
    return summary


def _build_matcher(name: str, config: RunConfig | None = None, api_key_env: str | None = None) -> FactMatcher:
    if name == "llm":
        from .matchers.llm import LLMMatcher
        if config is None:
            raise click.UsageError("LLM matcher requires --judge-provider and --judge-model (or use --matcher embedding)")
        judge_config = RunConfig(
            provider=config.judge_provider,
            model=config.judge_model,
            api_key_env=api_key_env,
        )
        return LLMMatcher(get_provider(judge_config))
    if name == "embedding":
        from .matchers.embedding import EmbeddingMatcher
        return EmbeddingMatcher()
    raise ValueError(f"Unknown matcher: {name}")


def _serialize_run(run_result, config, dataset_path, dataset) -> dict:
    """Serialize a RunResult to the raw run format."""
    memory_system = get_system(config.system)
    conv_map = {c.id: c for c in dataset.conversations}
    conversations = []
    for cr in run_result.conversation_results:
        conversation = conv_map.get(cr.conversation_id)
        turns = []
        for tr in cr.turn_results:
            turn_data: dict = {
                "turn_index": tr.turn_index,
                "role": tr.role,
            }
            if conversation and tr.turn_index < len(conversation.turns):
                gt = conversation.turns[tr.turn_index].ground_truth
                if gt.should_store:
                    turn_data["ground_truth"] = [
                        {"fact": f.fact, "type": f.type.value}
                        for f in gt.should_store
                    ]
            if tr.tool_calls:
                turn_data["tool_calls"] = [
                    {"name": tc.name, "arguments": tc.arguments}
                    for tc in tr.tool_calls
                ]
            if tr.input_tokens or tr.output_tokens:
                turn_data["input_tokens"] = tr.input_tokens
                turn_data["output_tokens"] = tr.output_tokens
            if tr.latency_ms:
                turn_data["latency_ms"] = round(tr.latency_ms, 1)
            turns.append(turn_data)
        conversations.append({
            "conversation_id": cr.conversation_id,
            "turns": turns,
        })

    tokens = _token_summary(run_result)
    return {
        "config": config.model_dump(),
        "memory_system": memory_system.version_info(),
        "dataset": {
            "path": str(Path(dataset_path).resolve()),
            "summary": dataset.summary(),
        },
        "usage": tokens,
        "conversations": conversations,
        "total_time_ms": run_result.total_time_ms,
    }


console = Console()


def _run_id(provider: str, model: str, system: str) -> str:
    model_short = model.replace("/", "-").replace("@", "-")
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{provider}_{model_short}_{system}_{ts}"


@click.group()
def cli():
    """Benchmark LLM memory extraction decisions via MCP tool interception."""


@cli.command()
@click.option("--source", type=click.Choice(["alpsbench"]), default="alpsbench")
@click.option("--output", type=click.Path(), default="datasets/converted")
@click.option("--max-conversations", type=int, default=None)
@click.option("--english-only/--all-languages", default=True)
def convert(source: str, output: str, max_conversations: int | None, english_only: bool):
    """Convert external datasets to benchmark format."""
    if source == "alpsbench":
        from .adapters.alpsbench import convert_alpsbench

        convert_alpsbench(
            output_dir=Path(output),
            max_conversations=max_conversations,
            english_only=english_only,
        )


@cli.command()
@click.option("--dataset", "dataset_path", type=click.Path(exists=True), required=True)
@click.option("--provider", default="anthropic")
@click.option("--model", default="claude-sonnet-4-20250514")
@click.option(
    "--system",
    "system_name",
    type=click.Choice(list_systems()),
    default="simple",
    help="Memory system to benchmark (defines prompt, tools, and extraction).",
)
@click.option("--max-conversations", type=int, default=None)
@click.option("--output", type=click.Path(), default=None)
@click.option("--api-key-env", default=None)
def run(
    dataset_path: str,
    provider: str,
    model: str,
    system_name: str,
    max_conversations: int | None,
    output: str | None,
    api_key_env: str | None,
):
    """Run the memory extraction benchmark. Writes raw run data (no evaluation)."""
    if not output:
        output = f"results/{_run_id(provider, model, system_name)}.json"

    config = RunConfig(
        provider=provider,
        model=model,
        system=system_name,
        max_conversations=max_conversations,
        api_key_env=api_key_env,
    )

    dataset = load_dataset(Path(dataset_path))
    console.print(f"Loaded dataset: {json.dumps(dataset.summary(), indent=2)}")

    console.print(
        f"\nRunning benchmark: provider={provider}, model={model}, system={system_name}"
    )
    try:
        run_result = asyncio.run(run_benchmark(dataset, config))
    except ProviderError as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        raise SystemExit(1)

    output_data = _serialize_run(run_result, config, dataset_path, dataset)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_data, indent=2))
    console.print(f"\nRun data written to {output_path}")

    tokens = output_data["usage"]
    _print_run_summary(output_data, tokens)


@cli.command()
@click.argument("run_file", type=click.Path(exists=True))
@click.option("--dataset", "dataset_path", type=click.Path(exists=True), default=None,
              help="Dataset path. Defaults to the path stored in the run file.")
@click.option(
    "--matcher",
    "matcher_name",
    type=click.Choice(list_matchers()),
    default="embedding",
    help="Fact matching strategy.",
)
@click.option("--judge-provider", default=None, help="Provider for LLM matcher.")
@click.option("--judge-model", default=None, help="Model for LLM matcher.")
@click.option("--output", type=click.Path(), default=None)
@click.option("--api-key-env", default=None)
def evaluate(
    run_file: str,
    dataset_path: str | None,
    matcher_name: str,
    judge_provider: str | None,
    judge_model: str | None,
    output: str | None,
    api_key_env: str | None,
):
    """Evaluate a run file against ground truth. Reads raw run data, writes evaluation results."""
    with open(run_file) as f:
        run_data = json.load(f)

    run_config = RunConfig(**run_data["config"])
    system_name = run_config.system

    from .dataset import Conversation, Dataset, ExpectedFact, FactType, GroundTruth, Turn
    from .runner import ConversationResult as CR, RunResult, TurnResult
    from .providers.base import ToolCall

    has_ground_truth = any(
        "ground_truth" in turn
        for conv in run_data["conversations"]
        for turn in conv.get("turns", [])
    )

    if has_ground_truth:
        dataset_convs = []
        for conv in run_data["conversations"]:
            turns = []
            for turn in conv.get("turns", []):
                gt_data = turn.get("ground_truth", [])
                gt = GroundTruth(should_store=[
                    ExpectedFact(fact=f["fact"], type=FactType(f.get("type", "direct")))
                    for f in gt_data
                ])
                turns.append(Turn(role=turn["role"], content="", ground_truth=gt))
            dataset_convs.append(Conversation(id=conv["conversation_id"], turns=turns))
        dataset = Dataset(conversations=dataset_convs)
    else:
        if not dataset_path:
            dataset_path = run_data.get("dataset", {}).get("path")
        if not dataset_path or not Path(dataset_path).exists():
            console.print(
                "[bold red]Error:[/bold red] Run file has no embedded ground truth and "
                "dataset path not found. Specify --dataset explicitly.",
            )
            raise SystemExit(1)
        dataset = load_dataset(Path(dataset_path))

    run_result = RunResult(config=run_config, total_time_ms=run_data.get("total_time_ms", 0))
    for conv in run_data["conversations"]:
        cr = CR(conversation_id=conv["conversation_id"])
        for turn in conv.get("turns", []):
            tool_calls = [
                ToolCall(name=tc["name"], arguments=tc["arguments"])
                for tc in turn.get("tool_calls", [])
            ]
            cr.turn_results.append(TurnResult(
                turn_index=turn["turn_index"],
                role=turn["role"],
                tool_calls=tool_calls,
                input_tokens=turn.get("input_tokens", 0),
                output_tokens=turn.get("output_tokens", 0),
                latency_ms=turn.get("latency_ms", 0),
            ))
        run_result.conversation_results.append(cr)

    judge_config = None
    if matcher_name == "llm":
        judge_config = RunConfig(
            provider=judge_provider or run_config.provider,
            model=judge_model or run_config.model,
            judge_provider=judge_provider or run_config.provider,
            judge_model=judge_model or run_config.model,
            api_key_env=api_key_env,
        )

    matcher = _build_matcher(matcher_name, judge_config, api_key_env)
    console.print(f"Evaluating {run_file} with matcher: {matcher_name}")

    try:
        eval_result = asyncio.run(evaluate_run(dataset, run_result, matcher=matcher))
    except ProviderError as e:
        console.print(f"\n[bold red]Evaluation error:[/bold red] {e}")
        raise SystemExit(1)

    metrics = eval_result.metrics()
    type_metrics = eval_result.metrics_by_fact_type()

    if not output:
        run_stem = Path(run_file).stem
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        output = f"results/eval_{run_stem}_{matcher_name}_{ts}.json"

    dataset_info: dict = {"summary": dataset.summary()}
    if dataset_path:
        dataset_info["path"] = str(Path(dataset_path).resolve())
    elif run_data.get("dataset", {}).get("path"):
        dataset_info["path"] = run_data["dataset"]["path"]

    eval_data = {
        "source_run": str(Path(run_file).resolve()),
        "matcher": matcher_name,
        "config": run_config.model_dump(),
        "memory_system": get_system(system_name).version_info(),
        "dataset": dataset_info,
        "metrics": metrics,
        "metrics_by_fact_type": type_metrics,
        "conversations": [
            ce.to_dict() for ce in eval_result.conversation_evaluations
        ],
    }

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(eval_data, indent=2))
    console.print(f"\nEvaluation written to {output_path}")

    tokens = _token_summary_from_data(run_data)
    _print_metrics_table(metrics, type_metrics, tokens)


@cli.command(name="list-systems")
def list_systems_cmd():
    """List available memory systems."""
    from .systems import SYSTEMS

    table = Table(title="Available Memory Systems")
    table.add_column("Name", style="bold")
    table.add_column("Description")

    for name, cls in SYSTEMS.items():
        instance = cls()
        table.add_row(name, instance.description)

    console.print(table)


@cli.command()
@click.argument("result_files", nargs=-1, type=click.Path(exists=True))
def compare(result_files: tuple[str, ...]):
    """Compare results from multiple benchmark runs."""
    if len(result_files) < 2:
        console.print("Need at least 2 result files to compare.", style="red")
        return

    results = []
    for f in result_files:
        with open(f) as fh:
            results.append(json.load(fh))

    table = Table(title="Benchmark Comparison")
    table.add_column("Metric", style="bold")

    for i, r in enumerate(results):
        label = f"{r['config']['provider']}/{r['config']['model']}\n({r['config'].get('system', 'simple')})"
        table.add_column(label, justify="right")

    metric_keys = [
        "extraction_precision",
        "extraction_recall",
        "extraction_f1",
        "noise_resistance_rate",
        "schema_validity_rate",
        "total_tool_calls",
    ]

    for key in metric_keys:
        row = [key]
        for r in results:
            val = r.get("metrics", {}).get(key, "N/A")
            if isinstance(val, float):
                row.append(f"{val:.4f}")
            else:
                row.append(str(val))
        table.add_row(*row)

    console.print(table)


@cli.command(name="value-generate")
@click.option("--source", type=click.Choice(["alpsbench"]), default="alpsbench")
@click.option("--output", type=click.Path(), default="scenarios")
@click.option("--max-scenarios", type=int, default=None)
@click.option("--provider", default="anthropic")
@click.option("--model", default="claude-sonnet-4-20250514")
@click.option("--english-only/--all-languages", default=True)
@click.option("--api-key-env", default=None)
def value_generate(
    source: str,
    output: str,
    max_scenarios: int | None,
    provider: str,
    model: str,
    english_only: bool,
    api_key_env: str | None,
):
    """Generate value benchmark scenarios from external datasets."""
    if source == "alpsbench":
        from .adapters.alpsbench_value import generate_value_scenarios

        gen_config = RunConfig(
            provider=provider,
            model=model,
            api_key_env=api_key_env,
        )

        try:
            asyncio.run(
                generate_value_scenarios(
                    output_dir=Path(output),
                    generator_config=gen_config,
                    max_scenarios=max_scenarios,
                    english_only=english_only,
                )
            )
        except ProviderError as e:
            console.print(f"\n[bold red]Error:[/bold red] {e}")
            raise SystemExit(1)


@cli.command(name="value-run")
@click.option("--scenarios", "scenarios_path", type=click.Path(exists=True), required=True)
@click.option("--provider", default="anthropic")
@click.option("--model", default="claude-sonnet-4-20250514")
@click.option(
    "--system",
    "system_name",
    type=click.Choice(list_systems()),
    default="simple",
)
@click.option("--mode", type=click.Choice(["prompt", "tool", "both"]), default="prompt")
@click.option("--max-turns", type=int, default=10)
@click.option("--max-scenarios", type=int, default=None)
@click.option("--user-sim-provider", default=None)
@click.option("--user-sim-model", default=None)
@click.option("--judge-provider", default=None)
@click.option("--judge-model", default=None)
@click.option("--output", type=click.Path(), default=None)
@click.option("--api-key-env", default=None)
def value_run(
    scenarios_path: str,
    provider: str,
    model: str,
    system_name: str,
    mode: str,
    max_turns: int,
    max_scenarios: int | None,
    user_sim_provider: str | None,
    user_sim_model: str | None,
    judge_provider: str | None,
    judge_model: str | None,
    output: str | None,
    api_key_env: str | None,
):
    """Run the memory value benchmark with paired trials."""
    if not output:
        output = f"results/value_{_run_id(provider, model, system_name)}.json"
    from .value.dataset import load_value_dataset
    from .value.evaluator import evaluate_value_run
    from .value.runner import run_value_benchmark

    config = ValueRunConfig(
        provider=provider,
        model=model,
        system=system_name,
        injection_mode=mode,
        max_turns=max_turns,
        max_scenarios=max_scenarios,
        user_simulator_provider=user_sim_provider or provider,
        user_simulator_model=user_sim_model or model,
        judge_provider=judge_provider or provider,
        judge_model=judge_model or model,
        api_key_env=api_key_env,
    )

    dataset = load_value_dataset(Path(scenarios_path))
    console.print(f"Loaded value dataset: {json.dumps(dataset.summary(), indent=2)}")

    console.print(
        f"\nRunning value benchmark: provider={provider}, model={model}, "
        f"system={system_name}, mode={mode}, max_turns={max_turns}"
    )
    try:
        run_result = asyncio.run(run_value_benchmark(dataset, config))
    except ProviderError as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        raise SystemExit(1)

    console.print(
        f"\nBenchmark complete ({run_result.total_time_ms:.0f}ms). "
        f"Evaluating with judge: {config.judge_provider}/{config.judge_model}"
    )
    judge_config = RunConfig(
        provider=config.judge_provider,
        model=config.judge_model,
        api_key_env=api_key_env,
    )
    try:
        eval_result = asyncio.run(evaluate_value_run(dataset, run_result, judge_config))
    except ProviderError as e:
        console.print(f"\n[bold red]Judge error:[/bold red] {e}")
        raise SystemExit(1)

    aggregated = eval_result.aggregate()

    per_scenario = []
    for se in eval_result.scenario_evaluations:
        entry = {"scenario_id": se.scenario_id}
        for mode_name, metrics in se.metrics_by_mode.items():
            entry[mode_name] = metrics
        per_scenario.append(entry)

    output_data = {
        "config": config.model_dump(),
        "dataset_summary": dataset.summary(),
        "aggregated_metrics": aggregated,
        "per_scenario": per_scenario,
        "total_time_ms": run_result.total_time_ms,
    }

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_data, indent=2))
    console.print(f"\nResults written to {output_path}")

    _print_value_metrics(aggregated)


def _print_value_metrics(aggregated: dict):
    for mode, metrics in aggregated.items():
        table = Table(title=f"Memory Value Results — {mode}")
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")

        table.add_row("Scenarios", str(metrics.get("scenario_count", 0)))
        table.add_section()

        for key in [
            "avg_token_efficiency_ratio",
            "avg_turn_compression",
            "avg_quality_delta",
            "avg_value_per_token",
        ]:
            val = metrics.get(key, 0)
            table.add_row(key, f"{val:.4f}")

        table.add_section()
        table.add_row(
            "total_baseline_tokens", str(metrics.get("total_baseline_tokens", 0))
        )
        table.add_row(
            "total_memory_tokens", str(metrics.get("total_memory_tokens", 0))
        )
        table.add_row(
            "total_memory_overhead", str(metrics.get("total_memory_overhead", 0))
        )
        table.add_row(
            "total_effective_savings",
            str(metrics.get("total_effective_savings", 0)),
        )

        table.add_section()
        table.add_row(
            "quality_improved",
            str(metrics.get("scenarios_quality_improved", 0)),
        )
        table.add_row(
            "quality_degraded",
            str(metrics.get("scenarios_quality_degraded", 0)),
        )
        table.add_row(
            "tokens_saved", str(metrics.get("scenarios_tokens_saved", 0))
        )

        console.print(table)


def _print_run_summary(run_data: dict, tokens: dict):
    total_calls = sum(
        len(tc)
        for conv in run_data["conversations"]
        for turn in conv["turns"]
        for tc in [turn.get("tool_calls", [])]
    )
    table = Table(title="Run Summary")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("conversations", str(len(run_data["conversations"])))
    table.add_row("total_tool_calls", str(total_calls))
    table.add_section()
    table.add_row("input_tokens", f"{tokens['input_tokens']:,}")
    table.add_row("output_tokens", f"{tokens['output_tokens']:,}")
    table.add_row("total_tokens", f"{tokens['total_tokens']:,}")
    if "estimated_cost_usd" in tokens:
        table.add_row("estimated_cost", f"${tokens['estimated_cost_usd']:.4f}")
    console.print(table)


def _print_metrics_table(metrics: dict, type_metrics: dict, tokens: dict | None = None):
    table = Table(title="Evaluation Results")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    for key in [
        "extraction_precision",
        "extraction_recall",
        "extraction_f1",
        "noise_resistance_rate",
        "schema_validity_rate",
    ]:
        val = metrics[key]
        table.add_row(key, f"{val:.4f}")

    table.add_section()
    table.add_row("true_positives", str(metrics["true_positives"]))
    table.add_row("false_negatives", str(metrics["false_negatives"]))
    table.add_row("false_positives", str(metrics["false_positives"]))
    table.add_row("noise_turns", str(metrics["noise_turns"]))
    table.add_row("noise_violations", str(metrics["noise_violations"]))
    table.add_row("total_tool_calls", str(metrics["total_tool_calls"]))

    if tokens:
        table.add_section()
        table.add_row("input_tokens", f"{tokens['input_tokens']:,}")
        table.add_row("output_tokens", f"{tokens['output_tokens']:,}")
        table.add_row("total_tokens", f"{tokens['total_tokens']:,}")
        if "estimated_cost_usd" in tokens:
            table.add_row("estimated_cost", f"${tokens['estimated_cost_usd']:.4f}")

    console.print(table)

    if type_metrics:
        type_table = Table(title="Recall by Fact Type")
        type_table.add_column("Fact Type", style="bold")
        type_table.add_column("Recall", justify="right")
        type_table.add_column("TP", justify="right")
        type_table.add_column("FN", justify="right")

        for t, m in sorted(type_metrics.items()):
            type_table.add_row(t, f"{m['recall']:.4f}", str(m["tp"]), str(m["fn"]))

        console.print(type_table)


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--output", type=click.Path(), default=None)
def bench(config_path: str, output: str | None):
    """Run a containerised benchmark from a config file.

    Builds the host image, installs the memory system, drives conversations
    through the real coding agent, and evaluates what got stored.
    """
    from .bench import RunConfig as BenchRunConfig, run_benchmark as run_bench

    config = BenchRunConfig.from_yaml(Path(config_path))

    if not output:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        output = f"results/{config.host_name}_{config.system_name}_{ts}.json"

    console.print(
        f"Benchmark: host={config.host_name}@{config.host_version}, "
        f"system={config.system_name}"
        f"{'@' + config.system_version if config.system_version else ''}, "
        f"model={config.model}"
    )

    result = run_bench(config)

    output_data = {
        "environment": result.environment,
        "dataset": {
            "path": config.dataset_path,
        },
        "conversations": [
            {
                "conversation_id": cr.conversation_id,
                "stored": cr.stored,
                "ground_truth": cr.ground_truth,
            }
            for cr in result.conversation_results
        ],
    }

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_data, indent=2))
    console.print(f"\nResults written to {output_path}")

    total_stored = sum(len(cr.stored) for cr in result.conversation_results)
    total_expected = sum(len(cr.ground_truth) for cr in result.conversation_results)
    console.print(
        f"  {len(result.conversation_results)} conversations, "
        f"{total_stored} items stored, {total_expected} facts expected"
    )

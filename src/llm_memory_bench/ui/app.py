from __future__ import annotations

import json
import html
import importlib.util
import os
import re
from pathlib import Path

from llm_memory_bench.real_system_container_extraction_runner import RunConfig as BenchRunConfig
from llm_memory_bench.application import EvaluationService, RunService
from llm_memory_bench.application.specs import EvaluationSpec, RunSpec
from llm_memory_bench.systems import list_prompt_versions, list_systems
from llm_memory_bench.value.dataset import load_value_dataset
from llm_memory_bench.value.clbench import default_clbench_python
from llm_memory_bench.value.tau2 import load_tau2_memory_plan


def memoryhub_runtime_status() -> dict[str, bool]:
    """Report live MemoryHub readiness without exposing credentials."""
    credential_names = (
        "MEMORYHUB_API_KEY",
        "MEMORYHUB_AUTH_TOKEN",
        "MEMORYHUB_CLIENT_ID",
        "MEMORYHUB_CLIENT_SECRET",
    )
    return {
        "endpoint": bool(os.environ.get("MEMORYHUB_URL")),
        "credentials": any(os.environ.get(name) for name in credential_names),
        "sdk": importlib.util.find_spec("memoryhub") is not None,
    }


def clbench_runtime_status(root: Path, python: str | None, task: str) -> dict[str, object]:
    """Report optional CL-Bench setup without attempting network operations."""
    resolved_root = root.expanduser().resolve()
    interpreter = python or default_clbench_python(resolved_root)
    task_dir = resolved_root / "data" / task
    return {
        "root": resolved_root,
        "python": interpreter,
        "checkout": (resolved_root / "src" / "interface.py").is_file(),
        "interpreter_ready": Path(interpreter).is_file(),
        "task_data": task_dir.is_dir(),
    }


def load_tau2_task_catalog(domain: str) -> dict[str, dict]:
    """Load readable official τ² task metadata for the run-definition UI."""
    candidates = [
        Path.cwd() / ".tau2-upstream" / "data" / "tau2" / "domains" / domain / "tasks.json",
        Path(__file__).resolve().parents[3]
        / ".tau2-upstream"
        / "data"
        / "tau2"
        / "domains"
        / domain
        / "tasks.json",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            tasks = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        return {str(task["id"]): task for task in tasks}
    return {}


def _tau2_task_instructions(task: dict) -> dict:
    return task.get("user_scenario", {}).get("instructions", {}) or {}


def tau2_task_label(task_id: str, catalog: dict[str, dict]) -> str:
    """Return a compact human-readable selector label."""
    task = catalog.get(str(task_id), {})
    instructions = _tau2_task_instructions(task)
    known_info = instructions.get("known_info") or ""
    identity = next(
        (line.strip() for line in known_info.splitlines() if line.strip()),
        "",
    )
    if identity.lower().startswith("you are "):
        identity = identity[8:].rstrip(".")
    reason = " ".join((instructions.get("reason_for_call") or "").split())
    summary = reason[:100] + ("…" if len(reason) > 100 else "")
    parts = [f"Task {task_id}"]
    if identity:
        parts.append(identity)
    if summary:
        parts.append(summary)
    return " · ".join(parts)


def render_tau2_task_details_content(st, task_id: str, catalog: dict[str, dict]) -> None:
    """Render the selected official τ² task inside the task picker dialog."""
    task = catalog.get(str(task_id))
    if not task:
        st.info("Official task details are unavailable for this task.")
        return
    instructions = _tau2_task_instructions(task)
    description = task.get("description") or {}
    st.caption(f"Official τ² task {task_id}")
    if description.get("purpose"):
        st.markdown(f"**Purpose**  \n{description['purpose']}")
    if instructions.get("reason_for_call"):
        st.markdown(f"**Reason for call**  \n{instructions['reason_for_call']}")
    if instructions.get("task_instructions"):
        st.markdown(f"**User instructions**  \n{instructions['task_instructions']}")
    detail_cols = st.columns(2)
    with detail_cols[0]:
        st.markdown("**Known information**")
        st.code(instructions.get("known_info") or "None", language="text")
    with detail_cols[1]:
        st.markdown("**Unknown information**")
        st.code(instructions.get("unknown_info") or "None", language="text")


def open_tau2_task_picker(
    st,
    *,
    task_options: list[str],
    task_catalog: dict[str, dict],
    selection_key: str,
) -> None:
    """Open the τ² task selector and details preview as a modal dialog."""
    st.session_state.setdefault(selection_key, [])
    dialog_key = f"{selection_key}::dialog"
    st.session_state.setdefault(
        dialog_key,
        list(st.session_state.get(selection_key, [])),
    )

    def sync_selection() -> None:
        """Copy modal widget state into state that survives modal closure."""
        st.session_state[selection_key] = list(st.session_state[dialog_key])

    def select_all() -> None:
        st.session_state[dialog_key] = list(task_options)
        st.session_state[selection_key] = list(task_options)

    def clear_selection() -> None:
        st.session_state[dialog_key] = []
        st.session_state[selection_key] = []

    @st.dialog("Select τ² tasks", width="large")
    def task_picker_dialog() -> None:
        st.multiselect(
            "Tasks to run",
            options=task_options,
            format_func=lambda task_id: tau2_task_label(task_id, task_catalog),
            key=dialog_key,
            on_change=sync_selection,
            help="Choose one or more official tasks. The memory plan must contain each selected task.",
        )

        action_cols = st.columns(3)
        action_cols[0].button(
            "Select all",
            use_container_width=True,
            on_click=select_all,
        )
        action_cols[1].button(
            "Clear",
            use_container_width=True,
            on_click=clear_selection,
        )
        if action_cols[2].button("Apply selection", type="primary", use_container_width=True):
            sync_selection()
            st.rerun()

        selected = st.session_state[dialog_key]
        if selected:
            preview_task_id = st.selectbox(
                "Task details",
                options=selected,
                format_func=lambda task_id: tau2_task_label(task_id, task_catalog),
            )
            st.divider()
            render_tau2_task_details_content(st, preview_task_id, task_catalog)
        else:
            st.info("Select at least one task to see its details.")

    task_picker_dialog()


def saved_evaluation_results() -> list[Path]:
    """Return raw benchmark result files that can be evaluated."""
    results_dir = Path("results")
    if not results_dir.is_dir():
        return []
    candidates = []
    for path in results_dir.rglob("*.json"):
        name = path.name
        if name.startswith("eval_") or "/evaluations/" in path.as_posix():
            continue
        if not (name.startswith("bench_") or name.endswith(".raw.json")):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        is_raw_run = "config" in data and "conversations" in data
        wrapped_run = data.get("result", {}).get("run", {})
        is_wrapped_run = (
            isinstance(wrapped_run, dict)
            and "config" in wrapped_run
            and "conversations" in wrapped_run
        )
        if is_raw_run or is_wrapped_run:
            candidates.append(path)
    return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)


def result_dataset_path(path: Path) -> str:
    """Read the embedded dataset path without failing the UI on old files."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if "dataset" in data:
            return str(data["dataset"].get("path", ""))
        spec = data.get("spec", {})
        if isinstance(spec, dict) and spec.get("dataset_path"):
            return str(spec["dataset_path"])
        result = data.get("result", {})
        run = result.get("run", {}) if isinstance(result, dict) else {}
        return str(run.get("dataset", {}).get("path", ""))
    except (OSError, json.JSONDecodeError, AttributeError):
        return ""


def evaluation_result_label(path: Path) -> str:
    """Make the saved-result selector distinguishable without exposing full paths."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        config = data.get("config", {})
        if not config and isinstance(data.get("result"), dict):
            config = data["result"].get("run", {}).get("config", {})
        system = config.get("system", "unknown")
        model = config.get("model", "unknown")
        return f"{path.name} · {system} · {model}"
    except (OSError, json.JSONDecodeError, AttributeError):
        return path.name


def render_evaluation_result(st, payload: dict) -> None:
    """Render matcher metrics and links for a saved-result evaluation."""
    output_path = Path(payload["output_path"]).resolve()
    st.markdown(f"[Evaluation JSON]({output_path.as_uri()})")
    st.caption(f"Saved under `{output_path.parent}`")
    for item in payload.get("results", []):
        source = Path(item.get("source_run", "")).name or "result"
        st.markdown(f"#### {source}")
        metrics = item.get("metrics", {})
        coverage = item.get("coverage", {})
        if coverage.get("partial_dataset"):
            st.warning(
                "This result covers "
                f"{coverage.get('result_conversations', 0)} of "
                f"{coverage.get('dataset_conversations', 0)} dataset conversations. "
                "Metrics use the complete selected ground truth; provide a task-scoped dataset "
                "for a task-level score."
            )
        columns = st.columns(7)
        columns[0].metric("Precision", f"{metrics.get('cumulative_extraction_precision', metrics.get('per_turn_extraction_precision', 0)):.1%}")
        columns[1].metric("Recall", f"{metrics.get('cumulative_extraction_recall', metrics.get('per_turn_extraction_recall', 0)):.1%}")
        columns[2].metric("F1", f"{metrics.get('cumulative_extraction_f1', metrics.get('per_turn_extraction_f1', 0)):.1%}")
        columns[3].metric("True positives", metrics.get("true_positives", 0))
        columns[4].metric("False positives", metrics.get("false_positives", 0))
        columns[5].metric("False negatives", metrics.get("false_negatives", 0))
        columns[6].metric("Tool calls", metrics.get("total_tool_calls", 0))
        stats = item.get("matcher_stats", {})
        if stats:
            st.caption(
                f"Hybrid comparisons: {stats.get('total_comparisons', 0)} · "
                f"LLM escalations: {stats.get('llm_escalations', 0)} "
                f"({stats.get('escalation_rate', 0):.1%})"
            )
        with st.expander("Evaluation details"):
            st.json({
                "dataset": item.get("dataset", {}),
                "metrics_by_fact_type": item.get("metrics_by_fact_type", {}),
                "conversations": item.get("conversations", []),
            })


def render_evaluation_panel(st) -> None:
    """Configure and run matcher-based evaluation for saved benchmark results."""
    with st.expander("Evaluation configuration", expanded=True):
        candidates = saved_evaluation_results()
        if not candidates:
            st.info("No raw benchmark result files found under `results/`.")
            return

        labels = {str(path): evaluation_result_label(path) for path in candidates}
        default_paths = [str(path) for path in candidates[:4]]
        selected_paths = st.multiselect(
            "Result sets",
            options=[str(path) for path in candidates],
            default=default_paths,
            format_func=lambda path: labels[path],
            help="Select one or more raw benchmark results to evaluate with the same configuration.",
        )
        first_dataset = result_dataset_path(Path(selected_paths[0])) if selected_paths else ""
        dataset_path = st.text_input(
            "Ground-truth dataset",
            value=first_dataset,
            help="The dataset containing the expected memories used by the matcher.",
        )
        config_columns = st.columns(3)
        with config_columns[0]:
            matcher = st.selectbox(
                "Matcher",
                ["hybrid", "embedding", "llm", "exact", "alpsbench"],
                index=0,
                help="Hybrid uses local embeddings first and sends only borderline pairs to the judge.",
            )
        with config_columns[1]:
            judge_provider = st.selectbox(
                "Judge provider",
                ["openai", "anthropic", "litellm", "vertex"],
                index=0,
                disabled=matcher not in {"hybrid", "llm"},
            )
        with config_columns[2]:
            judge_model = st.text_input(
                "Judge model",
                value="gpt-4o",
                disabled=matcher not in {"hybrid", "llm"},
            )
        api_key_env = st.text_input(
            "Judge API-key environment variable",
            value="OPENAI_API_KEY" if judge_provider == "openai" else "",
            disabled=matcher not in {"hybrid", "llm"},
        ) or None
        if matcher in {"hybrid", "llm"}:
            env_name = api_key_env or "OPENAI_API_KEY"
            if os.environ.get(env_name):
                st.success(f"{env_name} is available to Streamlit.")
            else:
                st.warning(
                    f"{env_name} is not available to Streamlit. Set it before starting the UI."
                )
        if st.button("▶ Evaluate selected results", type="primary", use_container_width=True):
            if not selected_paths:
                st.error("Select at least one result set.")
            elif not dataset_path:
                st.error("Select or enter a ground-truth dataset path.")
            else:
                evaluation_spec = EvaluationSpec(
                    result_paths=selected_paths,
                    dataset_path=dataset_path,
                    matcher=matcher,
                    judge_provider=judge_provider,
                    judge_model=judge_model,
                    api_key_env=api_key_env,
                )
                try:
                    with st.spinner("Evaluating selected results..."):
                        payload = EvaluationService().evaluate_sync(evaluation_spec)
                    st.session_state["last_evaluation"] = payload
                    st.success(f"Evaluation completed: {payload['evaluation_id']}")
                except Exception as exc:
                    st.error(str(exc))


def render_app() -> None:
    import streamlit as st

    st.set_page_config(page_title="LLM Memory Bench", page_icon="🧠", layout="wide")
    st.title("🧠 LLM Memory Bench")
    st.caption("Single-run workspace · comparison UI v0.2")
    with st.sidebar:
        sidebar_run_tab, sidebar_eval_tab = st.tabs(["RUN", "EVAL"])
        with sidebar_run_tab:
            st.header("Run definition")
            st.caption("Controls for the **Run benchmark** tab")
            family = st.radio(
                "Experiment",
                ["value", "extraction"],
                format_func=lambda item: {
                    "value": "Value",
                    "extraction": "Extraction",
                }[item],
            )
            kind = family
            value_source = None
            if family == "value":
                value_source = st.radio(
                    "Scenario source",
                    ["harness", "tau2", "clbench"],
                    format_func=lambda item: {
                        "harness": "Harness scenarios",
                        "tau2": "Official τ²-bench",
                        "clbench": "Continual Learning Bench",
                    }[item],
                    help=(
                        "All sources use the value experiment contract; τ² and CL-Bench "
                        "retain their native task evaluators."
                    ),
                )
                if value_source == "tau2":
                    execution_label = st.radio(
                        "Runtime",
                        ["τ² direct", "τ² pinned container"],
                        index=1,
                        help="The pinned container is the most reproducible τ² runtime.",
                    )
                    st.caption(
                        "Runs official τ² tasks with matched baseline and memory arms."
                    )
                elif value_source == "clbench":
                    execution_label = st.radio(
                        "Runtime",
                        ["Native CL-Bench", "Real-system container"],
                        help=(
                            "Native runs CL-Bench's provider directly. The container "
                            "runtime sends the same task lifecycle through the real "
                            "agent and MCP integration proxy."
                        ),
                    )
                    st.caption(
                        "Runs the native CL-Bench task lifecycle and evaluator with the "
                        "selected memory setup."
                    )
                else:
                    execution_label = st.radio(
                        "Runtime",
                        ["Simulated-tool", "Real-system container"],
                        help=(
                            "Real-system container runs a baseline trial and a separate "
                            "trial with the configured memory integration."
                        ),
                    )
            else:
                execution_label = st.radio(
                    "Runtime",
                    ["Simulated-tool", "Real-system container"],
                    help="Real-system extraction runs the real agent and memory integration.",
                )
            execution = "container" if execution_label in {
                "Real-system container",
                "Real-system container",
                "τ² pinned container",
                "CL-Bench container",
            } else "api"

            dataset_path = ""
            goal_id = None
            goal_dataset = None
            tau2_domain = None
            tau2_task_ids: list[str] = []
            tau2_memory_plan = None
            tau2_agent_model = "openai/gpt-4.1"
            tau2_user_model = "openai/gpt-4.1"
            tau2_memory_system = "simple"
            tau2_seed = 300
            tau2_max_steps = 20
            tau2_memoryhub_live = False
            tau2_memoryhub_cleanup = True
            clbench_root = None
            clbench_python = None
            clbench_task = "database_exploration"
            clbench_schedule = "default"
            clbench_variant = None
            clbench_seed = 42
            clbench_max_instances = None
            clbench_memory_system = "simple"
            clbench_memory_prompt_version = None
            clbench_seed_memories_path = None
            provider = "openai"
            model = "gpt-4o"
            simulated_tool_system = "simple"
            mode = "prompt"
            memory_setup_mode = "none"
            user_simulator_provider = None
            user_simulator_model = None
            judge_provider = None
            judge_model = None

            with st.expander("Scenario", expanded=True):
                value_datasets = {
                    "Quick demo": "scenarios/quick-demo-goal.yaml",
                    "τ²-derived curated airline + retail": "scenarios/tau2-derived/airline-retail.yaml",
                }
                dataset_preset = None
                if value_source == "clbench":
                    clbench_root = st.text_input(
                        "CL-Bench checkout",
                        value=".cl-bench",
                        help="Root of the official continual-learning-bench checkout.",
                    )
                    clbench_python = st.text_input(
                        "CL-Bench Python (optional)",
                        value="",
                        help="Use the checkout's .venv/bin/python when blank.",
                    ) or None
                    clbench_task = st.selectbox(
                        "CL-Bench task",
                        ["database_exploration"],
                        help="The first adapter slice targets database exploration.",
                    )
                    clbench_schedule = st.text_input("CL-Bench schedule", value="default") or None
                    clbench_variant = st.text_input(
                        "CL-Bench variant (optional)", value=""
                    ) or None
                    clbench_max_instances = st.number_input(
                        "Max task instances (0 = task default)", min_value=0, value=0
                    )
                    dataset_path = clbench_root
                    clbench_status = clbench_runtime_status(
                        Path(clbench_root), clbench_python, clbench_task
                    )
                    if all(
                        clbench_status[key]
                        for key in ("checkout", "interpreter_ready", "task_data")
                    ):
                        st.success(
                            "CL-Bench is ready. Setup is complete and will be reused "
                            "for this and later runs."
                        )
                        st.caption(
                            f"Runtime: `{clbench_status['python']}` · "
                            f"task data: `{clbench_status['root'] / 'data' / clbench_task}`"
                        )
                    else:
                        missing = [
                            label
                            for key, label in (
                                ("checkout", "checkout"),
                                ("interpreter_ready", "Python runtime"),
                                ("task_data", "task data"),
                            )
                            if not clbench_status[key]
                        ]
                        st.warning(
                            "Optional CL-Bench is not ready: missing "
                            + ", ".join(missing)
                            + ". This run cannot start until setup is complete."
                        )
                        st.code("llm-memory-bench clbench-setup", language="bash")
                        st.caption(
                            "This is a one-time setup step; it downloads the external "
                            "runtime and task artifacts. Normal installs do not include it."
                        )
                elif kind == "value" and value_source == "harness":
                    dataset_preset = st.selectbox("Dataset example", list(value_datasets))
                    dataset_path = st.text_input(
                        "Dataset path",
                        value=(
                            value_datasets[dataset_preset]
                            if dataset_preset
                            else "datasets/test/alpsbench-task1.yaml"
                        ),
                    )
                    try:
                        goal_dataset = load_value_dataset(Path(dataset_path))
                        goal_id = st.selectbox(
                            "Goal",
                            options=[scenario.id for scenario in goal_dataset.scenarios],
                            format_func=lambda scenario_id: next(
                                scenario.task_prompt
                                for scenario in goal_dataset.scenarios
                                if scenario.id == scenario_id
                            ),
                        )
                    except Exception as exc:
                        st.warning(f"Goals unavailable: {exc}")
                elif value_source == "tau2":
                    tau2_domain = st.selectbox("τ² domain", ["airline", "retail", "telecom"])
                    plan_paths = sorted(
                        str(path)
                        for path in Path("scenarios").rglob("*-memory-plan.yaml")
                    )
                    default_plan = "scenarios/tau2-derived/airline-memory-plan.yaml"
                    if default_plan not in plan_paths:
                        plan_paths.insert(0, default_plan)
                    tau2_memory_plan = st.selectbox("Memory plan", plan_paths)
                    dataset_path = tau2_memory_plan
                    try:
                        plan = load_tau2_memory_plan(Path(tau2_memory_plan))
                        task_options = [task.task_id for task in plan.tasks]
                        st.caption(f"{len(plan.tasks)} task definitions loaded from the memory plan.")
                        task_catalog = load_tau2_task_catalog(tau2_domain)
                        if not task_catalog:
                            st.warning(
                                "Official τ² task details are unavailable; showing task IDs only."
                            )
                        selection_key = f"tau2_tasks::{tau2_domain}::{tau2_memory_plan}"
                        st.session_state.setdefault(selection_key, [])
                        tau2_task_ids = [
                            task_id
                            for task_id in st.session_state[selection_key]
                            if task_id in task_options
                        ]
                        if st.button(
                            f"Choose τ² tasks · {len(tau2_task_ids)} selected",
                            use_container_width=True,
                        ):
                            open_tau2_task_picker(
                                st,
                                task_options=task_options,
                                task_catalog=task_catalog,
                                selection_key=selection_key,
                            )
                        if tau2_task_ids:
                            st.caption(tau2_task_label(tau2_task_ids[0], task_catalog))
                        else:
                            st.warning("No τ² tasks selected.")
                    except Exception as exc:
                        task_options = []
                        st.warning(f"Task definitions unavailable: {exc}")
                elif kind == "extraction":
                    extraction_paths = sorted(
                        str(path)
                        for path in Path("datasets").rglob("*.yaml")
                    )
                    default_dataset = "datasets/test/alpsbench-task1.yaml"
                    if default_dataset not in extraction_paths and extraction_paths:
                        default_dataset = extraction_paths[0]
                    extraction_options = ["Custom path"] + extraction_paths
                    selected_dataset = st.selectbox(
                        "Extraction dataset",
                        extraction_options,
                        index=(
                            extraction_options.index(default_dataset)
                            if default_dataset in extraction_options
                            else 0
                        ),
                        help="Choose a checked-in dataset or enter a custom YAML path.",
                    )
                    if selected_dataset == "Custom path":
                        dataset_path = st.text_input(
                            "Dataset path",
                            value="",
                        )
                    else:
                        dataset_path = selected_dataset
            container_config_path = None
            container_preview = None
            if (
                execution == "container"
                and kind in {"value", "extraction"}
                and value_source != "tau2"
            ):
                with st.expander("Container", expanded=False):
                    config_paths = sorted(str(path) for path in Path("configs").glob("*.yaml"))
                    default_config = "configs/claude-memoryhub-remote.yaml"
                    if default_config not in config_paths and config_paths:
                        default_config = config_paths[0]
                    if config_paths:
                        container_config_path = st.selectbox(
                            "Container config",
                            config_paths,
                            index=config_paths.index(default_config),
                        )
                    else:
                        container_config_path = st.text_input(
                            "Container config path",
                            value=default_config,
                        )
                    try:
                        container_preview = BenchRunConfig.from_yaml(Path(container_config_path))
                        st.info(
                            f"Claude {container_preview.host_version} · "
                            f"{container_preview.model} · "
                            f"{container_preview.system_name}"
                        )
                        if container_preview.system_prompt_file:
                            st.caption(
                                "Memory prompt: "
                                f"{Path(container_preview.system_prompt_file).name} · "
                                f"{container_preview.system_prompt_mode}"
                            )
                        else:
                            st.caption("Memory prompt: none appended")
                    except Exception as exc:
                        container_preview = None
                        st.warning(f"Container config unavailable: {exc}")
                    provider = "anthropic"
                    model = container_preview.model if container_preview else "claude-container"
                    simulated_tool_system = "simple"
                    mode = "integration"
                    if kind == "value":
                        user_simulator_provider = st.selectbox(
                            "User simulator provider",
                            ["openai", "anthropic", "litellm", "vertex"],
                        )
                        user_simulator_model = st.text_input("User simulator model", value="gpt-4o")
                        judge_provider = st.selectbox(
                            "Outcome judge provider",
                            ["openai", "anthropic", "litellm", "vertex"],
                        )
                        judge_model = st.text_input("Outcome judge model", value="gpt-4o")
                    else:
                        st.caption("Container extraction uses the agent and memory system from this config.")
            if value_source == "clbench":
                with st.expander("CL-Bench model & memory", expanded=True):
                    if execution == "container" and container_preview is not None:
                        model = container_preview.model
                        st.caption(f"Container agent model: `{model}`")
                    else:
                        model = st.text_input("Agent model", value="openai/gpt-4.1")
                    memory_system_options = list_systems()
                    memory_system_default = (
                        "memoryhub" if execution == "container" and "memoryhub" in memory_system_options else "simple"
                    )
                    clbench_memory_system = st.selectbox(
                        "Memory integration", memory_system_options,
                        index=memory_system_options.index(memory_system_default),
                    )
                    prompt_versions = list_prompt_versions(clbench_memory_system)
                    if prompt_versions:
                        clbench_memory_prompt_version = st.selectbox(
                            "Memory prompt version",
                            prompt_versions,
                            index=0,
                            format_func=lambda version: (
                                f"{version} (default)" if version == prompt_versions[0] else version
                            ),
                            help=(
                                "Choices are loaded from the selected memory system's "
                                "versioned prompt files. The default is defined in that "
                                "system's metadata.yaml."
                            ),
                        )
                    else:
                        clbench_memory_prompt_version = None
                        st.caption(
                            "The selected memory system has no versioned prompt; "
                            "its built-in behavior will be used."
                        )
                    memory_setup_mode = st.selectbox(
                        "Memory setup",
                        ["none", "seeded", "capture_first"],
                        index=0,
                        format_func=lambda value: {
                            "none": "None — no memory control",
                            "seeded": "Seed declared memories",
                            "capture_first": "Capture baseline, then replay",
                        }[value],
                        help=(
                            "None is the causal control. Capture-first extracts durable "
                            "facts during the first CL-Bench run and replays them in a "
                            "matched second run."
                        ),
                    )
                    if memory_setup_mode == "seeded":
                        clbench_seed_memories_path = st.text_input(
                            "Seed memories file",
                            value="",
                            help="YAML/JSON list of {fact, category} entries.",
                        ) or None
                    clbench_seed = st.number_input("CL-Bench seed", min_value=0, value=42)
                    st.info(
                        "The CL-Bench task reward remains authoritative; memory is "
                        "measured by the paired native run score and gain."
                    )
            elif kind == "value" and value_source == "harness":
                with st.expander("Model & memory", expanded=False):
                    provider = st.selectbox("Provider", ["openai", "anthropic", "litellm", "vertex"])
                    model = st.text_input("Model", value="gpt-4o")
                    simulated_tool_system = st.selectbox("Memory integration", list_systems())
                    mode = st.selectbox("Memory delivery", ["prompt", "tool", "both"])
                    memory_setup_mode = st.selectbox(
                        "Memory setup",
                        ["none", "seeded", "capture_first"],
                        index=0,
                        format_func=lambda value: {
                            "none": "None — no memory control",
                            "seeded": "Seed declared memories",
                            "capture_first": "Capture baseline, then replay",
                        }[value],
                        help=(
                            "Seeded uses the scenario memory bank. Capture-first "
                            "hands baseline writes to the memory arm."
                        ),
                    )
            elif value_source == "tau2":
                with st.expander("Models & memory", expanded=False):
                    memory_setup_mode = st.selectbox(
                        "Memory setup",
                        ["none", "capture_first", "seeded"],
                        index=0,
                        format_func=lambda value: {
                            "none": "None — no memory control",
                            "seeded": "Seed declared memories",
                            "capture_first": "Capture baseline, then replay",
                        }[value],
                        help=(
                            "None is the causal control. Capture-first starts empty, "
                            "captures baseline writes, then replays the captured snapshot."
                        ),
                    )
                    tau2_agent_model = st.text_input("τ² agent model", value="openai/gpt-4.1")
                    tau2_user_model = st.text_input("τ² user simulator model", value="openai/gpt-4.1")
                    tau2_memory_system = st.selectbox(
                        "Memory backend",
                        ["simple", "memoryhub"],
                        format_func=lambda value: {
                            "simple": "Local simulated memory",
                            "memoryhub": "MemoryHub",
                        }[value],
                        key="tau2_memory_system",
                    )
                    if tau2_memory_system != "memoryhub":
                        st.session_state["tau2_memoryhub_live"] = False
                    tau2_memoryhub_live = st.checkbox(
                        "Connect to live MemoryHub",
                        value=False,
                        key="tau2_memoryhub_live",
                        disabled=tau2_memory_system != "memoryhub",
                        help=(
                            "Connect through MemoryHubClient using MEMORYHUB_URL and "
                            "the configured MEMORYHUB_* credentials."
                        ),
                    )
                    tau2_memoryhub_cleanup = st.checkbox(
                        "Clean up MemoryHub projects after the run",
                        value=True,
                        disabled=not tau2_memoryhub_live,
                        key="tau2_memoryhub_cleanup",
                    )
                    if tau2_memoryhub_live:
                        status = memoryhub_runtime_status()
                        status_cols = st.columns(3)
                        status_cols[0].metric(
                            "Endpoint", "Ready" if status["endpoint"] else "Missing"
                        )
                        status_cols[1].metric(
                            "Credentials", "Ready" if status["credentials"] else "Missing"
                        )
                        status_cols[2].metric(
                            "SDK", "Installed" if status["sdk"] else "Missing"
                        )
                        if all(status.values()):
                            st.success(
                                "Ready. Each task will use isolated baseline and replay "
                                "MemoryHub project scopes."
                            )
                        else:
                            st.warning(
                                "Live mode is selected but not ready. Configure "
                                "MEMORYHUB_URL, credentials, and the memoryhub package "
                                "in the environment running Streamlit."
                            )
                    tau2_seed = st.number_input("τ² seed", min_value=0, value=300)
                    tau2_max_steps = st.number_input("τ² max steps", min_value=1, value=20)
                    st.info(
                        "Set OPENAI_API_KEY (or the provider key required by the selected model) "
                        "in the environment running Streamlit."
                    )
                provider = "openai"
                model = tau2_agent_model
                simulated_tool_system = tau2_memory_system
                mode = "prompt"
                user_simulator_model = tau2_user_model
            else:
                with st.expander("Model & memory", expanded=False):
                    if execution == "api":
                        provider = st.selectbox("Provider", ["openai", "anthropic", "litellm", "vertex"])
                        model = st.text_input("Model", value="gpt-4o")
                        simulated_tool_system = st.selectbox("Memory integration", list_systems())
            with st.expander("Limits & evaluation", expanded=False):
                max_turns = st.number_input(
                    "Max turns", min_value=1, value=10, disabled=kind != "value"
                )
                max_scenarios = st.number_input(
                    "Max scenarios (0 = all)",
                    min_value=0,
                    value=1,
                    disabled=kind != "value" or value_source in {"tau2", "clbench"},
                )
                max_conversations = st.number_input(
                    "Max conversations (0 = all)", min_value=0, value=1, disabled=kind != "extraction"
                )
                judge = st.checkbox(
                    "Run outcome judge",
                    value=True,
                    disabled=kind != "value" or value_source in {"tau2", "clbench"},
                )
            submit = st.button("▶ Run benchmark", type="primary", use_container_width=True)

        with sidebar_eval_tab:
            st.caption("Evaluate saved runs independently from benchmark execution.")
            render_evaluation_panel(st)

    spec = RunSpec(
        kind=kind,
        execution=execution,
        dataset_path=dataset_path,
        container_config_path=container_config_path,
        goal_id=goal_id,
        provider=provider,
        model=model,
        system=simulated_tool_system,
        mode=mode,
        memory_setup_mode=memory_setup_mode,
        max_turns=int(max_turns),
        max_scenarios=int(max_scenarios) or None,
        max_conversations=int(max_conversations) or None,
        judge_during_run=judge,
        judge_provider=judge_provider,
        judge_model=judge_model,
        user_simulator_provider=user_simulator_provider,
        user_simulator_model=user_simulator_model,
        tau2_domain=tau2_domain,
        tau2_task_ids=tau2_task_ids,
        tau2_memory_plan=tau2_memory_plan,
        tau2_agent_model=tau2_agent_model,
        tau2_user_model=tau2_user_model,
        tau2_memory_system=tau2_memory_system,
        tau2_seed=int(tau2_seed),
        tau2_max_steps=int(tau2_max_steps),
        tau2_memoryhub_live=tau2_memoryhub_live,
        tau2_memoryhub_cleanup=tau2_memoryhub_cleanup,
        value_source=(
            "dataset" if value_source in {None, "harness"} else value_source
        ),
        clbench_root=clbench_root,
        clbench_python=clbench_python,
        clbench_task=clbench_task,
        clbench_schedule=clbench_schedule,
        clbench_variant=clbench_variant,
        clbench_seed=int(clbench_seed),
        clbench_max_instances=(
            int(clbench_max_instances) if clbench_max_instances else None
        ),
        clbench_memory_system=clbench_memory_system,
        clbench_memory_prompt_version=clbench_memory_prompt_version,
        clbench_seed_memories_path=clbench_seed_memories_path,
    )

    evaluation_payload = st.session_state.get("last_evaluation")
    if evaluation_payload:
        st.subheader("Latest evaluation")
        st.caption(
            "Evaluation results are shown here; use the EVAL pane to change the "
            "evaluation configuration or start another evaluation."
        )
        st.json(evaluation_payload.get("spec", {}))
        render_evaluation_result(st, evaluation_payload)
        st.divider()

    with st.expander("Run configuration", expanded=False):
        st.json(spec.model_dump(mode="json"))
        if goal_dataset is not None and goal_id:
            st.markdown("#### Goal definition")
            goal = next(scenario for scenario in goal_dataset.scenarios if scenario.id == goal_id)
            st.markdown(f"**Task**  \n{goal.task_prompt}")
            st.markdown("**Success criteria**")
            for outcome in goal.outcomes:
                st.markdown(f"- `{outcome.id}` — {outcome.statement}")
            st.markdown("**Seeded memory**")
            for memory in goal.memory_bank:
                st.markdown(f"- `{memory.category}` — {memory.fact}")

    st.subheader("Execution")
    if submit:
        try:
            with st.spinner("Running benchmark..."):
                payload = RunService().execute_sync(spec)
            st.session_state["last_run"] = payload
            st.success(f"Completed: {payload['run_id']}")
        except Exception as exc:
            st.error(str(exc))

    payload = st.session_state.get("last_run")
    if payload is not None:
        st.divider()
        st.subheader("Latest result · comparison view")
        output_path = Path(payload["output_path"]).resolve()
        artifact_links = [f"[Result JSON]({output_path.as_uri()})"]
        for key, label in (
            ("raw_output_path", "Raw run JSON"),
            ("native_output_path", "Native τ² JSON"),
            ("log_path", "Execution log"),
        ):
            # Runs kept in Streamlit session state may have been created before
            # these artifact fields existed. Do not break the comparison view
            # for those older payloads.
            artifact_path = payload.get(key)
            if artifact_path:
                artifact_links.append(f"[{label}]({Path(artifact_path).resolve().as_uri()})")
        st.markdown(" · ".join(artifact_links))
        st.caption(f"Artifacts saved under `{output_path.parent}`")
        if not payload.get("raw_output_path") or not payload.get("log_path"):
            st.info("This result predates raw-artifact links. Run it again to create them.")
        result = payload["result"]
        container = result.get("container")
        if container:
            host = container.get("host", {})
            st.info(
                f"Claude Code {host.get('version', '')} · "
                f"{container.get('model', 'unknown')} · "
                f"{container.get('integration_profile', 'unknown')}"
            )
            if container.get("prompt_file"):
                st.caption(
                    "Memory prompt: "
                    f"{Path(container['prompt_file']).name} · "
                    f"{container.get('prompt_mode', 'append')}"
                )
        run_kind = payload.get("spec", {}).get("kind", spec.kind)
        run_source = payload.get("spec", {}).get("value_source", "dataset")
        if run_kind == "tau2" or run_source == "tau2":
            render_tau2_result(st, result)
        elif run_source == "clbench":
            render_clbench_result(st, result)
        elif run_kind == "value":
            render_value_result(st, result, payload.get("spec", {}))
        else:
            render_extraction_result(st, result)
    else:
        st.info("Run a benchmark to display the baseline-versus-memory comparison here.")


def render_tau2_result(st, result: dict) -> None:
    """Render paired τ² arms with prompts and chronological native messages."""
    benchmark = result.get("benchmark", {})
    config = result.get("config", {})
    scenarios = result.get("scenarios", [])
    st.caption(
        f"{benchmark.get('domain', 'unknown')} · "
        f"tasks: {', '.join(benchmark.get('task_ids', []))} · "
        f"τ² revision: {benchmark.get('source_revision', 'unknown')}"
    )
    if not scenarios:
        st.info("No τ² scenarios were produced.")
        return

    summary_cols = st.columns(4)
    summary_cols[0].metric("Agent model", config.get("agent_model", "unknown"))
    summary_cols[1].metric("User model", config.get("user_model", "unknown"))
    summary_cols[2].metric("Memory system", config.get("memory_system", "unknown"))
    summary_cols[3].metric("Tasks", len(scenarios))
    backend = "live MemoryHub" if config.get("memoryhub_live") else "harness adapter"
    prompt_version = (config.get("memory_system_info") or {}).get("prompt_version")
    st.caption(
        f"Memory backend: {backend} · prompt/schema: {prompt_version or 'n/a'}"
    )

    selected_task = st.selectbox(
        "τ² task",
        [str(item.get("task_id", "unknown")) for item in scenarios],
    )
    scenario = next(
        item for item in scenarios if str(item.get("task_id")) == selected_task
    )
    fixture_memories = scenario.get("memory_bank", [])
    st.markdown("#### Fixture memory (not preloaded)")
    if fixture_memories:
        st.caption("Retained as task provenance; neither arm starts with these facts.")
        for memory in fixture_memories:
            st.markdown(
                f"- `{memory.get('category', 'general')}` — {memory.get('fact', '')}"
            )
    else:
        st.caption("No fixture facts configured.")

    baseline_captured = scenario.get("baseline", {}).get("captured_memories", [])
    replay_initial = scenario.get("memory_prompt", {}).get("initial_memories", [])
    st.markdown("#### Memory handoff")
    handoff_cols = st.columns(2)
    handoff_cols[0].metric("Captured by baseline", len(baseline_captured))
    handoff_cols[1].metric("Available to replay", len(replay_initial))
    projects = scenario.get("memoryhub_projects")
    if projects:
        with st.expander("MemoryHub project scopes", expanded=False):
            st.caption(
                "These scopes were created for this task; they may already be "
                "deleted when cleanup was enabled."
            )
            scope_cols = st.columns(2)
            scope_cols[0].code(
                f"Baseline\n{projects.get('baseline', 'unknown')}",
                language="text",
            )
            scope_cols[1].code(
                f"Replay\n{projects.get('replay', 'unknown')}",
                language="text",
            )
            st.caption(
                "Cleanup: "
                + ("enabled" if projects.get("cleaned_up") else "disabled")
            )
    if replay_initial:
        st.caption("The replay arm starts with the baseline snapshot:")
        st.markdown(
            " · ".join(f"`{item.get('fact', '')}`" for item in replay_initial)
        )

    baseline, memory = st.columns(2)
    with baseline:
        render_tau2_arm(st, "Baseline", scenario.get("baseline", {}), "⚪")
    with memory:
        render_tau2_arm(st, "Memory replay", scenario.get("memory_prompt", {}), "🧠")


def render_tau2_arm(st, title: str, arm: dict, avatar: str) -> None:
    reward = arm.get("reward")
    telemetry = arm.get("telemetry", {})
    simulation = arm.get("simulation", {})
    total_cost = (simulation.get("agent_cost") or 0) + (simulation.get("user_cost") or 0)
    latency_ms = float(telemetry.get("latency_ms") or 0)
    token_usage_exact = bool(telemetry.get("token_usage_exact"))
    input_tokens = int(
        telemetry.get("input_tokens" if token_usage_exact else "estimated_input_tokens")
        or 0
    )
    output_tokens = int(
        telemetry.get("output_tokens" if token_usage_exact else "estimated_output_tokens")
        or 0
    )
    total_tokens = input_tokens + output_tokens
    token_label = "Tokens" if token_usage_exact else "Est. tokens"
    st.markdown(f"### {avatar} {title}")
    cols = st.columns(4)
    cols[0].metric("τ² reward", reward if reward is not None else "n/a")
    cols[1].metric("Native cost", f"${total_cost:.5f}")
    cols[2].metric("Latency", f"{latency_ms / 1000:.2f} s")
    cols[3].metric(token_label, f"{total_tokens:,}")
    detail_cols = st.columns(3)
    detail_cols[0].caption(f"Agent calls: {telemetry.get('llm_calls', 0)}")
    detail_cols[1].caption(f"Tool calls: {telemetry.get('tool_calls', 0)}")
    detail_cols[2].caption(
        f"Termination: {simulation.get('termination_reason', 'unknown')}"
    )
    st.caption(
        f"{token_label} split — input: {input_tokens:,} · "
        f"output: {output_tokens:,}. Native cost is reported by τ²."
    )

    with st.expander("Effective system prompt", expanded=False):
        st.code(arm.get("system_prompt", ""), language="text")
    with st.expander("τ² reward details", expanded=False):
        st.json(simulation.get("reward_info", {}))

    st.markdown("#### Chronological conversation")
    messages = simulation.get("messages", [])
    if not messages:
        st.caption("No messages recorded.")
        return
    interaction_step = 0
    for index, message in enumerate(messages, start=1):
        role = message.get("role", "unknown")
        content = message.get("content")
        tool_calls = message.get("tool_calls") or []
        if role == "user":
            interaction_step += 1
        turn_label = f"Turn {interaction_step}"
        if role in {"user", "assistant"}:
            with st.chat_message(role):
                st.caption(f"{turn_label} · {role} message")
                if content:
                    st.markdown(str(content))
                if tool_calls:
                    render_collapsed_tool_calls(st, tool_calls)
        else:
            with st.container(border=True):
                st.caption(f"{turn_label} · {role} event")
                if content:
                    st.code(str(content), language="json" if role == "tool" else "text")
                if tool_calls:
                    render_collapsed_tool_calls(st, tool_calls)


def render_collapsed_tool_calls(st, tool_calls: list[dict]) -> None:
    """Keep tool-call payloads available without dominating the trace."""
    count = len(tool_calls)
    label = "Tool call" if count == 1 else "Tool calls"
    with st.expander(f"{label} · {count}", expanded=False):
        st.json(tool_calls)


def render_clbench_result(st, result: dict) -> None:
    """Render native CL-Bench scores and the capture/replay handoff."""
    run = result.get("run", result)
    comparison = run.get("comparison", {})
    baseline = comparison.get("baseline_score")
    memory = comparison.get("memory_score")
    gain = comparison.get("gain")
    columns = st.columns(3)
    columns[0].metric("Baseline score", "—" if baseline is None else f"{baseline:.3f}")
    columns[1].metric("Memory score", "—" if memory is None else f"{memory:.3f}")
    columns[2].metric("Memory gain", "—" if gain is None else f"{gain:+.3f}")
    st.caption(
        f"{run.get('task', 'CL-Bench')} · schedule={run.get('schedule') or 'default'} · "
        f"setup={run.get('memory_setup_mode', 'none')}"
    )
    captured = run.get("captured_memories", [])
    if captured:
        st.markdown("**Captured memories**")
        for memory_entry in captured:
            st.markdown(
                f"- `{memory_entry.get('category', 'general')}` — "
                f"{memory_entry.get('fact', '')}"
            )
    with st.expander("Native CL-Bench results", expanded=False):
        st.json(run)


def render_value_result(st, result: dict, spec: dict) -> None:
    """Render the single-run value comparison workspace."""
    summary = result.get("dataset_summary", {})
    aggregate = result.get("evaluation_aggregate", {})
    st.caption(
        f"{summary.get('scenarios', 0)} scenarios · "
        f"{summary.get('total_memories', 0)} seeded memories"
    )

    modes = list(aggregate)
    if aggregate:
        st.markdown("#### Aggregate comparison")
        for mode, metrics in aggregate.items():
            st.markdown(f"**{mode.replace('_', ' ').title()}**")
            cols = st.columns(5)
            cols[0].metric("Quality Δ", metrics.get("avg_quality_delta", 0))
            cols[1].metric("Token ratio", metrics.get("avg_token_efficiency_ratio", 0))
            cols[2].metric("Turn ratio", metrics.get("avg_turn_compression", 0))
            cols[3].metric("Effective savings", metrics.get("total_effective_savings", 0))
            cols[4].metric("Improved", metrics.get("scenarios_quality_improved", 0))

    scenario_results = result.get("run", {}).get("scenario_results", [])
    if not scenario_results:
        st.info("No scenario results were produced.")
        return

    scenario_ids = [item.get("scenario_id", "unknown") for item in scenario_results]
    selected_id = st.selectbox("Scenario", scenario_ids)
    selected = next(item for item in scenario_results if item.get("scenario_id") == selected_id)
    trials = selected.get("trials", {})
    memory_modes = [mode for mode in trials if mode != "baseline"]
    if not memory_modes:
        st.info("This run contains no memory-enabled trial to compare.")
        return
    memory_mode = st.selectbox("Memory trial", memory_modes)

    baseline = trials.get("baseline", {})
    memory = trials.get(memory_mode, {})
    context = result.get("scenario_context", {}).get(selected_id, {})
    memory_facts = [item.get("fact", "") for item in context.get("memory_bank", [])]

    st.markdown("#### Memory used")
    meta_cols = st.columns(3)
    container = result.get("container", {})
    backend = container.get("integration_profile") or spec.get("simulated_tool_system", "unknown")
    meta_cols[0].metric("Backend", backend.split("/")[-1])
    meta_cols[1].metric("Mode", memory_mode.replace("_", " ").title())
    meta_cols[2].metric("Seeded facts", len(memory_facts))
    if memory_facts:
        st.caption("Facts available to the memory-enabled trial:")
        st.markdown(" · ".join(f"`{fact}`" for fact in memory_facts))

    left, right = st.columns(2)
    with left:
        render_trial(st, "⚪ Claude baseline", baseline, memory_facts=[])
    with right:
        render_trial(
            st,
            (
                f"🧠 Claude + {backend.split('@')[0].split('/')[-1]}"
                if container
                else f"🧠 {memory_mode.replace('_', ' ').title()}"
            ),
            memory,
            memory_facts=memory_facts,
        )

    evaluation = result.get("evaluation", {})
    scenario_eval = next(
        (item for item in evaluation.get("scenario_evaluations", [])
         if item.get("scenario_id") == selected_id),
        None,
    )
    if scenario_eval:
        st.markdown("#### Outcome evaluation")
        render_outcome_evaluation(st, scenario_eval, memory_mode)


def render_trial(st, title: str, trial: dict, memory_facts: list[str]) -> None:
    st.markdown(f"### {title}")
    turns = trial.get("turns", [])
    agent_turns = [turn for turn in turns if turn.get("role") == "agent"]
    total_tokens = sum(
        turn.get("input_tokens", 0) + turn.get("output_tokens", 0)
        for turn in turns
    )
    cols = st.columns(3)
    cols[0].metric("Turns", len(agent_turns))
    cols[1].metric("Tokens", total_tokens)
    cols[2].metric("Latency ms", round(sum(turn.get("latency_ms", 0) for turn in turns)))
    status = trial.get("goal_status", "incomplete")
    reason = trial.get("stop_reason")
    if status == "complete":
        st.success(f"Goal complete{f' · {reason}' if reason else ''}")
    elif reason:
        st.warning(f"Goal not verified · stopped because: {reason}")

    for index, turn in enumerate(turns, start=1):
        role = turn.get("role", "unknown")
        chat_role = "assistant" if role == "agent" else "user"
        avatar = "🧠" if role == "agent" and memory_facts else ("⚪" if role == "agent" else "👤")
        with st.chat_message(chat_role, avatar=avatar):
            if role == "user_sim":
                st.caption("Simulated user")
            if turn.get("content"):
                if role == "agent" and memory_facts:
                    render_highlighted_text(st, turn["content"], memory_facts)
                else:
                    st.markdown(turn["content"])
            if turn.get("tool_calls"):
                st.caption(f"Tool calls: {len(turn['tool_calls'])}")
                render_collapsed_tool_calls(st, turn["tool_calls"])
            if turn.get("integration_events"):
                st.caption(f"Integration events: {len(turn['integration_events'])}")
                st.code(json.dumps(turn["integration_events"], indent=2), language="json")


def _memory_terms(facts: list[str]) -> list[str]:
    stop_words = {
        "about", "after", "based", "being", "from", "have", "into", "that",
        "their", "there", "this", "user", "with", "your",
    }
    terms: set[str] = set()
    for fact in facts:
        for term in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]{3,}", fact.lower()):
            if term not in stop_words:
                terms.add(term)
    return sorted(terms, key=len, reverse=True)


def render_highlighted_text(st, text: str, memory_facts: list[str]) -> None:
    """Highlight sentences containing terms from seeded memory.

    This is a presentation heuristic for the value UI. The evaluator's
    outcome explanations remain the authoritative evidence; this makes the
    likely memory contribution visible in the response pane.
    """
    terms = _memory_terms(memory_facts)
    if not terms:
        st.markdown(text)
        return

    chunks = re.split(r"(\n+|(?<=[.!?])\s+)", text)
    rendered: list[str] = []
    for chunk in chunks:
        if not chunk or chunk.isspace():
            rendered.append(chunk)
            continue
        escaped = html.escape(chunk)
        lowered = chunk.lower()
        if any(term in lowered for term in terms):
            rendered.append(
                '<mark title="Matched seeded memory" '
                'style="background:#fff3a3;padding:2px 4px;border-radius:3px">'
                f"{escaped}</mark>"
            )
        else:
            rendered.append(escaped)

    st.markdown(
        "<div style=\"white-space:pre-wrap;line-height:1.55\">"
        + "".join(rendered)
        + "</div>",
        unsafe_allow_html=True,
    )


def render_outcome_evaluation(st, scenario_eval: dict, memory_mode: str) -> None:
    validations = scenario_eval.get("validations_by_mode", {})
    baseline = validations.get("baseline", [])
    memory = validations.get(memory_mode, [])
    columns = st.columns(2)
    for column, label, items in [
        (columns[0], "Baseline", baseline),
        (columns[1], memory_mode.replace("_", " ").title(), memory),
    ]:
        with column:
            st.markdown(f"**{label}**")
            if not items:
                st.caption("No outcome validation recorded.")
            for item in items:
                status = item.get("status", "uncertain").upper()
                score = item.get("score")
                score_text = f" · {score:.2f}" if isinstance(score, (int, float)) else ""
                st.markdown(f"`{status}`{score_text} — {item.get('outcome_id', 'outcome')}")
                explanation = item.get("explanation")
                if explanation:
                    st.caption(explanation)
                evidence = item.get("evidence")
                if evidence:
                    st.caption(f"Evidence: {evidence}")


def render_extraction_result(st, result: dict) -> None:
    """Render extraction run evidence; scoring is saved separately by EVAL."""
    container_summary = result.get("container_summary")
    if container_summary:
        st.markdown("#### Container extraction summary")
        columns = st.columns(4)
        columns[0].metric("Conversations", container_summary.get("conversations", 0))
        columns[1].metric("Stored items", container_summary.get("stored_items", 0))
        columns[2].metric("Expected facts", container_summary.get("expected_facts", 0))
        columns[3].metric("Hook events", container_summary.get("hook_storage_events", 0))
        st.caption(
            "Container extraction currently shows observed storage evidence. "
            "Run a separate evaluation from the EVAL pane for matcher-based metrics."
        )
        for conversation in result.get("run", {}).get("conversation_results", []):
            with st.expander(conversation.get("conversation_id", "conversation")):
                st.markdown("**Stored items**")
                st.json(conversation.get("stored", []))
                st.markdown("**Ground truth**")
                st.json(conversation.get("ground_truth", []))
                if conversation.get("conversation_turns"):
                    st.markdown("**Conversation**")
                    st.json(conversation["conversation_turns"])
                if conversation.get("hook_storage_events"):
                    st.markdown("**Hook storage events**")
                    st.json(conversation["hook_storage_events"])
        return

    run = result.get("run", {})
    conversations = run.get("conversation_results", [])
    tool_calls = sum(
        len(turn.get("tool_calls", []))
        for conversation in conversations
        for turn in conversation.get("turn_results", [])
    )
    st.caption(
        "Benchmark run saved. Use the EVAL pane to score this run and create a "
        "separate evaluation artifact."
    )
    st.json({
        "dataset_summary": result.get("dataset_summary", {}),
        "run_summary": {
            "conversations": len(conversations),
            "tool_calls": tool_calls,
            "total_time_ms": run.get("total_time_ms", 0),
        },
    })


if __name__ == "__main__":
    render_app()

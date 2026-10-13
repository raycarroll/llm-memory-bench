from __future__ import annotations

import json
import html
import re
from pathlib import Path

from llm_memory_bench.bench import RunConfig as BenchRunConfig
from llm_memory_bench.application import RunService, RunSpec
from llm_memory_bench.systems import list_systems
from llm_memory_bench.value.dataset import load_value_dataset


def render_app() -> None:
    import streamlit as st

    st.set_page_config(page_title="LLM Memory Bench", page_icon="🧠", layout="wide")
    st.title("🧠 LLM Memory Bench")
    st.caption("Single-run workspace · comparison UI v0.2")

    with st.sidebar:
        st.header("Run definition")
        kind = st.radio("Benchmark", ["value", "extraction"], format_func=str.title)
        if kind == "value":
            execution_label = st.radio(
                "Execution",
                ["API-level", "Claude container"],
                help=(
                    "Claude container runs a baseline Claude Code trial and a "
                    "separate trial with the configured memory integration."
                ),
            )
        else:
            execution_label = "API-level"
            st.caption("Extraction runs currently use API-level execution.")
        execution = "container" if execution_label == "Claude container" else "api"
        value_datasets = {
            "Quick demo": "scenarios/quick-demo-goal.yaml",
            "τ² airline + retail": "scenarios/tau2/airline-retail.yaml",
        }
        dataset_preset = None
        if kind == "value":
            dataset_preset = st.selectbox("Dataset example", list(value_datasets))
        dataset_path = st.text_input(
            "Dataset path",
            value=(
                value_datasets[dataset_preset]
                if dataset_preset
                else "datasets/test/alpsbench-task1.yaml"
            ),
        )
        goal_id = None
        goal_dataset = None
        if kind == "value":
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
        container_config_path = None
        if execution == "container":
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
            except Exception as exc:
                container_preview = None
                st.warning(f"Container config unavailable: {exc}")
            provider = "anthropic"
            model = container_preview.model if container_preview else "claude-container"
            system = "memoryhub"
            mode = "integration"
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
            provider = st.selectbox("Provider", ["openai", "anthropic", "litellm", "vertex"])
            model = st.text_input("Model", value="gpt-4o")
            system = st.selectbox("Memory system", list_systems())
            mode = st.selectbox("Memory mode", ["prompt", "tool", "both"])
            user_simulator_provider = None
            user_simulator_model = None
            judge_provider = None
            judge_model = None
        max_turns = st.number_input("Max turns", min_value=1, value=10)
        max_scenarios = st.number_input("Max scenarios (0 = all)", min_value=0, value=1)
        judge = st.checkbox("Run outcome judge", value=True)
        submit = st.button("▶ Run benchmark", type="primary", use_container_width=True)

    spec = RunSpec(
        kind=kind,
        execution=execution,
        dataset_path=dataset_path,
        container_config_path=container_config_path,
        goal_id=goal_id,
        provider=provider,
        model=model,
        system=system,
        mode=mode,
        max_turns=int(max_turns),
        max_scenarios=int(max_scenarios) or None,
        judge_during_run=judge,
        judge_provider=judge_provider,
        judge_model=judge_model,
        user_simulator_provider=user_simulator_provider,
        user_simulator_model=user_simulator_model,
    )

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
        run_kind = payload.get("spec", {}).get("kind", spec.kind)
        if run_kind == "value":
            render_value_result(st, result, payload.get("spec", {}))
        else:
            render_extraction_result(st, result)
    else:
        st.info("Run a benchmark to display the baseline-versus-memory comparison here.")


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
    backend = container.get("integration_profile") or spec.get("system", "unknown")
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
                st.code(json.dumps(turn["tool_calls"], indent=2), language="json")
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
    """Keep extraction output inspectable until its dedicated comparison view lands."""
    st.json({
        "dataset_summary": result.get("dataset_summary", {}),
        "evaluation": result.get("evaluation", {}),
    })


if __name__ == "__main__":
    render_app()

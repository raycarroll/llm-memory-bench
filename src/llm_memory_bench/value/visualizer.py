"""Generate HTML visualization of value benchmark results."""

from __future__ import annotations

import json
from pathlib import Path


def generate_html_visualization(result_json: dict, output_path: Path) -> None:
    """Generate standalone HTML with side-by-side chat comparisons."""

    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Value Benchmark Results</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            line-height: 1.6;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
        }

        .header {
            background: white;
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }

        .header h1 {
            color: #2d3748;
            font-size: 2em;
            margin-bottom: 10px;
        }

        .header .summary {
            color: #718096;
            font-size: 0.95em;
        }

        .scenario {
            background: white;
            border-radius: 12px;
            padding: 25px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }

        .scenario-header {
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 2px solid #e2e8f0;
        }

        .scenario-id {
            color: #4a5568;
            font-size: 0.85em;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .task-prompt {
            font-size: 1.2em;
            color: #2d3748;
            font-weight: 600;
            margin: 10px 0;
        }

        .context-section {
            background: #f7fafc;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            border-left: 4px solid #667eea;
        }

        .context-label {
            font-size: 0.85em;
            color: #718096;
            font-weight: 600;
            margin-bottom: 8px;
        }

        .memory-badge {
            background: white;
            border: 1px solid #cbd5e0;
            padding: 8px 12px;
            border-radius: 6px;
            margin: 5px 0;
            font-size: 0.9em;
            display: inline-block;
        }

        .memory-badge::before {
            content: "💾 ";
        }

        .comparison {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 20px 0;
        }

        @media (max-width: 968px) {
            .comparison {
                grid-template-columns: 1fr;
            }
        }

        .trial {
            background: #fafafa;
            border-radius: 8px;
            overflow: hidden;
            border: 2px solid transparent;
        }

        .trial.baseline { border-color: #fc8181; }
        .trial.memory { border-color: #68d391; }

        .trial-header {
            padding: 12px 15px;
            font-weight: 700;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .trial-header.baseline {
            background: linear-gradient(135deg, #fc8181 0%, #f56565 100%);
            color: white;
        }

        .trial-header.memory {
            background: linear-gradient(135deg, #68d391 0%, #48bb78 100%);
            color: white;
        }

        .system-prompt {
            background: #fffbeb;
            border-left: 3px solid #f59e0b;
            padding: 12px;
            margin: 15px;
            font-size: 0.85em;
            color: #78350f;
            border-radius: 4px;
            white-space: pre-wrap;
            font-family: 'Menlo', 'Monaco', monospace;
        }

        .system-prompt-label {
            font-weight: 700;
            display: block;
            margin-bottom: 5px;
        }

        .chat-container {
            padding: 15px;
            min-height: 200px;
        }

        .message {
            margin: 12px 0;
            animation: fadeIn 0.3s ease-in;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .message.user {
            text-align: right;
        }

        .message-bubble {
            display: inline-block;
            padding: 12px 16px;
            border-radius: 18px;
            max-width: 80%;
            word-wrap: break-word;
        }

        .message.user .message-bubble {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-bottom-right-radius: 4px;
        }

        .message.agent .message-bubble {
            background: #e2e8f0;
            color: #2d3748;
            border-bottom-left-radius: 4px;
        }

        .message-role {
            font-size: 0.75em;
            color: #718096;
            margin-bottom: 4px;
            font-weight: 600;
        }

        .message.user .message-role {
            text-align: right;
        }

        .message-role::before {
            margin-right: 4px;
        }

        .message.user .message-role::before { content: "👤 "; }
        .message.agent .message-role::before { content: "🤖 "; }

        .tool-calls {
            background: #fef3c7;
            border: 1px dashed #f59e0b;
            border-radius: 8px;
            padding: 10px;
            margin: 8px 0;
            font-family: 'Menlo', 'Monaco', monospace;
            font-size: 0.85em;
        }

        .tool-call {
            background: white;
            padding: 8px;
            margin: 5px 0;
            border-radius: 4px;
            border-left: 3px solid #f59e0b;
        }

        .tool-call-name {
            font-weight: 700;
            color: #92400e;
            margin-bottom: 3px;
        }

        .tool-call-name::before {
            content: "🔧 ";
        }

        .tool-call-args {
            color: #78350f;
            font-size: 0.9em;
            margin-left: 20px;
            white-space: pre-wrap;
            word-break: break-word;
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
            padding: 20px;
            background: #f7fafc;
            border-radius: 8px;
        }

        .metric {
            text-align: center;
        }

        .metric-label {
            font-size: 0.8em;
            color: #718096;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 5px;
        }

        .metric-value {
            font-size: 1.5em;
            font-weight: 700;
            color: #2d3748;
        }

        .metric-delta {
            font-size: 0.85em;
            margin-left: 5px;
        }

        .metric-delta.positive { color: #38a169; }
        .metric-delta.negative { color: #e53e3e; }

        .footer {
            text-align: center;
            color: white;
            padding: 20px;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
"""

    # Header
    config = result_json.get("config", {})
    summary = result_json.get("dataset_summary", {})
    aggregate_modes = result_json.get("aggregated_metrics", {})
    memory_mode = next(
        (mode for mode in aggregate_modes if mode != "baseline"),
        "memory_prompt",
    )
    aggregated = aggregate_modes.get(memory_mode, {})

    html += f"""
        <div class="header">
            <h1>Value Benchmark Results</h1>
            <div class="summary">
                <strong>Model:</strong> {config.get('provider', 'N/A')}/{config.get('model', 'N/A')} |
                <strong>System:</strong> {config.get('system', 'N/A')} |
                <strong>Mode:</strong> {config.get('injection_mode', 'N/A')} |
                <strong>Scenarios:</strong> {summary.get('scenarios', 0)} |
                <strong>Avg Quality Δ:</strong> <span style="color: {'#38a169' if aggregated.get('avg_quality_delta', 0) > 0 else '#e53e3e'};">{aggregated.get('avg_quality_delta', 0):+.2f}</span>
            </div>
        </div>
"""

    # Scenarios
    per_scenario = {s["scenario_id"]: s for s in result_json.get("per_scenario", [])}

    for conv in result_json.get("raw_conversations", []):
        scenario_id = conv["scenario_id"]
        ctx = conv.get("scenario_context", {})
        metrics = per_scenario.get(scenario_id, {}).get(memory_mode, {})

        html += f"""
        <div class="scenario">
            <div class="scenario-header">
                <div class="scenario-id">{scenario_id}</div>
                <div class="task-prompt">{ctx.get('task_prompt', 'N/A')}</div>
            </div>
"""

        # Context section
        if ctx.get("memory_bank"):
            html += """
            <div class="context-section">
                <div class="context-label">Memory Bank</div>
"""
            for mem in ctx.get("memory_bank", []):
                html += f'                <div class="memory-badge">{mem.get("fact", "")}</div>\n'

            html += "            </div>\n"

        # Comparison
        html += '            <div class="comparison">\n'

        # Baseline trial
        baseline = conv["trials"].get("baseline", {})
        html += """
                <div class="trial baseline">
                    <div class="trial-header baseline">🚫 Baseline (No Memory)</div>
"""

        if baseline.get("system_prompt"):
            system_lines = baseline["system_prompt"].split("\n")
            short_prompt = "\n".join(system_lines[:3])
            if len(system_lines) > 3:
                short_prompt += "\n..."
            html += f'                    <div class="system-prompt"><span class="system-prompt-label">System:</span>{short_prompt}</div>\n'

        html += '                    <div class="chat-container">\n'

        for turn in baseline.get("turns", []):
            if turn["role"] == "user":
                role_class = "user"
                role_label = "User"
            elif turn["role"] == "user_sim":
                role_class = "user"
                role_label = "User (simulated)"
            else:
                role_class = "agent"
                role_label = "Assistant"

            html += f"""                        <div class="message {role_class}">
                            <div class="message-role">{role_label}</div>
"""

            # Show tool calls if present
            if turn.get("tool_calls"):
                html += '                            <div class="tool-calls">\n'
                for tc in turn["tool_calls"]:
                    import json as json_mod
                    args_json = json_mod.dumps(tc.get("arguments", {}), indent=2)
                    html += f"""                                <div class="tool-call">
                                    <div class="tool-call-name">{tc.get("name", "unknown")}</div>
                                    <div class="tool-call-args">{args_json}</div>
                                </div>
"""
                html += '                            </div>\n'

            # Show message content if present
            if turn.get("content"):
                html += f'                            <div class="message-bubble">{turn["content"]}</div>\n'

            html += "                        </div>\n"

        html += "                    </div>\n                </div>\n"

        # Memory trial
        memory = conv["trials"].get(memory_mode, {})
        html += """
                <div class="trial memory">
                    <div class="trial-header memory">✅ With Memory</div>
"""

        if memory.get("system_prompt"):
            system_lines = memory["system_prompt"].split("\n")
            # Show the memory part specifically
            short_prompt = "\n".join(system_lines[-5:]) if len(system_lines) > 5 else memory["system_prompt"]
            html += f'                    <div class="system-prompt"><span class="system-prompt-label">System (with memories):</span>{short_prompt}</div>\n'

        html += '                    <div class="chat-container">\n'

        for turn in memory.get("turns", []):
            if turn["role"] == "user":
                role_class = "user"
                role_label = "User"
            elif turn["role"] == "user_sim":
                role_class = "user"
                role_label = "User (simulated)"
            else:
                role_class = "agent"
                role_label = "Assistant"

            html += f"""                        <div class="message {role_class}">
                            <div class="message-role">{role_label}</div>
"""

            # Show tool calls if present
            if turn.get("tool_calls"):
                html += '                            <div class="tool-calls">\n'
                for tc in turn["tool_calls"]:
                    import json as json_mod
                    args_json = json_mod.dumps(tc.get("arguments", {}), indent=2)
                    html += f"""                                <div class="tool-call">
                                    <div class="tool-call-name">{tc.get("name", "unknown")}</div>
                                    <div class="tool-call-args">{args_json}</div>
                                </div>
"""
                html += '                            </div>\n'

            # Show message content if present
            if turn.get("content"):
                html += f'                            <div class="message-bubble">{turn["content"]}</div>\n'

            html += "                        </div>\n"

        html += "                    </div>\n                </div>\n"
        html += "            </div>\n"  # close comparison

        # Metrics
        quality_delta = metrics.get("quality_delta", 0)
        quality_class = "positive" if quality_delta > 0 else "negative"

        html += f"""
            <div class="metrics-grid">
                <div class="metric">
                    <div class="metric-label">Quality</div>
                    <div class="metric-value">
                        {metrics.get('quality_baseline', 0):.2f} → {metrics.get('quality_memory', 0):.2f}
                        <span class="metric-delta {quality_class}">({quality_delta:+.2f})</span>
                    </div>
                </div>
                <div class="metric">
                    <div class="metric-label">Tokens</div>
                    <div class="metric-value">
                        {metrics.get('baseline_agent_tokens', 0)} → {metrics.get('memory_agent_tokens', 0)}
                    </div>
                </div>
                <div class="metric">
                    <div class="metric-label">Efficiency</div>
                    <div class="metric-value">
                        {metrics.get('token_efficiency_ratio', 0):.2f}×
                    </div>
                </div>
                <div class="metric">
                    <div class="metric-label">Turns</div>
                    <div class="metric-value">
                        {metrics.get('baseline_turns', 0)} → {metrics.get('memory_turns', 0)}
                    </div>
                </div>
                <div class="metric">
                    <div class="metric-label">Value/Token</div>
                    <div class="metric-value">
                        {metrics.get('value_per_token', 0):.4f}
                    </div>
                </div>
            </div>
        </div>
"""

    html += """
        <div class="footer">
            Generated by llm-memory-bench value-visualize
        </div>
    </div>
</body>
</html>
"""

    output_path.write_text(html)

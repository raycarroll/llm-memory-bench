"""Generate value benchmark scenarios from AlpsBench Task 2 data.

AlpsBench Task 2 entries have a memory_list (prior session memories) and
new_dialogue. We use the memory_list as the memory bank and generate
task prompts and quality rubrics via LLM.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from huggingface_hub import hf_hub_download
from rich.console import Console

from ..config import RunConfig, get_provider

console = Console()

REPO_ID = "Cosineyx/Alpsbench"
TASK2_FILES = {
    "dev": "dataset/dev/task2_input.jsonl",
    "dev_ref": "dataset/dev/task2_reference_output.jsonl",
    "validation": "dataset/validation/task2_input.jsonl",
    "validation_ref": "dataset/validation/task2_reference_output.jsonl",
}


def _download_file(filename: str, cache_dir: Path) -> Path:
    return Path(
        hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            repo_type="dataset",
            cache_dir=str(cache_dir),
        )
    )


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _detect_language(text: str) -> str:
    ascii_count = sum(1 for c in text if ord(c) < 128)
    ratio = ascii_count / len(text) if text else 0
    return "en" if ratio > 0.85 else "other"


TASK_GENERATION_PROMPT = """\
You are generating a realistic user request for a benchmark. Given the \
following facts about a user, write a single, natural request that the user \
might send to an AI assistant — one where knowing these facts would help \
the assistant give a better, more personalized answer.

User facts:
{memories}

Requirements:
- Write ONE user message (1-3 sentences)
- The request should be something where the user's background, preferences, \
or constraints are relevant but NOT explicitly stated in the request itself
- Make it sound natural, like a real person typing to an assistant
- Do NOT reference the facts directly — the user assumes the assistant \
either knows or doesn't know these things

Respond with ONLY the user message, no quotes or explanation."""


RUBRIC_GENERATION_PROMPT = """\
You are creating an evaluation rubric for an AI benchmark. Given a user \
request and facts about the user, write a rubric for scoring the assistant's \
response.

User request: {task_prompt}

User facts:
{memories}

Write a rubric that covers:
1. How well the response addresses the user's actual needs (given their facts)
2. Whether the response is personalized vs generic
3. Whether the response avoids asking for information already in the facts
4. Completeness and actionability

Respond with ONLY the rubric text (3-5 bullet points), no preamble."""


def _build_persona_prompt(memory_list: list[dict]) -> str:
    lines = []
    for m in memory_list:
        value = m.get("value", "")
        if value:
            lines.append(f"- {value}")
    return "\n".join(lines)


async def _generate_task_and_rubric(
    memory_list: list[dict],
    provider,
) -> tuple[str, str]:
    memories_text = "\n".join(
        f"- {m.get('value', '')}" for m in memory_list if m.get("value")
    )

    task_response = await provider.generate(
        messages=[
            {
                "role": "user",
                "content": TASK_GENERATION_PROMPT.format(memories=memories_text),
            }
        ],
        tools=[],
        system="You are a benchmark data generator. Be concise and realistic.",
    )
    task_prompt = task_response.text.strip().strip('"')

    rubric_response = await provider.generate(
        messages=[
            {
                "role": "user",
                "content": RUBRIC_GENERATION_PROMPT.format(
                    task_prompt=task_prompt, memories=memories_text
                ),
            }
        ],
        tools=[],
        system="You are a benchmark data generator. Be concise and precise.",
    )
    rubric = rubric_response.text.strip()

    return task_prompt, rubric


async def generate_value_scenarios(
    output_dir: Path,
    generator_config: RunConfig,
    max_scenarios: int | None = None,
    english_only: bool = True,
    cache_dir: Path | None = None,
):
    cache = cache_dir or Path("datasets/alpsbench/raw")
    provider = get_provider(generator_config)

    console.print("Downloading AlpsBench Task 2 data...")

    t2_inputs: list[dict] = []
    for split in ["dev", "validation"]:
        path = _download_file(TASK2_FILES[split], cache)
        t2_inputs.extend(_load_jsonl(path))

    console.print(f"Loaded {len(t2_inputs)} Task 2 entries")

    scenarios = []
    for i, entry in enumerate(t2_inputs):
        memory_list = entry.get("memory_list", [])
        if not memory_list:
            continue

        if english_only:
            all_values = " ".join(m.get("value", "") for m in memory_list[:5])
            if _detect_language(all_values) != "en":
                continue

        console.print(f"  Generating scenario {len(scenarios) + 1}...")

        try:
            task_prompt, rubric = await _generate_task_and_rubric(
                memory_list, provider
            )
        except Exception as e:
            console.print(f"  [yellow]Skipping entry {i}: {e}[/yellow]")
            continue

        session_id = entry.get("session_id", f"alpsbench-value-{i:04d}")

        scenario = {
            "id": session_id,
            "memory_bank": [
                {
                    "fact": m.get("value", ""),
                    "category": m.get("type", "general"),
                }
                for m in memory_list
                if m.get("value")
            ],
            "task_prompt": task_prompt,
            "persona_prompt": _build_persona_prompt(memory_list),
            "quality_rubric": rubric,
            "noise_memories": [],
        }

        scenarios.append(scenario)

        if max_scenarios and len(scenarios) >= max_scenarios:
            break

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "alpsbench-value.yaml"
    with open(output_path, "w") as f:
        yaml.dump(
            {"scenarios": scenarios},
            f,
            default_flow_style=False,
            allow_unicode=True,
            width=120,
        )

    total_memories = sum(len(s["memory_bank"]) for s in scenarios)
    console.print(f"\n[bold]Value Scenarios:[/bold]")
    console.print(f"  Scenarios: {len(scenarios)}")
    console.print(f"  Total memories: {total_memories}")
    console.print(
        f"  Avg memories/scenario: {total_memories / len(scenarios):.1f}"
        if scenarios
        else ""
    )
    console.print(f"  Written to: {output_path}")

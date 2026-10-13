"""Generate TARGETED value benchmark scenarios from AlpsBench.

This version filters for high-value memories and uses templates for quick,
demonstrable value scenarios.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from rich.console import Console

console = Console()

# High-value memory patterns that typically change task outcomes
# Patterns are specific phrases, not single keywords, to avoid false positives
HIGH_VALUE_PATTERNS = {
    'dietary': [
        'is vegetarian', 'is vegan', 'is pescatarian',
        'kosher diet', 'halal diet',
        'allergic to', 'allergy to',
        'gluten-free', 'lactose intolerant',
        'does not eat', 'cannot eat', 'avoid eating'
    ],
    'skill': [
        'senior developer', 'junior developer', 'software engineer',
        'beginner programmer', 'expert in',
        'proficient in python', 'proficient in java',
        'experienced with', 'new to programming'
    ],
    'language_pref': [
        'prefers python', 'prefers java', 'prefers typescript',
        'prefers javascript', 'uses rust', 'codes in go',
        'works with python', 'works with java'
    ],
    'location': [
        'lives in', 'live in', 'based in', 'located in',
        'resides in', 'from portland', 'from seattle'
    ],
    'constraint': [
        'budget of', 'budget is', 'can afford',
        'costs less than', 'under $', 'maximum budget',
        'time limit', 'deadline', 'available time'
    ],
}

# Template tasks that work well with each memory type
VALUE_TEMPLATES = {
    'dietary': [
        ("What should I make for dinner tonight?",
         "- Does response respect dietary restrictions?\n- Is recipe specific and actionable?\n- No questions about diet when memory available"),
        ("Recommend a restaurant for lunch",
         "- Suggests places with appropriate options?\n- Specific recommendations vs generic\n- No asking about dietary needs"),
        ("Quick meal ideas for this week",
         "- All suggestions align with diet?\n- Practical and specific\n- Doesn't ask about preferences"),
    ],
    'skill': [
        ("Help me write a function to validate JSON data",
         "- Appropriate complexity for skill level?\n- Uses right language/tools for background\n- No asking about language preference or skill level"),
        ("Explain how async/await works",
         "- Explanation depth matches skill level?\n- Examples appropriate for background\n- Not over-explained or under-explained"),
        ("Review this code for best practices",
         "- Feedback appropriate for skill level?\n- References relevant to their stack\n- Specific actionable advice"),
    ],
    'language_pref': [
        ("Write a script to process CSV files",
         "- Uses preferred language?\n- Idioms appropriate to that language\n- No asking about language choice"),
        ("Quick example of reading environment variables",
         "- Code in preferred language?\n- Uses standard libraries for that ecosystem\n- Specific not generic"),
    ],
    'location': [
        ("Where should I go for coffee this afternoon?",
         "- Suggests specific local places?\n- Relevant to their city/neighborhood\n- Not generic chains"),
        ("Weekend activity ideas",
         "- Location-appropriate suggestions?\n- Considers local climate/culture\n- Specific venues/places"),
    ],
    'constraint': [
        ("Recommend a laptop for work",
         "- Stays within budget constraint?\n- Specific models within limits\n- No asking about budget"),
        ("Gift ideas for a colleague",
         "- All suggestions respect constraints?\n- Actionable within limits\n- Doesn't ask for budget/time"),
    ],
}


def _detect_language(text: str) -> str:
    ascii_count = sum(1 for c in text if ord(c) < 128)
    ratio = ascii_count / len(text) if text else 0
    return "en" if ratio > 0.85 else "other"


def _classify_memory(value: str) -> str | None:
    """Classify memory into high-value category, or None if not valuable."""
    value_lower = value.lower()

    for category, patterns in HIGH_VALUE_PATTERNS.items():
        if any(p in value_lower for p in patterns):
            return category

    return None


def _build_persona_prompt(memory_list: list[dict]) -> str:
    lines = []
    for m in memory_list:
        value = m.get("value", "")
        if value:
            lines.append(f"- {value}")
    return "\n".join(lines)


def generate_targeted_scenarios(
    output_dir: Path,
    max_scenarios: int | None = None,
    english_only: bool = True,
    cache_dir: Path | None = None,
):
    """Generate value scenarios using templates for quick demonstrable value."""

    console.print("Loading AlpsBench Task 2 data from local files...")

    # Use the existing converted AlpsBench data
    base_dir = Path("datasets/alpsbench/raw/datasets--Cosineyx--Alpsbench/snapshots")

    # Find the snapshot directory
    snapshot_dirs = list(base_dir.glob("*")) if base_dir.exists() else []
    if not snapshot_dirs:
        console.print("[red]Error: AlpsBench data not found. Run 'llm-memory-bench convert --source alpsbench' first.[/red]")
        return

    snapshot_dir = snapshot_dirs[0]  # Use first/only snapshot
    task2_file = snapshot_dir / "dataset/dev/task2/model_input.jsonl"

    if not task2_file.exists():
        console.print(f"[red]Error: Task 2 file not found at {task2_file}[/red]")
        return

    # Load JSONL
    records = []
    with open(task2_file) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    console.print(f"Loaded {len(records)} Task 2 entries")
    console.print("Filtering for high-value memories and generating scenarios...")

    scenarios = []
    stats = {cat: 0 for cat in HIGH_VALUE_PATTERNS.keys()}

    for i, record in enumerate(records):
        # Extract memories from the input field
        input_data = record.get("input", {})
        memory_list = input_data.get("memory", [])

        if not memory_list:
            continue

        # Language filter
        if english_only:
            all_values = " ".join(m.get("value", "") for m in memory_list[:5])
            if _detect_language(all_values) != "en":
                continue

        # Classify memories by value category
        categorized: dict[str, list[dict]] = {}
        for mem in memory_list:
            value = mem.get("value", "")
            if not value:
                continue

            category = _classify_memory(value)
            if category:
                categorized.setdefault(category, []).append({
                    "fact": value,
                    "category": mem.get("type", "general"),
                })

        if not categorized:
            continue

        # Generate scenarios for each category that has memories
        session_id = record.get("session_id", f"alpsbench-value-{i:04d}")

        for category, memories in categorized.items():
            if category not in VALUE_TEMPLATES:
                continue

            # Create one scenario per template task for this category
            for task_idx, (task_prompt, outcome_text) in enumerate(VALUE_TEMPLATES[category]):
                scenario_id = f"{session_id}_{category}_{task_idx}"
                outcomes = [
                    {"id": f"outcome_{n}", "statement": statement.strip("- ")}
                    for n, statement in enumerate(outcome_text.splitlines(), start=1)
                    if statement.strip("- ")
                ]

                scenario = {
                    "id": scenario_id,
                    "memory_bank": memories[:5],  # Keep memory bank small
                    "task_prompt": task_prompt,
                    "persona_prompt": _build_persona_prompt(memories),
                    "outcomes": outcomes,
                    "validators": [{"type": "llm_judge"}],
                    "noise_memories": [],
                    "value_category": category,
                }

                scenarios.append(scenario)
                stats[category] += 1

                if max_scenarios and len(scenarios) >= max_scenarios:
                    break

            if max_scenarios and len(scenarios) >= max_scenarios:
                break

        if max_scenarios and len(scenarios) >= max_scenarios:
            break

    # Write output
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "alpsbench-value-targeted.yaml"
    with open(output_path, "w") as f:
        yaml.dump(
            {"scenarios": scenarios},
            f,
            default_flow_style=False,
            allow_unicode=True,
            width=120,
        )

    # Print stats
    total_memories = sum(len(s["memory_bank"]) for s in scenarios)
    console.print(f"\n[bold]Targeted Value Scenarios:[/bold]")
    console.print(f"  Total scenarios: {len(scenarios)}")
    console.print(f"  Total memories: {total_memories}")
    console.print(
        f"  Avg memories/scenario: {total_memories / len(scenarios):.1f}"
        if scenarios
        else ""
    )
    console.print(f"\n  By category:")
    for cat, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            console.print(f"    {cat}: {count} scenarios")
    console.print(f"\n  Written to: {output_path}")

"""Convert BFCL Memory prerequisite conversations to benchmark format.

BFCL Memory tests end-to-end storage and retrieval using cumulative snapshots.
We extract prerequisite conversations and derive cumulative ground truth from
BFCL's evaluation questions.

Since BFCL uses sequential snapshot accumulation (memory persists across
conversations), we generate cumulative ground truth per conversation based on
which facts' source text appears in that conversation.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from rich.console import Console

console = Console()

MEMORY_DATA_PATH = "berkeley-function-call-leaderboard/bfcl_eval/data"
PREREQ_PATH = "memory_prereq_conversation"

SCENARIOS = ["customer", "healthcare", "finance", "student", "notetaker"]


def _load_jsonl(path: Path) -> list[dict]:
    """Load JSONL file (one JSON object per line)."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _conversation_contains_source(turns: list[dict], source_text: str) -> bool:
    """Check if any turn in the conversation contains the source text.

    Returns True if source text found in any turn, False otherwise.
    """
    if not source_text:
        return False

    # Clean up source text for matching (remove quotes, extra spaces)
    source_clean = source_text.strip().strip('"').strip()

    # Combine all turn content for searching
    full_content = " ".join(turn.get("content", "") for turn in turns)

    # Try exact substring match first
    if source_clean in full_content or full_content in source_clean:
        return True

    # Try fuzzy match - check word overlap
    source_words = set(source_clean.lower().split())
    if len(source_words) < 3:  # Too short for fuzzy matching
        return False

    content_words = set(full_content.lower().split())
    overlap = len(source_words & content_words)

    # Require at least 50% word overlap
    return overlap >= len(source_words) * 0.5


def _map_answers_to_conversation(
    turns: list[dict],
    questions: list[dict],
    answers: list[dict],
) -> list[dict]:
    """Map BFCL questions to this conversation via answer source text.

    Returns a list of facts that should be stored from this conversation.
    Uses cumulative approach - timing doesn't matter, only completeness.
    """
    conversation_facts: list[dict] = []

    # Create answer lookup by question ID
    answer_map = {a["id"]: a for a in answers}

    for question in questions:
        q_id = question["id"]
        answer = answer_map.get(q_id)
        if not answer:
            continue

        # Get the source text that indicates where this fact came from
        source_text = answer.get("source", "")
        ground_truth = answer.get("ground_truth", [])

        if not ground_truth:
            continue

        # Check if this conversation contains the source
        if _conversation_contains_source(turns, source_text):
            # Derive fact from the question if possible
            question_content = question["question"][0][0]["content"]

            # Create a descriptive fact based on question and answer
            fact_description = _derive_fact_description(
                question_content, ground_truth, source_text
            )

            fact = {
                "fact": fact_description,
                "type": "direct",
                "bfcl_question_id": q_id,
                "bfcl_answer": ground_truth,
            }

            conversation_facts.append(fact)

    return conversation_facts


def _derive_fact_description(
    question: str, answer: list | str, source: str
) -> str:
    """Create a descriptive fact from question, answer, and source.

    BFCL provides the source text where the fact appears - this is the most
    accurate representation of what should be stored. We use that directly
    instead of deriving synthetic descriptions from question patterns.

    Examples:
      Q: "What is my first name?"
      A: ["Michael"]
      Source: "My name is Michael and I work at..."
      → "My name is Michael and I work at..."
    """
    # Use source text directly if available (most accurate)
    if source:
        # BFCL source text is already the fact as it appears in conversation
        # Limit to reasonable length (first sentence or ~200 chars)
        source_clean = source.strip()

        # Try to extract first sentence
        sentence_end = min(
            (i for i in [source_clean.find('. '), source_clean.find('? '), source_clean.find('! ')]
             if i > 0),
            default=len(source_clean)
        )

        # Cap at 200 chars or first sentence, whichever is shorter
        if sentence_end < 200:
            return source_clean[:sentence_end + 1].strip()
        else:
            return source_clean[:200].strip() + "..."

    # Fallback if no source (shouldn't happen with BFCL data)
    if isinstance(answer, list):
        answer_str = answer[0] if answer else ""
    else:
        answer_str = str(answer)

    return f"Fact: {answer_str}"


def convert_bfcl_memory(
    bfcl_repo_path: Path,
    output_dir: Path,
    scenarios: list[str] | None = None,
):
    """Convert BFCL memory prerequisite conversations to benchmark format.

    Args:
        bfcl_repo_path: Path to cloned gorilla repo
        output_dir: Directory to write converted datasets
        scenarios: List of scenarios to convert (default: all)
    """
    scenarios = scenarios or SCENARIOS

    data_path = bfcl_repo_path / MEMORY_DATA_PATH
    prereq_path = data_path / PREREQ_PATH

    # Load questions and answers (same for all scenarios)
    questions_file = data_path / "BFCL_v4_memory.json"
    answers_file = data_path / "possible_answer" / "BFCL_v4_memory.json"

    if not questions_file.exists() or not answers_file.exists():
        console.print(
            f"[bold red]Error:[/bold red] BFCL memory data not found in {data_path}\n"
            "Make sure you've cloned the full gorilla repository."
        )
        return

    all_questions = _load_jsonl(questions_file)
    all_answers = _load_jsonl(answers_file)

    output_dir.mkdir(parents=True, exist_ok=True)

    total_conversations = 0
    total_facts = 0

    for scenario in scenarios:
        prereq_file = prereq_path / f"memory_{scenario}.json"

        if not prereq_file.exists():
            console.print(f"[yellow]Skipping {scenario}: file not found[/yellow]")
            continue

        console.print(f"Converting {scenario}...")

        # Load prerequisite conversations for this scenario
        prereq_entries = _load_jsonl(prereq_file)

        # Filter questions and answers for this scenario
        scenario_questions = [q for q in all_questions if q.get("scenario") == scenario]
        scenario_answers = [
            a for a in all_answers
            if any(q["id"] == a["id"] for q in scenario_questions)
        ]

        conversations = []
        all_turns_combined = []  # Collect all turns for fact mapping

        # First pass: build conversations and collect all turns
        for entry in prereq_entries:
            # Each entry is a prerequisite conversation
            entry_id = entry["id"]
            topic = entry.get("topic", "Conversation")
            question_turns = entry.get("question", [])

            # Flatten nested question structure: [[{role, content}], [{...}]]
            turns = []
            for turn_group in question_turns:
                if isinstance(turn_group, list):
                    for turn in turn_group:
                        turns.append({
                            "role": turn.get("role", "user"),
                            "content": turn.get("content", ""),
                        })
                else:
                    turns.append({
                        "role": turn_group.get("role", "user"),
                        "content": turn_group.get("content", ""),
                    })

            # Build conversation WITHOUT ground truth (it's at dataset level)
            conversation_turns = []
            for turn in turns:
                conversation_turns.append({
                    "role": turn["role"],
                    "content": turn["content"],
                    "ground_truth": {
                        "should_store": [],
                        "should_not_store": [],
                    },
                })

            conversation = {
                "id": entry_id,
                "source": f"bfcl-memory-{scenario}",
                "metadata": {
                    "topic": topic,
                    "scenario": scenario,
                },
                "turns": conversation_turns,
            }

            conversations.append(conversation)
            all_turns_combined.extend(turns)
            total_conversations += 1

        # Second pass: derive cumulative ground truth from ALL conversations
        all_cumulative_facts = _map_answers_to_conversation(
            all_turns_combined, scenario_questions, scenario_answers
        )
        total_facts = len(all_cumulative_facts)

        # Write to YAML with dataset-level cumulative ground truth
        output_file = output_dir / f"bfcl_memory_{scenario}.yaml"
        with open(output_file, "w") as f:
            yaml.dump(
                {
                    "ground_truth_type": "cumulative",
                    "cumulative_ground_truth": {
                        "should_store": all_cumulative_facts,
                        "should_not_store": [],
                    },
                    "conversations": conversations,
                },
                f,
                default_flow_style=False,
                allow_unicode=True,
                width=120,
            )

        conv_count = len(conversations)
        fact_count = len(all_cumulative_facts)
        turn_count = sum(len(c["turns"]) for c in conversations)

        console.print(f"  Conversations: {conv_count}")
        console.print(f"  Total turns: {turn_count}")
        console.print(f"  Cumulative facts (derived from BFCL questions): {fact_count}")
        console.print(f"  Ground truth type: cumulative")
        console.print(f"  Written to: {output_file}")
        console.print()

    console.print(f"[bold]Summary:[/bold]")
    console.print(f"  Total conversations: {total_conversations}")
    console.print(f"  Total facts: {total_facts}")
    console.print(f"  Average facts per conversation: {total_facts / total_conversations if total_conversations else 0:.1f}")

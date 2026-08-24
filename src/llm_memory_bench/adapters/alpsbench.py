"""Convert AlpsBench dataset to llm-memory-bench format.

AlpsBench provides conversations with human-verified memory items, each linked
to a source utterance via evidence.utterance_index. This adapter maps those to
our per-turn ground truth format.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from huggingface_hub import hf_hub_download
from rich.console import Console

console = Console()

REPO_ID = "Cosineyx/Alpsbench"
TASK1_FILES = {
    "dev": "dataset/dev/task1/model_input.jsonl",
    "dev_ref": "dataset/dev/task1/reference_output.jsonl",
    "validation": "dataset/validation/task1/model_input.jsonl",
    "validation_ref": "dataset/validation/task1/reference_output.jsonl",
}
TASK2_FILES = {
    "dev": "dataset/dev/task2/model_input.jsonl",
    "dev_ref": "dataset/dev/task2/reference_output.jsonl",
    "validation": "dataset/validation/task2/model_input.jsonl",
    "validation_ref": "dataset/validation/task2/reference_output.jsonl",
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


def _extract_dialogue(input_entry: dict) -> list[dict]:
    """Extract turns from the current schema: input.sessions[].turns[]."""
    sessions = input_entry.get("input", {}).get("sessions", [])
    all_turns = []
    for session in sessions:
        for turn in session.get("turns", []):
            all_turns.append({
                "utterance_index": turn.get("utterance_index", len(all_turns)),
                "role": turn.get("role", "user"),
                "content": turn.get("text", ""),
            })
    return all_turns


def _extract_memories(ref_entry: dict) -> list[dict]:
    """Extract memory items from gold — uses 'memory_items' (task1) or 'answer' (task2)."""
    gold = ref_entry.get("gold", {})
    return gold.get("memory_items", gold.get("answer", []))


def _convert_task1_entry(
    input_entry: dict, ref_entry: dict, index: int
) -> dict | None:
    dialogue = _extract_dialogue(input_entry)
    if not dialogue:
        return None

    memory_items = _extract_memories(ref_entry)

    utterance_to_memories: dict[int, list[dict]] = {}
    for mem in memory_items:
        if mem.get("confidence", 1.0) < 0.5:
            continue
        evidence = mem.get("evidence") or {}
        utt_idx = evidence.get("utterance_index")
        if utt_idx is None:
            continue
        if isinstance(utt_idx, list):
            utt_idx = utt_idx[-1]
        utterance_to_memories.setdefault(utt_idx, []).append(mem)

    turns = []
    for turn in dialogue:
        j = turn["utterance_index"]
        role = turn["role"]
        content = turn["content"]

        should_store = []
        memories_for_turn = utterance_to_memories.get(j, [])
        for mem in memories_for_turn:
            fact_type = "direct" if mem.get("type") == "direct" else "indirect"
            should_store.append(
                {
                    "fact": mem.get("value", ""),
                    "type": fact_type,
                    "source_id": mem.get("memory_id", ""),
                }
            )

        turns.append(
            {
                "role": role,
                "content": content,
                "ground_truth": {
                    "should_store": should_store,
                    "should_not_store": [],
                },
            }
        )

    conv_id = input_entry.get("session_id", f"alpsbench-task1-{index:04d}")

    return {
        "id": conv_id,
        "source": "alpsbench-task1",
        "turns": turns,
    }


def _extract_task2_dialogue(input_entry: dict) -> list[dict]:
    """Extract turns from task2's input.new_dialogue format."""
    new_dialogue = input_entry.get("input", {}).get("new_dialogue", [])
    turns = []
    for i, turn in enumerate(new_dialogue):
        turns.append({
            "utterance_index": i,
            "role": turn.get("role", "user"),
            "content": turn.get("text", ""),
        })
    return turns


def _convert_task2_entry(
    input_entry: dict, ref_entry: dict, index: int
) -> dict | None:
    dialogue = _extract_task2_dialogue(input_entry)
    if not dialogue:
        return None

    ref_memories = _extract_memories(ref_entry)
    input_memories = input_entry.get("input", {}).get("memory", [])

    input_ids = {m.get("memory_id") for m in input_memories}
    ref_ids = {m.get("memory_id") for m in ref_memories}

    new_ids = set()
    updated_ids = set()
    for mid in ref_ids:
        if mid not in input_ids:
            if mid.endswith("_updated"):
                updated_ids.add(mid)
            else:
                new_ids.add(mid)

    new_and_updated = {}
    last_turn_idx = len(dialogue) - 1
    for mem in ref_memories:
        if mem.get("confidence", 1.0) < 0.5:
            continue
        mid = mem.get("memory_id", "")
        if mid not in new_ids and mid not in updated_ids:
            continue
        evidence = mem.get("evidence") or {}
        ev_text = evidence.get("text", "")
        utt_idx = None
        if ev_text:
            for ti, t in enumerate(dialogue):
                if ev_text[:40] in t["content"]:
                    utt_idx = ti
                    break
        if utt_idx is None:
            utt_idx = last_turn_idx
        new_and_updated.setdefault(utt_idx, []).append(mem)

    turns = []
    for turn in dialogue:
        j = turn["utterance_index"]
        role = turn["role"]
        content = turn["content"]

        should_store = []
        for mem in new_and_updated.get(j, []):
            fact_type = "direct" if mem.get("type") == "direct" else "indirect"
            should_store.append(
                {
                    "fact": mem.get("value", ""),
                    "type": fact_type,
                    "source_id": mem.get("memory_id", ""),
                }
            )

        turns.append(
            {
                "role": role,
                "content": content,
                "ground_truth": {
                    "should_store": should_store,
                    "should_not_store": [],
                },
            }
        )

    conv_id = input_entry.get("session_id", f"alpsbench-task2-{index:04d}")

    return {
        "id": conv_id,
        "source": "alpsbench-task2",
        "turns": turns,
    }


def convert_alpsbench(
    output_dir: Path,
    max_conversations: int | None = None,
    english_only: bool = True,
    cache_dir: Path | None = None,
):
    cache = cache_dir or Path("datasets/alpsbench/raw")

    console.print("Downloading AlpsBench dataset from HuggingFace...")

    t1_inputs = []
    t1_refs = []
    t2_inputs = []
    t2_refs = []

    for split in ["dev", "validation"]:
        t1_input_path = _download_file(TASK1_FILES[split], cache)
        t1_ref_path = _download_file(TASK1_FILES[f"{split}_ref"], cache)
        t1_inputs.extend(_load_jsonl(t1_input_path))
        t1_refs.extend(_load_jsonl(t1_ref_path))

        t2_input_path = _download_file(TASK2_FILES[split], cache)
        t2_ref_path = _download_file(TASK2_FILES[f"{split}_ref"], cache)
        t2_inputs.extend(_load_jsonl(t2_input_path))
        t2_refs.extend(_load_jsonl(t2_ref_path))

    console.print(f"Loaded {len(t1_inputs)} Task 1 and {len(t2_inputs)} Task 2 entries")

    task1_convs = []
    for i, (inp, ref) in enumerate(zip(t1_inputs, t1_refs)):
        if english_only:
            dialogue = _extract_dialogue(inp)
            sample_text = " ".join(t["content"] for t in dialogue[:3])
            if _detect_language(sample_text) != "en":
                continue

        conv = _convert_task1_entry(inp, ref, i)
        if conv:
            task1_convs.append(conv)

        if max_conversations and len(task1_convs) >= max_conversations:
            break

    task2_convs = []
    for i, (inp, ref) in enumerate(zip(t2_inputs, t2_refs)):
        if english_only:
            dialogue = _extract_task2_dialogue(inp)
            sample_text = " ".join(t["content"] for t in dialogue[:3])
            if _detect_language(sample_text) != "en":
                continue

        conv = _convert_task2_entry(inp, ref, i)
        if conv:
            task2_convs.append(conv)

        if max_conversations and len(task2_convs) >= max_conversations:
            break

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    task1_path = output_dir / "alpsbench-task1.yaml"
    with open(task1_path, "w") as f:
        yaml.dump(
            {"conversations": task1_convs},
            f,
            default_flow_style=False,
            allow_unicode=True,
            width=120,
        )

    task2_path = output_dir / "alpsbench-task2.yaml"
    with open(task2_path, "w") as f:
        yaml.dump(
            {"conversations": task2_convs},
            f,
            default_flow_style=False,
            allow_unicode=True,
            width=120,
        )

    t1_facts = sum(
        len(t["ground_truth"]["should_store"])
        for c in task1_convs
        for t in c["turns"]
    )
    t1_turns = sum(len(c["turns"]) for c in task1_convs)
    t1_noise = sum(
        1
        for c in task1_convs
        for t in c["turns"]
        if not t["ground_truth"]["should_store"]
    )

    t2_facts = sum(
        len(t["ground_truth"]["should_store"])
        for c in task2_convs
        for t in c["turns"]
    )
    t2_turns = sum(len(c["turns"]) for c in task2_convs)
    t2_noise = sum(
        1
        for c in task2_convs
        for t in c["turns"]
        if not t["ground_truth"]["should_store"]
    )

    console.print(f"\n[bold]Task 1 (Extraction):[/bold]")
    console.print(f"  Conversations: {len(task1_convs)}")
    console.print(f"  Total turns: {t1_turns}")
    console.print(f"  Total facts: {t1_facts}")
    if t1_turns:
        console.print(f"  Noise turns: {t1_noise} ({t1_noise/t1_turns*100:.1f}%)")
    console.print(f"  Written to: {task1_path}")

    console.print(f"\n[bold]Task 2 (Updating):[/bold]")
    console.print(f"  Conversations: {len(task2_convs)}")
    console.print(f"  Total turns: {t2_turns}")
    console.print(f"  Total facts (new/updated): {t2_facts}")
    if t2_turns:
        console.print(f"  Noise turns: {t2_noise} ({t2_noise/t2_turns*100:.1f}%)")
    console.print(f"  Written to: {task2_path}")

"""Drive multi-turn conversations through Claude Code via the Agent SDK.

Feeds user messages sequentially, resuming the same session so the agent
accumulates context and decides what to save — with its full prompt stack
(system prompt, CLAUDE.md, MCP tools, plugins).
"""

import argparse
import asyncio
import json
import sys

from claude_agent_sdk import query, ClaudeAgentOptions, ResultMessage


async def feed_conversation(
    turns: list[dict],
    model: str | None = None,
    cwd: str = "/workspace",
    max_turns: int | None = None,
) -> dict:
    responses = []
    session_id = None

    for i, turn in enumerate(turns):
        if turn["role"] != "user":
            continue

        options = ClaudeAgentOptions(
            resume=session_id,
            permission_mode="auto",
            model=model,
            cwd=cwd,
            max_turns=max_turns,
        )

        result_text = ""
        subtype = None
        try:
            async for message in query(prompt=turn["content"], options=options):
                if isinstance(message, ResultMessage):
                    session_id = message.session_id
                    subtype = message.subtype
                    if message.subtype == "success":
                        result_text = message.result
        except Exception as e:
            responses.append({"turn": i, "error": str(e)[:500]})
            continue

        responses.append({
            "turn": i,
            "response": result_text[:2000],
            "session_id": session_id,
            "subtype": subtype,
        })

    return {
        "turns_processed": len(responses),
        "session_id": session_id,
        "responses": responses,
    }


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--turns", help="JSON array of turns (inline)")
    group.add_argument("--turns-file", help="Path to JSON file with turns")
    parser.add_argument("--model", help="Model to use (e.g. claude-sonnet-4-20250514)")
    parser.add_argument("--cwd", default="/workspace", help="Working directory")
    parser.add_argument("--max-turns", type=int, default=None, help="Max agent turns per user message")
    args = parser.parse_args()

    if args.turns_file:
        with open(args.turns_file) as f:
            turns = json.load(f)
    else:
        turns = json.loads(args.turns)

    result = asyncio.run(feed_conversation(
        turns,
        model=args.model,
        cwd=args.cwd,
        max_turns=args.max_turns,
    ))
    json.dump(result, sys.stdout, indent=2)


if __name__ == "__main__":
    main()

"""Append a safe, provider-neutral integration event to the run event log.

Integration hooks, plugins, and bridge scripts can use this small standalone
helper without importing the benchmark package.  Event records are metadata
only; memory contents, prompts, credentials, and tool arguments do not belong
in this stream.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from pathlib import Path


_SENSITIVE_DETAIL_WORDS = (
    "content",
    "credential",
    "password",
    "prompt",
    "query",
    "secret",
    "token",
    "authorization",
    "api_key",
)


def _safe_detail_key(key: str) -> bool:
    lowered = key.lower()
    return not any(word in lowered for word in _SENSITIVE_DETAIL_WORDS)


def _safe_detail_value(value: str) -> str:
    return value[:120]


def append_event(
    name: str,
    source: str,
    status: str,
    details: dict[str, str] | None = None,
    path: str | None = None,
) -> None:
    configured_path = path or os.environ.get("LLM_MEMORY_BENCH_EVENT_LOG", "").strip()
    if not configured_path:
        return
    event_path = Path(configured_path)
    event_path.parent.mkdir(parents=True, exist_ok=True)
    safe_details = {
        str(key): _safe_detail_value(str(value))
        for key, value in (details or {}).items()
        if _safe_detail_key(str(key))
    }
    event = {
        "id": uuid.uuid4().hex,
        "timestamp": time.time(),
        "name": name[:120],
        "source": source[:120],
        "status": status[:40],
    }
    if safe_details:
        event["details"] = safe_details
    with event_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--detail", action="append", default=[])
    args = parser.parse_args()
    details: dict[str, str] = {}
    for item in args.detail:
        key, separator, value = item.partition("=")
        if separator and key:
            details[key] = value
    append_event(args.name, args.source, args.status, details)


if __name__ == "__main__":
    main()

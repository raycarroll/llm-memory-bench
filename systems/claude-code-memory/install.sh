#!/usr/bin/env bash
set -euo pipefail

# Claude Code's auto-memory is built in — no separate install needed.
# We just need to ensure the memory directory exists.

MEMORY_DIR="${CLAUDE_MEMORY_DIR:-/workspace/.claude/memory}"
mkdir -p "$MEMORY_DIR"

echo "Claude Code auto-memory ready at ${MEMORY_DIR}"

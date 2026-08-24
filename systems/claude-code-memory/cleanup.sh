#!/usr/bin/env bash
set -euo pipefail

MEMORY_DIR="${CLAUDE_MEMORY_DIR:-/workspace/.claude/memory}"

# Remove all memory files but keep the directory and MEMORY.md index
find "$MEMORY_DIR" -name '*.md' ! -name 'MEMORY.md' -delete 2>/dev/null || true

# Reset MEMORY.md to empty index
echo "" > "$MEMORY_DIR/MEMORY.md"

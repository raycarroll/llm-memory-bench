#!/usr/bin/env bash
set -euo pipefail

# Read all memory files written by Claude Code's auto-memory system.
# Memory files are markdown with YAML frontmatter in the memory directory.

MEMORY_DIR="${CLAUDE_MEMORY_DIR:-/workspace/.claude/memory}"

python3 -c "
import json, sys, os, re

memory_dir = '$MEMORY_DIR'
stored = []

for f in sorted(os.listdir(memory_dir)) if os.path.isdir(memory_dir) else []:
    if not f.endswith('.md') or f == 'MEMORY.md':
        continue
    path = os.path.join(memory_dir, f)
    with open(path) as fh:
        content = fh.read()

    # Parse YAML frontmatter
    meta = {}
    body = content
    if content.startswith('---'):
        parts = content.split('---', 2)
        if len(parts) >= 3:
            import yaml
            try:
                meta = yaml.safe_load(parts[1]) or {}
            except:
                pass
            body = parts[2].strip()

    stored.append({
        'type': 'memory',
        'file': f,
        'name': meta.get('name', f.replace('.md', '')),
        'description': meta.get('description', ''),
        'memory_type': meta.get('metadata', {}).get('type', 'unknown') if isinstance(meta.get('metadata'), dict) else 'unknown',
        'content': body,
    })

json.dump({'stored': stored}, sys.stdout, indent=2)
"

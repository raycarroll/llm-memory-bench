#!/usr/bin/env bash
set -euo pipefail

# Trust the /workspace directory for Claude Code
# This allows project-scoped .claude/settings.json to take effect
mkdir -p /root/.claude
cat > /root/.claude.json <<'EOF'
{
  "projects": {
    "/workspace": {
      "hasTrustDialogAccepted": true
    }
  }
}
EOF

# Run the system installer if provided
if [ -n "${SYSTEM_INSTALL_SCRIPT:-}" ] && [ -f "$SYSTEM_INSTALL_SCRIPT" ]; then
    echo "Installing memory system..."
    bash "$SYSTEM_INSTALL_SCRIPT"
fi

exec "$@"

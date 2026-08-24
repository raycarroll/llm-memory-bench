#!/usr/bin/env bash
set -euo pipefail

# Run the system installer if provided
if [ -n "${SYSTEM_INSTALL_SCRIPT:-}" ] && [ -f "$SYSTEM_INSTALL_SCRIPT" ]; then
    echo "Installing memory system..."
    bash "$SYSTEM_INSTALL_SCRIPT"
fi

exec "$@"

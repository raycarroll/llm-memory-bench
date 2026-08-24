#!/usr/bin/env bash
set -euo pipefail

# Wipe all stored data but keep gbrain installed and wired.
# Re-initialises the PGLite database from scratch.

rm -rf ~/.gbrain/data
gbrain init --pglite --yes 2>/dev/null || gbrain init --pglite

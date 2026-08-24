#!/usr/bin/env bash
set -euo pipefail

GBRAIN_VERSION="${GBRAIN_VERSION:-latest-stable}"
GBRAIN_SURFACE="${GBRAIN_SURFACE:-starter}"

echo "Installing gbrain@${GBRAIN_VERSION}..."
npm install -g "github:garrytan/gbrain#${GBRAIN_VERSION}"

echo "Initialising test brain..."
gbrain init --pglite

echo "Wiring gbrain to Claude Code (surface: ${GBRAIN_SURFACE})..."
claude mcp add gbrain -- gbrain serve --surface "$GBRAIN_SURFACE"

echo "gbrain install complete."
gbrain --version

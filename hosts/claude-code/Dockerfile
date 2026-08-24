ARG CLAUDE_CODE_VERSION=1.0.33
FROM node:22-slim

ARG CLAUDE_CODE_VERSION
ENV CLAUDE_CODE_VERSION=${CLAUDE_CODE_VERSION}

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates python3 python3-pip python3-venv unzip \
    && rm -rf /var/lib/apt/lists/*

# Install Claude Code CLI
RUN npm install -g @anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}

# Install the Python Agent SDK (bundles its own Claude Code binary)
RUN pip3 install --break-system-packages claude-agent-sdk

WORKDIR /workspace

# Initialise a git repo so Claude Code doesn't complain
RUN git init && git config user.email "bench@test" && git config user.name "bench"

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

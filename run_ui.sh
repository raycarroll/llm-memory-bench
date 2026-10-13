#!/bin/bash
# Launch the LLM Memory Bench UI
cd "$(dirname "$0")"
python3 -c "from src.llm_memory_bench.ui import main; main()"

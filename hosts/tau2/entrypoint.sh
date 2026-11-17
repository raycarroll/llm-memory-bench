#!/bin/sh
set -eu

if [ "$#" -eq 0 ]; then
    set -- --help
fi

exec llm-memory-bench tau2-value-run "$@"

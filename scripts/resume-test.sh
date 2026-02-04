#!/bin/bash

# Resume a previous fastchess test

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Auto-detect native fastchess vs Docker
USE_NATIVE=false
if command -v fastchess &>/dev/null; then
    USE_NATIVE=true
fi

CONFIG_FILE="$PROJECT_ROOT/target/fastchess/current.json"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "No previous test found. Config file missing: $CONFIG_FILE"
    echo "Start a new test with: make sprt-gain"
    exit 1
fi

echo "Resuming previous test..."
echo "Config: $CONFIG_FILE"

# Run fastchess
cd "$PROJECT_ROOT"
if [ "$USE_NATIVE" = true ]; then
    echo "Using native fastchess..."
    fastchess -config file="$CONFIG_FILE"
else
    echo "Using Docker fastchess..."
    docker compose run --rm fastchess -config file=/state/current.json
fi

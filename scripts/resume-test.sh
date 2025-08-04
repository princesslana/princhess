#!/bin/bash

# Resume a previous fastchess test

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

CONFIG_FILE="$PROJECT_ROOT/target/fastchess/current.json"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "No previous test found. Config file missing: $CONFIG_FILE"
    echo "Start a new test with: make sprt-gain"
    exit 1
fi

echo "Resuming previous test..."
echo "Config: $CONFIG_FILE"

# Start docker compose with resume command
cd "$PROJECT_ROOT"
docker compose run --rm fastchess -config file=/state/current.json
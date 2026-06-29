#!/bin/bash

# Resume a previous tuning session

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

CONFIG_FILE="$PROJECT_ROOT/target/tuning/tuning_config.json"
DATA_FILE="$PROJECT_ROOT/target/tuning/data/data.npz"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "No previous tuning session found. Config file missing: $CONFIG_FILE"
    echo "Start a new tuning session with: ./scripts/start-tune.sh"
    exit 1
fi

if [ ! -f "$DATA_FILE" ]; then
    echo "No previous tuning data found: $DATA_FILE"
    echo "This doesn't look like an interrupted session. Start fresh with: ./scripts/start-tune.sh"
    exit 1
fi

echo "Resuming previous tuning session..."
echo "Config: $CONFIG_FILE"
echo "Data: $DATA_FILE"
echo ""

# Start docker compose
cd "$PROJECT_ROOT"
docker compose run --rm tune

#!/bin/bash

# Start a new tuning session
# Usage: start-tune.sh <tune_type> <size> <param1> [param2] [param3] ...

set -e

TUNE_TYPE=$1
SIZE=$2
shift 2
PARAMS=("$@")

if [ -z "$TUNE_TYPE" ] || [ -z "$SIZE" ] || [ ${#PARAMS[@]} -eq 0 ]; then
    echo "Usage: $0 <tune_type> <size> <param1> [param2] [param3] ..."
    echo "  tune_type: 25k, stc"
    echo "  size: small (±5%), medium (±25%), large (±100%)"
    echo "  params: UCI option names (e.g., CPuct, CPuctTau, PolicyTemperatureRoot)"
    echo ""
    echo "Examples:"
    echo "  $0 25k small CPuct CPuctTau PolicyTemperatureRoot"
    echo "  $0 stc medium CPuctJitter"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Wipe previous tuning state
echo "Wiping previous tuning state..."
rm -rf "$PROJECT_ROOT/target/tuning"
mkdir -p "$PROJECT_ROOT/target/tuning"

# Generate tuning config
echo "Generating tuning config for $TUNE_TYPE tune (size: $SIZE)"
echo "Parameters: ${PARAMS[*]}"
echo ""
"$SCRIPT_DIR/generate-tuning-config.sh" "$TUNE_TYPE" "$SIZE" "${PARAMS[@]}" > "$PROJECT_ROOT/target/tuning/tuning_config.json"
echo "Config written to target/tuning/tuning_config.json"

# Start docker compose
echo ""
echo "Starting tuning..."
cd "$PROJECT_ROOT"
docker compose run --rm tune

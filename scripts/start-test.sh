#!/bin/bash

# Start a new fastchess test
# Usage: start-test.sh <test_type> <time_control> <engine1> <engine2> [thread_config] [syzygy]

set -e

TEST_TYPE=$1
TIME_CONTROL=$2
ENGINE1=$3
ENGINE2=$4
THREAD_CONFIG=${5:-1t}
USE_SYZYGY=${6:-true}

if [ -z "$TEST_TYPE" ] || [ -z "$TIME_CONTROL" ] || [ -z "$ENGINE1" ] || [ -z "$ENGINE2" ]; then
    echo "Usage: $0 <test_type> <time_control> <engine1> <engine2> [thread_config] [syzygy]"
    echo "  test_type: sprt_gain, sprt_equal, elo_check, debug"
    echo "  time_control: stc, ltc, nodes25k"
    echo "  engine1: princhess, princhess-main, etc"
    echo "  engine2: princhess-main, stockfish, etc"
    echo "  thread_config: 1t, 2t, 4t, etc (default: 1t)"
    echo "  syzygy: true or false (default: true)"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Create target directory if it doesn't exist
mkdir -p "$PROJECT_ROOT/target/fastchess"

# Generate fastchess arguments
echo "Generating command for $TEST_TYPE test ($TIME_CONTROL, $THREAD_CONFIG, syzygy=$USE_SYZYGY): $ENGINE1 vs $ENGINE2"
FASTCHESS_ARGS=$("$SCRIPT_DIR/generate-args.sh" "$TEST_TYPE" "$TIME_CONTROL" "$ENGINE1" "$ENGINE2" "$THREAD_CONFIG" "$USE_SYZYGY")

echo "Starting fastchess with args:"
echo "$FASTCHESS_ARGS"

# Start docker compose with command override
cd "$PROJECT_ROOT"
docker compose run --rm fastchess $FASTCHESS_ARGS
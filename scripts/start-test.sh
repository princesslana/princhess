#!/bin/bash

# Start a new fastchess test

set -e

show_help() {
    cat << 'EOF'
Usage: $0 --test-type <type> --tc <control> --engine1 <name> --engine2 <name> [options]

Required:
  --test-type <type>     Test type: sprt_gain, sprt_equal, elo_check, debug
  --tc <control>         Time control: stc, ltc, nodes25k
  --engine1 <name>       First engine name
  --engine2 <name>       Second engine name

Optional:
  --threads <n>          Threads per game (default: 1)
  --syzygy <bool>        Use Syzygy tablebases: true/false (default: true)
  --max-cores <n>        Max cores available (overrides auto-detection)
  -h, --help             Show this help
EOF
    exit 0
}

# Parse arguments and forward to generate-args.sh
ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Auto-detect native fastchess vs Docker
USE_NATIVE=false
if command -v fastchess &>/dev/null; then
    USE_NATIVE=true
fi

# Create target directory if it doesn't exist
mkdir -p "$PROJECT_ROOT/target/fastchess"
mkdir -p "$PROJECT_ROOT/target/fastchess/pgn"

# Generate fastchess arguments
echo "Starting fastchess test..."
if [ "$USE_NATIVE" = true ]; then
    FASTCHESS_ARGS=$("$SCRIPT_DIR/generate-args.sh" --native "${ARGS[@]}")
else
    FASTCHESS_ARGS=$("$SCRIPT_DIR/generate-args.sh" "${ARGS[@]}")
fi

echo "Starting fastchess with args:"
echo "$FASTCHESS_ARGS"

# Run fastchess
cd "$PROJECT_ROOT"
if [ "$USE_NATIVE" = true ]; then
    echo "Using native fastchess..."
    fastchess $FASTCHESS_ARGS
else
    echo "Using Docker fastchess..."
    docker compose run --rm fastchess $FASTCHESS_ARGS
fi

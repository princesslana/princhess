#!/bin/bash

# Compare NPS between two engine builds by running bench multiple times.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

ENGINE1="${1:-princhess}"
ENGINE2="${2:-princhess-main}"
RUNS="${3:-10}"

get_engine_path() {
    local engine=$1
    if [ -f "$PROJECT_ROOT/builds/$engine" ]; then
        echo "$PROJECT_ROOT/builds/$engine"
    else
        echo "$PROJECT_ROOT/target/release/$engine"
    fi
}

# Prints per-run nps lines, then prints the average as the last line.
run_bench() {
    local engine_path=$1
    local total=0
    local min=0
    local max=0
    local first=true

    for i in $(seq 1 "$RUNS"); do
        local nps
        # "Bench: <nodes> nodes <nps> nps" — the number is the second-to-last field
        nps=$("$engine_path" bench 2>&1 | awk '/^Bench:/ { print $(NF-1) }')
        total=$((total + nps))
        if [ "$first" = true ]; then
            min=$nps
            max=$nps
            first=false
        else
            [ "$nps" -lt "$min" ] && min=$nps
            [ "$nps" -gt "$max" ] && max=$nps
        fi
        printf "  run %2d: %d nps\n" "$i" "$nps" >&2
    done

    local avg=$((total / RUNS))
    printf "  avg: %d  min: %d  max: %d\n" "$avg" "$min" "$max" >&2
    echo "$avg"
}

ENGINE1_PATH=$(get_engine_path "$ENGINE1")
ENGINE2_PATH=$(get_engine_path "$ENGINE2")

for path in "$ENGINE1_PATH" "$ENGINE2_PATH"; do
    if [ ! -f "$path" ]; then
        echo "Error: engine not found: $path"
        exit 1
    fi
done

echo "Bench: $ENGINE1 vs $ENGINE2 ($RUNS runs)"
echo ""

echo "$ENGINE1 ($ENGINE1_PATH):"
AVG1=$(run_bench "$ENGINE1_PATH" | tail -1)
echo ""

echo "$ENGINE2 ($ENGINE2_PATH):"
AVG2=$(run_bench "$ENGINE2_PATH" | tail -1)
echo ""

RATIO=$(awk -v a="$AVG1" -v b="$AVG2" 'BEGIN { printf "%.2f", (a / b) * 100 }')
DIFF=$((AVG1 - AVG2))
echo "Result: $ENGINE1 is ${RATIO}% of $ENGINE2 ($(printf '%+d' "$DIFF") nps)"

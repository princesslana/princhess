#!/bin/bash

# Compare NPS between two engine builds by running bench multiple times.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

ENGINE1="${1:-princhess}"
ENGINE2="${2:-princhess-main}"
RUNS="${3:-10}"

if ! [[ "$RUNS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: RUNS must be a positive integer, got '$RUNS'"
    exit 1
fi

get_engine_path() {
    local engine=$1
    if [ -f "$PROJECT_ROOT/builds/$engine" ]; then
        echo "$PROJECT_ROOT/builds/$engine"
    else
        echo "$PROJECT_ROOT/target/release/$engine"
    fi
}

print_fingerprint() {
    local engine_path=$1
    "$engine_path" fingerprint 2>&1 | sed 's/^info string /  /' >&2
}

# Prints per-run nodes/nps lines, then prints avg nps as the last line.
# Also checks that node count is consistent across all runs.
run_bench() {
    local engine_path=$1
    local total=0
    local min=0
    local max=0
    local expected_nodes=""
    local nodes_ok=true
    local first=true

    for i in $(seq 1 "$RUNS"); do
        local nodes nps
        # "Bench: <nodes> nodes <nps> nps"
        read -r nodes nps < <("$engine_path" bench 2>&1 | awk '/^Bench:/ { print $2, $(NF-1) }')
        total=$((total + nps))
        if [ "$first" = true ]; then
            min=$nps
            max=$nps
            expected_nodes=$nodes
            first=false
        else
            [ "$nps" -lt "$min" ] && min=$nps
            [ "$nps" -gt "$max" ] && max=$nps
            [ "$nodes" != "$expected_nodes" ] && nodes_ok=false
        fi
        printf "  run %2d: %d nodes  %d nps\n" "$i" "$nodes" "$nps" >&2
    done

    local avg=$((total / RUNS))
    if [ "$nodes_ok" = true ]; then
        printf "  avg: %d  min: %d  max: %d  nodes: %d\n" "$avg" "$min" "$max" "$expected_nodes" >&2
    else
        printf "  avg: %d  min: %d  max: %d  nodes: INCONSISTENT\n" "$avg" "$min" "$max" >&2
    fi
    echo "$avg $expected_nodes"
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
print_fingerprint "$ENGINE1_PATH"
read -r AVG1 NODES1 < <(run_bench "$ENGINE1_PATH" | tail -1)
echo ""

echo "$ENGINE2 ($ENGINE2_PATH):"
print_fingerprint "$ENGINE2_PATH"
read -r AVG2 NODES2 < <(run_bench "$ENGINE2_PATH" | tail -1)
echo ""

RATIO=$(awk -v a="$AVG1" -v b="$AVG2" 'BEGIN { printf "%.2f", (a / b) * 100 }')
DIFF=$((AVG1 - AVG2))
if [ "$NODES1" = "$NODES2" ]; then
    NODE_STATUS="nodes match ($NODES1)"
else
    NODE_STATUS="nodes MISMATCH: $ENGINE1=$NODES1 $ENGINE2=$NODES2"
fi
echo "Result: $ENGINE1 is ${RATIO}% of $ENGINE2 ($(printf '%+d' "$DIFF") nps)  $NODE_STATUS"

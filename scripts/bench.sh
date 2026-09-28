#!/bin/bash

# Compare NPS between two engine builds by running bench in interleaved pairs, alternating which
# engine goes first, so machine load drifts affect both engines equally.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

ENGINE1="${1:-princhess}"
ENGINE2="${2:-princhess-main}"
RUNS="${3:-20}"

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
    "$engine_path" fingerprint 2>&1 | sed 's/^info string /  /'
}

# Prints "nodes nps" for a single bench run, or fails if the run fails or produces non-numeric output.
run_bench() {
    local engine_path=$1
    local bench_out nodes nps
    bench_out=$("$engine_path" bench 2>&1) || return 1
    read -r nodes nps <<< "$(printf '%s\n' "$bench_out" | awk '/^Bench:/ { print $2, $(NF-1) }')"

    [[ "$nodes" =~ ^[0-9]+$ ]] && [[ "$nps" =~ ^[1-9][0-9]*$ ]] || return 1
    echo "$nodes $nps"
}

ENGINE1_PATH=$(get_engine_path "$ENGINE1")
ENGINE2_PATH=$(get_engine_path "$ENGINE2")

for path in "$ENGINE1_PATH" "$ENGINE2_PATH"; do
    if [ ! -f "$path" ]; then
        echo "Error: engine not found: $path"
        exit 1
    fi
done

echo "Bench: $ENGINE1 vs $ENGINE2 ($RUNS interleaved pairs)"
echo ""
echo "$ENGINE1 ($ENGINE1_PATH):"
print_fingerprint "$ENGINE1_PATH"
echo ""
echo "$ENGINE2 ($ENGINE2_PATH):"
print_fingerprint "$ENGINE2_PATH"
echo ""

NPS1=()
NPS2=()
NODES1=""
NODES2=""
NODES_OK=true

for i in $(seq 1 "$RUNS"); do
    if [ $((i % 2)) -eq 1 ]; then
        RESULT1=$(run_bench "$ENGINE1_PATH") || { echo "  pair $i: $ENGINE1 FAILED"; exit 1; }
        RESULT2=$(run_bench "$ENGINE2_PATH") || { echo "  pair $i: $ENGINE2 FAILED"; exit 1; }
    else
        RESULT2=$(run_bench "$ENGINE2_PATH") || { echo "  pair $i: $ENGINE2 FAILED"; exit 1; }
        RESULT1=$(run_bench "$ENGINE1_PATH") || { echo "  pair $i: $ENGINE1 FAILED"; exit 1; }
    fi

    read -r N1 P1 <<< "$RESULT1"
    read -r N2 P2 <<< "$RESULT2"

    [ -z "$NODES1" ] && NODES1=$N1
    [ -z "$NODES2" ] && NODES2=$N2
    [[ "$N1" != "$NODES1" || "$N2" != "$NODES2" ]] && NODES_OK=false

    NPS1+=("$P1")
    NPS2+=("$P2")
    printf "  pair %2d: %s %d nps  %s %d nps\n" "$i" "$ENGINE1" "$P1" "$ENGINE2" "$P2"
done

echo ""

# Mean of per-pair NPS ratios with a 95% confidence interval using Student's t (needs two pairs)
read -r AVG1 AVG2 MEAN LOW HIGH < <(
    paste <(printf '%s\n' "${NPS1[@]}") <(printf '%s\n' "${NPS2[@]}") | awk '
        BEGIN {
            split("12.706 4.303 3.182 2.776 2.571 2.447 2.365 2.306 2.262 2.228 " \
                  "2.201 2.179 2.160 2.145 2.131 2.120 2.110 2.101 2.093 2.086 " \
                  "2.080 2.074 2.069 2.064 2.060 2.056 2.052 2.048 2.045 2.042", t)
        }
        { s1 += $1; s2 += $2; r[NR] = $1 / $2; sum += r[NR] }
        END {
            n = NR; mean = sum / n
            if (n < 2) {
                printf "%d %d %.2f - -\n", s1 / n, s2 / n, mean * 100
                exit
            }
            for (i = 1; i <= n; i++) ss += (r[i] - mean) ^ 2
            df = n - 1
            crit = df <= 30 ? t[df] : 1.96 + 2.4 / df
            half = crit * sqrt(ss / df) / sqrt(n)
            printf "%d %d %.2f %.2f %.2f\n", s1 / n, s2 / n, mean * 100, (mean - half) * 100, (mean + half) * 100
        }'
)

printf "  avg: %s %d nps  %s %d nps\n" "$ENGINE1" "$AVG1" "$ENGINE2" "$AVG2"
echo ""

if [ "$NODES_OK" != true ]; then
    NODE_STATUS="nodes INCONSISTENT"
elif [ "$NODES1" = "$NODES2" ]; then
    NODE_STATUS="nodes match ($NODES1)"
else
    NODE_STATUS="nodes MISMATCH: $ENGINE1=$NODES1 $ENGINE2=$NODES2"
fi

if [ "$LOW" = "-" ]; then
    CI="no CI with one pair"
else
    CI="95% CI ${LOW}% to ${HIGH}%"
fi

echo "Result: $ENGINE1 is ${MEAN}% of $ENGINE2 ($CI)  $NODE_STATUS"

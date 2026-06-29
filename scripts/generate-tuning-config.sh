#!/bin/bash

# Generate tuning configuration file
# Usage: generate-tuning-config.sh <tune_type> <size> <param1> [param2] [param3] ...

set -e

# Helper to write to stderr
err() { echo "$@" >&2; }

TUNE_TYPE=$1
SIZE=$2
shift 2
PARAMS=("$@")

if [ -z "$TUNE_TYPE" ] || [ -z "$SIZE" ] || [ ${#PARAMS[@]} -eq 0 ]; then
    echo "Usage: $0 <tune_type> <size> <param1> [param2] [param3] ..."
    echo "  tune_type: 25k, stc"
    echo "  size: small (±25%), medium (±50%), large (±100%)"
    echo "  params: UCI option names (e.g., CPuct, CPuctTau, PolicyTemperatureRoot)"
    echo ""
    echo "To see available parameters, run: ./target/release/princhess \"uci\" | grep \"option name\""
    exit 1
fi

# Validate tune type
case $TUNE_TYPE in
    25k|stc) ;;
    *)
        echo "Invalid tune type: $TUNE_TYPE (must be '25k' or 'stc')"
        exit 1
        ;;
esac

# Validate size
case $SIZE in
    small|medium|large) ;;
    *)
        echo "Invalid size: $SIZE (must be 'small', 'medium', or 'large')"
        exit 1
        ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

ENGINE="$PROJECT_ROOT/target/release/princhess"

# Check engine exists
if [ ! -f "$ENGINE" ]; then
    err "Engine not found: $ENGINE"
    err "Run 'cargo build --release' first"
    exit 1
fi

# Parse UCI options from engine
# Try to run locally first, if that fails (cross-compiled binary), use docker
if UCI_OUTPUT=$("$ENGINE" "uci" 2>/dev/null); then
    : # Success, UCI_OUTPUT is set
else
    # Binary is cross-compiled, run via docker
    UCI_OUTPUT=$(cd "$PROJECT_ROOT" && docker compose run --rm --entrypoint /engines/princhess fastchess "uci" 2>/dev/null)
    if [ $? -ne 0 ]; then
        err "Failed to get UCI output from engine"
        err "Try running: docker compose run --rm --entrypoint /engines/princhess fastchess \"uci\""
        exit 1
    fi
fi

# Get default value for a UCI option
get_default_value() {
    local option_name=$1
    echo "$UCI_OUTPUT" | grep "option name $option_name " | sed -n 's/.*default \([^ ]*\).*/\1/p'
}

# Get min value for a UCI option
get_min_value() {
    local option_name=$1
    echo "$UCI_OUTPUT" | grep "option name $option_name " | sed -n 's/.*min \([^ ]*\).*/\1/p'
}

# Get max value for a UCI option
get_max_value() {
    local option_name=$1
    echo "$UCI_OUTPUT" | grep "option name $option_name " | sed -n 's/.*max \([^ ]*\).*/\1/p'
}

# Validate option exists
validate_option() {
    local option_name=$1
    if ! echo "$UCI_OUTPUT" | grep -q "option name $option_name "; then
        err "Error: Unknown UCI option '$option_name'"
        err ""
        err "Available options:"
        echo "$UCI_OUTPUT" | grep "option name" | sed 's/option name \([^ ]*\).*/  \1/' >&2
        exit 1
    fi
}

# Calculate range based on size mode
calculate_range() {
    local value=$1
    local size=$2
    local percentage

    case $size in
        small)
            percentage=25
            ;;
        medium)
            percentage=50
            ;;
        large)
            percentage=100
            ;;
    esac

    # Calculate min and max using bash arithmetic
    local min=$(( value * (100 - percentage) / 100 ))
    local max=$(( value * (100 + percentage) / 100 ))

    # Ensure min is at least 1 for positive values
    if [ "$min" -lt 1 ] && [ "$value" -gt 0 ]; then
        min=1
    fi

    echo "$min" "$max"
}

# Build parameter ranges JSON
build_parameter_ranges() {
    local ranges="{"
    local first=true

    for param in "${PARAMS[@]}"; do
        # Validate option exists
        validate_option "$param"

        # Get default value
        local default=$(get_default_value "$param")
        if [ -z "$default" ]; then
            err "Error: Could not find default value for $param"
            exit 1
        fi

        # Get UCI min/max bounds
        local uci_min=$(get_min_value "$param")
        local uci_max=$(get_max_value "$param")

        # Calculate range
        read calc_min calc_max < <(calculate_range "$default" "$SIZE")

        # Clamp to UCI bounds
        local min=$calc_min
        local max=$calc_max
        if [ -n "$uci_min" ] && [ "$min" -lt "$uci_min" ]; then
            min=$uci_min
        fi
        if [ -n "$uci_max" ] && [ "$max" -gt "$uci_max" ]; then
            max=$uci_max
        fi

        if [ "$first" = true ]; then
            first=false
        else
            ranges+=","
        fi

        ranges+="
    \"$param\": \"Integer($min, $max)\""
    done

    ranges+="
  }"

    echo "$ranges"
}

# Set tune type specific values
ROUNDS=15

case $TUNE_TYPE in
    25k)
        NPM=25000
        ADJUDICATE_DRAWS=false
        ADJUDICATE_RESIGN=false
        ADJUDICATE_TB=false
        ACQ_FUNCTION="ts"
        ;;
    stc)
        TC="8+0.08"
        ADJUDICATE_DRAWS=true
        ADJUDICATE_RESIGN=true
        ADJUDICATE_TB=true
        ACQ_FUNCTION="pvrs"
        ;;
esac

# Generate parameter ranges
PARAMETER_RANGES=$(build_parameter_ranges)

# Output config to stdout
cat <<EOF
{
  "engines": [
    {
      "command": "/engines/princhess",
      "fixed_parameters": {
        "Threads": 1,
        "Hash": 128,
        "SyzygyPath": "/syzygy"
      }
    },
    {
      "command": "/engines/builds/princhess-main",
      "fixed_parameters": {
        "Threads": 1,
        "Hash": 128,
        "SyzygyPath": "/syzygy"
      }
    }
  ],
  "parameter_ranges": $PARAMETER_RANGES,
  "rounds": $ROUNDS,
EOF

# Add time control or nodes per move
if [ "$TUNE_TYPE" = "25k" ]; then
    cat <<EOF
  "engine1_npm": $NPM,
  "engine2_npm": $NPM,
EOF
else
    cat <<EOF
  "engine1_tc": "$TC",
  "engine2_tc": "$TC",
EOF
fi

# Add rest of config
cat <<EOF
  "opening_file": "/books/UHO_Lichess_4852_v1.epd",
  "adjudicate_draws": $ADJUDICATE_DRAWS,
  "adjudicate_resign": $ADJUDICATE_RESIGN,
  "adjudicate_tb": $ADJUDICATE_TB,
  "concurrency": 5,
  "plot_every": 50,
  "result_every": 10,
  "acq_function": "$ACQ_FUNCTION"
}
EOF

#!/bin/bash

# Generate fastchess command line arguments
# Usage: generate-args.sh <test_type> <time_control> <engine1> <engine2> [thread_config] [syzygy]

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

# Parse and validate thread config (must be positive integer followed by 't')
if ! [[ "$THREAD_CONFIG" =~ ^[1-9][0-9]*t$ ]]; then
    echo "Invalid thread config: $THREAD_CONFIG (must be positive integer followed by 't', like '2t', '4t', etc)"
    exit 1
fi

# Extract thread count (e.g., "2t" -> 2)
THREADS=$(echo "$THREAD_CONFIG" | sed 's/t$//')

# Validate syzygy parameter
if [ "$USE_SYZYGY" != "true" ] && [ "$USE_SYZYGY" != "false" ]; then
    echo "Invalid syzygy parameter: $USE_SYZYGY (must be 'true' or 'false')"
    exit 1
fi

# Determine if this is a "long" test (needs adjudication and relaxed SPRT)
is_long_test() {
    case $TIME_CONTROL in
        ltc) return 0 ;;  # Long time control
        *) 
            # Multi-threaded tests are long due to reduced concurrency
            if [ "$THREADS" -gt 1 ]; then
                return 0
            else
                return 1
            fi
            ;;
    esac
}

# Set common values for all time controls
HASH=128
# Calculate concurrency to use 6 total threads (6 / threads_per_game)
# Debug tests always use concurrency=1 for easier troubleshooting
if [ "$TEST_TYPE" = "debug" ]; then
    CONCURRENCY=1
else
    CONCURRENCY=$((6 / THREADS))
    if [ $CONCURRENCY -lt 1 ]; then
        CONCURRENCY=1
    fi
fi
OPENING_BOOK="UHO_Lichess_4852_v1.epd"

# Set time control specific values
case $TIME_CONTROL in
    stc)
        TC="8+0.08"
        ;;
    ltc)
        TC="40+0.4"
        ;;
    nodes25k)
        TC="inf nodes=25000"
        ;;
    *)
        echo "Unknown time control: $TIME_CONTROL"
        exit 1
        ;;
esac

# Generate command line arguments
# Function to find engine path
get_engine_path() {
    local engine=$1
    if [ -f "./builds/$engine" ]; then
        # Engine exists in local builds directory, will be available at /engines/builds/ in container
        echo "/engines/builds/$engine"
    else
        # Use direct engine mount (e.g., princhess from target/release)
        echo "/engines/$engine"
    fi
}

echo "-engine cmd=$(get_engine_path $ENGINE1) name=$ENGINE1"
echo "-engine cmd=$(get_engine_path $ENGINE2) name=$ENGINE2"
echo ""
echo "-each proto=uci tc=$TC"
if [ "$USE_SYZYGY" = "true" ]; then
    echo "      option.SyzygyPath=/syzygy option.Hash=$HASH option.Threads=$THREADS"
else
    echo "      option.SyzygyPath=<empty> option.Hash=$HASH option.Threads=$THREADS"
fi

# Add adjudication for long tests
if is_long_test; then
    echo "-resign movecount=3 score=500 twosided=true"
    echo "-draw movenumber=40 movecount=8 score=15"
fi

# Add test-specific arguments
case $TEST_TYPE in
    sprt_gain)
        ROUNDS=7500
        if is_long_test; then
            echo "-sprt elo0=0 elo1=10 alpha=0.1 beta=0.2 model=normalized"
        else
            echo "-sprt elo0=0 elo1=10 alpha=0.05 beta=0.1 model=normalized"
        fi
        ;;
    sprt_equal)
        ROUNDS=7500
        if is_long_test; then
            echo "-sprt elo0=-10 elo1=0 alpha=0.1 beta=0.2 model=normalized"
        else
            echo "-sprt elo0=-10 elo1=0 alpha=0.05 beta=0.1 model=normalized"
        fi
        ;;
    elo_check)
        ROUNDS=500
        ;;
    debug)
        ROUNDS=50
        echo "-log file=/pgn/debug.log engine=true"
        echo "-pgnout /pgn/debug.pgn"
        ;;
    *)
        echo "Unknown test type: $TEST_TYPE"
        exit 1
        ;;
esac

echo "-openings file=/books/$OPENING_BOOK format=epd order=random"
echo "-games 2 -repeat -rounds $ROUNDS"
echo "-ratinginterval 10 -concurrency $CONCURRENCY"

if [ "$TEST_TYPE" != "debug" ]; then
    echo "-recover"
fi

echo "-config outname=/state/current.json"
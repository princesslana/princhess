#!/bin/bash

# Generate fastchess command line arguments
# Usage: generate-args.sh <test_type> <time_control> <engine1> <engine2> [thread_config]

set -e

TEST_TYPE=$1
TIME_CONTROL=$2
ENGINE1=$3
ENGINE2=$4
THREAD_CONFIG=${5:-1t}

if [ -z "$TEST_TYPE" ] || [ -z "$TIME_CONTROL" ] || [ -z "$ENGINE1" ] || [ -z "$ENGINE2" ]; then
    echo "Usage: $0 <test_type> <time_control> <engine1> <engine2> [thread_config]"
    echo "  test_type: sprt_gain, sprt_equal, elo_check"
    echo "  time_control: stc, ltc, nodes25k"
    echo "  engine1: princhess, princhess-main, etc"
    echo "  engine2: princhess-main, stockfish, etc"
    echo "  thread_config: 1t, 2t, 4t, etc (default: 1t)"
    exit 1
fi

# Parse and validate thread config (must be positive integer followed by 't')
if ! [[ "$THREAD_CONFIG" =~ ^[1-9][0-9]*t$ ]]; then
    echo "Invalid thread config: $THREAD_CONFIG (must be positive integer followed by 't', like '2t', '4t', etc)"
    exit 1
fi

# Extract thread count (e.g., "2t" -> 2)
THREADS=$(echo "$THREAD_CONFIG" | sed 's/t$//')

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
CONCURRENCY=$((6 / THREADS))
if [ $CONCURRENCY -lt 1 ]; then
    CONCURRENCY=1
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
echo "-engine cmd=/engines/$ENGINE1 name=$ENGINE1"
echo "-engine cmd=/engines/$ENGINE2 name=$ENGINE2"
echo ""
echo "-each proto=uci tc=$TC"
echo "      option.SyzygyPath=/syzygy option.Hash=$HASH option.Threads=$THREADS"

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
        # No SPRT for elo check
        ;;
esac

echo "-openings file=/books/$OPENING_BOOK format=epd order=random"
echo "-games 2 -repeat -rounds $ROUNDS"
echo "-recover -ratinginterval 10 -concurrency $CONCURRENCY"
echo "-config outname=/state/current.json"
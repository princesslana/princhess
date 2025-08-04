#!/bin/bash

# Generate fastchess command line arguments
# Usage: generate-args.sh <test_type> <time_control> <engine1> <engine2>

set -e

TEST_TYPE=$1
TIME_CONTROL=$2
ENGINE1=$3
ENGINE2=$4

if [ -z "$TEST_TYPE" ] || [ -z "$TIME_CONTROL" ] || [ -z "$ENGINE1" ] || [ -z "$ENGINE2" ]; then
    echo "Usage: $0 <test_type> <time_control> <engine1> <engine2>"
    echo "  test_type: sprt_gain, sprt_equal, elo_check"
    echo "  time_control: stc, ltc, nodes25k"
    echo "  engine1: princhess, princhess-main, etc"
    echo "  engine2: princhess-main, stockfish, etc"
    exit 1
fi

# Determine if this is a "long" test (needs adjudication and relaxed SPRT)
is_long_test() {
    case $TIME_CONTROL in
        ltc) return 0 ;;  # Long time control
        *) return 1 ;;
    esac
}

# Set common values for all time controls
HASH=128
THREADS=1
CONCURRENCY=6
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
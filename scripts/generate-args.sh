#!/bin/bash

# Generate fastchess command line arguments

set -e

show_help() {
    cat << EOF
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

Examples:
  $0 --test-type sprt_gain --tc stc --engine1 princhess --engine2 princhess-main
  $0 --test-type elo_check --tc ltc --engine1 princhess --engine2 princhess-main --threads 2
  $0 --test-type debug --tc nodes25k --engine1 princhess --engine2 princhess-main --max-cores 4
EOF
    exit 0
}

# Defaults
THREADS=1
USE_SYZYGY=true
MAX_CORES=""
TEST_TYPE=""
TIME_CONTROL=""
ENGINE1=""
ENGINE2=""
NATIVE_MODE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --test-type)
            TEST_TYPE="$2"
            shift 2
            ;;
        --tc)
            TIME_CONTROL="$2"
            shift 2
            ;;
        --engine1)
            ENGINE1="$2"
            shift 2
            ;;
        --engine2)
            ENGINE2="$2"
            shift 2
            ;;
        --threads)
            THREADS="$2"
            shift 2
            ;;
        --syzygy)
            USE_SYZYGY="$2"
            shift 2
            ;;
        --max-cores)
            MAX_CORES="$2"
            shift 2
            ;;
        --native)
            NATIVE_MODE=true
            shift
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run '$0 --help' for usage"
            exit 1
            ;;
    esac
done

# Validate required parameters
if [ -z "$TEST_TYPE" ] || [ -z "$TIME_CONTROL" ] || [ -z "$ENGINE1" ] || [ -z "$ENGINE2" ]; then
    echo "Error: Missing required arguments"
    echo "Run '$0 --help' for usage"
    exit 1
fi

# Validate threads
if ! [[ "$THREADS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid threads: $THREADS (must be positive integer)"
    exit 1
fi

# Validate syzygy
if [ "$USE_SYZYGY" != "true" ] && [ "$USE_SYZYGY" != "false" ]; then
    echo "Invalid syzygy parameter: $USE_SYZYGY (must be 'true' or 'false')"
    exit 1
fi

# Validate max-cores if provided
if [ -n "$MAX_CORES" ] && ! [[ "$MAX_CORES" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid max-cores: $MAX_CORES (must be positive integer)"
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

# Get default concurrency: physical cores capped at (total - 2), minimum 1
get_default_concurrency() {
    local cores total

    # Use explicit max-cores override if provided
    if [ -n "$MAX_CORES" ]; then
        echo "$MAX_CORES"
        return
    fi

    # macOS: try P-cores, fall back to all physical cores
    cores=$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null)
    if [ -z "$cores" ] || [ "$cores" -eq 0 ]; then
        cores=$(sysctl -n hw.physicalcpu 2>/dev/null)
    fi
    # Linux: try /sys first (works on minimal containers), fall back to lscpu
    if [ -z "$cores" ] || [ "$cores" -eq 0 ]; then
        if [ -d "/sys/devices/system/cpu/cpu0/topology" ]; then
            cores=$(cat /sys/devices/system/cpu/cpu*/topology/core_id 2>/dev/null | sort -u | wc -l)
        fi
    fi
    if [ -z "$cores" ] || [ "$cores" -eq 0 ]; then
        cores=$(lscpu -p=Core 2>/dev/null | grep -v '^#' | sort -u | wc -l)
    fi
    # Fall back to nproc
    if [ -z "$cores" ] || [ "$cores" -eq 0 ]; then
        cores=$(nproc --all 2>/dev/null || echo 1)
    fi

    # Get total logical CPUs
    total=$(sysctl -n hw.logicalcpu 2>/dev/null)
    if [ -z "$total" ] || [ "$total" -eq 0 ]; then
        total=$(nproc --all 2>/dev/null || echo "$cores")
    fi

    # Take min(physical_cores - 1, total_cores - 2), minimum 1
    local physical_max=$((cores - 1))
    local total_max=$((total - 2))

    # Take the smaller of the two maximums
    local result=$physical_max
    if [ $total_max -lt $result ]; then
        result=$total_max
    fi

    # Ensure minimum of 1
    if [ $result -lt 1 ]; then
        result=1
    fi

    echo $result
}

# Set common values for all time controls
HASH=128
# Calculate concurrency: default_cores / threads_per_game, minimum 1
if [ "$TEST_TYPE" = "debug" ]; then
    CONCURRENCY=1
else
    DEFAULT_CORES=$(get_default_concurrency)
    CONCURRENCY=$((DEFAULT_CORES / THREADS))
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
# Set path prefixes based on native mode
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

if [ "$NATIVE_MODE" = true ]; then
    ENGINES_DIR="$PROJECT_ROOT/target/release"
    BUILDS_DIR="$PROJECT_ROOT/builds"
    SYZYGY_PATH="$PROJECT_ROOT/syzygy"
    BOOKS_DIR="$PROJECT_ROOT/books"
    STATE_DIR="$PROJECT_ROOT/target/fastchess"
    PGN_DIR="$PROJECT_ROOT/target/fastchess/pgn"
else
    ENGINES_DIR="/engines"
    BUILDS_DIR="/engines/builds"
    SYZYGY_PATH="/syzygy"
    BOOKS_DIR="/books"
    STATE_DIR="/state"
    PGN_DIR="/pgn"
fi

# Function to find engine path
get_engine_path() {
    local engine=$1
    if [ -f "$PROJECT_ROOT/builds/$engine" ]; then
        echo "$BUILDS_DIR/$engine"
    else
        echo "$ENGINES_DIR/$engine"
    fi
}

echo "-engine cmd=$(get_engine_path $ENGINE1) name=$ENGINE1"
echo "-engine cmd=$(get_engine_path $ENGINE2) name=$ENGINE2"
echo ""
echo "-each proto=uci tc=$TC"
if [ "$USE_SYZYGY" = "true" ]; then
    echo "      option.SyzygyPath=$SYZYGY_PATH option.Hash=$HASH option.Threads=$THREADS"
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
        echo "-log file=$PGN_DIR/debug.log engine=true"
        echo "-pgnout $PGN_DIR/debug.pgn"
        ;;
    *)
        echo "Unknown test type: $TEST_TYPE"
        exit 1
        ;;
esac

echo "-openings file=$BOOKS_DIR/$OPENING_BOOK format=epd order=random"
echo "-games 2 -repeat -rounds $ROUNDS"
echo "-ratinginterval 10 -concurrency $CONCURRENCY"

if [ "$TEST_TYPE" != "debug" ]; then
    echo "-recover"
fi

echo "-config outname=$STATE_DIR/current.json"

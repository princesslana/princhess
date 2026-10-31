#!/bin/bash

# Sanity check two engines before a test: princhess-main must be built from origin/main,
# and both must share compiler and target-cpu.
# Usage: check-engines.sh <engine1> <engine2> [--force]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

ENGINE1=$1
ENGINE2=$2
FORCE=false
[ "$3" = "--force" ] && FORCE=true

ERRORS=0

error() {
    echo "ERROR: $1"
    ERRORS=$((ERRORS + 1))
}

get_engine_path() {
    local engine=$1
    if [ -f "$PROJECT_ROOT/builds/$engine" ]; then
        echo "$PROJECT_ROOT/builds/$engine"
    else
        echo "$PROJECT_ROOT/target/release/$engine"
    fi
}

fingerprint_field() {
    local engine_path=$1 field=$2
    "$engine_path" fingerprint 2>/dev/null | sed -n "s/^info string $field //p"
}

# Resolves a `git describe --tags --dirty --always` string to a full commit hash.
describe_to_commit() {
    local ref=${1%-dirty}
    [[ $ref =~ -g([0-9a-f]+)$ ]] && ref=${BASH_REMATCH[1]}
    git -C "$PROJECT_ROOT" rev-parse --verify -q "$ref^{commit}"
}

require_known() {
    local engine=$1 field=$2 value=$3
    if [ -z "$value" ] || [ "$value" = unknown ]; then
        error "$engine has no $field in its fingerprint"
    fi
}

check_princhess_main() {
    local git=$1
    local commit
    commit=$(describe_to_commit "$git") || true
    if [ "$commit" != "$(git -C "$PROJECT_ROOT" rev-parse origin/main)" ]; then
        error "princhess-main is built from $git, not origin/main ($(git -C "$PROJECT_ROOT" rev-parse --short origin/main))"
    elif [[ $git == *-dirty ]]; then
        error "princhess-main is built from uncommitted changes ($git)"
    fi
}

if ! git -C "$PROJECT_ROOT" fetch -q origin main; then
    echo "WARNING: could not fetch origin/main, using last fetched"
fi

declare -A GIT RUSTC CPU
for engine in "$ENGINE1" "$ENGINE2"; do
    path=$(get_engine_path "$engine")
    if [ ! -x "$path" ]; then
        error "$engine not found at $path"
        continue
    fi
    GIT[$engine]=$(fingerprint_field "$path" git)
    RUSTC[$engine]=$(fingerprint_field "$path" rustc)
    CPU[$engine]=$(fingerprint_field "$path" target-cpu)
    echo "$engine: git ${GIT[$engine]:-unknown}, ${RUSTC[$engine]:-rustc unknown}, target-cpu ${CPU[$engine]:-unknown}"

    require_known "$engine" rustc "${RUSTC[$engine]}"
    require_known "$engine" target-cpu "${CPU[$engine]}"

    if [ "$engine" = princhess-main ]; then
        check_princhess_main "${GIT[$engine]}"
    fi
done

if [ "$ERRORS" -eq 0 ]; then
    [ "${RUSTC[$ENGINE1]}" != "${RUSTC[$ENGINE2]}" ] && error "rustc differs between $ENGINE1 and $ENGINE2"
    [ "${CPU[$ENGINE1]}" != "${CPU[$ENGINE2]}" ] && error "target-cpu differs between $ENGINE1 and $ENGINE2"
fi

if [ "$ERRORS" -gt 0 ]; then
    if [ "$FORCE" = true ]; then
        echo "Continuing anyway (FORCE=1)"
    else
        echo "Engine check failed, use FORCE=1 to run anyway"
        exit 1
    fi
fi

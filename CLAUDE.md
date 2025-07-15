# CLAUDE.md - Working with Princhess

## Project Overview
Princhess is a CPU-optimized chess engine written in Rust using Monte Carlo Tree Search (MCTS). It implements the UCI protocol and uses neural networks for position evaluation.

## Project Goals
**Primary Goal**: Achieve the highest possible ELO in tournament play using CPU-only MCTS

**Core Constraints**:
- CPU-only (no GPU acceleration)
- MCTS search algorithm
- UCI compatibility (required for tournament participation)

**Success Metric**: Tournament ELO rating

**Development Philosophy**: Any improvement that increases ELO is valuable, as long as it maintains the CPU/MCTS approach that defines the project's unique position.

## Key Architecture
- `src/main.rs` - UCI entry point
- `src/engine.rs` - Main engine coordination
- `src/mcts.rs` - Monte Carlo Tree Search implementation
- `src/chess/` - Chess game logic (bitboards, moves, etc.)
- `src/uci.rs` - UCI protocol implementation
- `crates/princhess-train/` - Training infrastructure

## Development Commands
```bash
cargo build --release    # Standard build
make native              # CPU-optimized build
cargo fmt               # Format code (always run after changes)
cargo clippy            # Lint code
```

## Code Standards
- Follow Rust conventions (snake_case, PascalCase, SCREAMING_SNAKE_CASE)
- Self-documenting code preferred, minimal comments
- KISS, YAGNI, DRY principles
- Minimal unsafe blocks with explanatory comments

## Strategic Comments
- **High-impact only**: Comments should clarify complex algorithms or non-obvious architectural decisions
- **Algorithm understanding**: Brief comments on core concepts (e.g., "PUCT selection", "tree traversal phase")
- **Avoid over-commenting**: Don't explain what the code does, explain why or what algorithm step it represents
- **Preserve readability**: Comments should enhance, not clutter the code

## Commit Messages (Emoji Log)
- 📦 NEW: Add entirely new features
- 👌 IMP: Improve/enhance existing code
- 🐛 FIX: Fix bugs
- 📖 DOC: Documentation updates
- 🤖 TST: Test additions/updates
- Keep messages under 50 characters, imperative mood
- **SINGLE LINE ONLY** - no multi-line commit messages

## Testing
- `./target/release/princhess` - Run UCI engine (interactive mode)
- `./target/release/princhess "uci"` - Run UCI commands and exit
- `./target/release/princhess "uci" "isready" "position startpos" "go nodes 5000"` - Chain commands
- `cargo test --release` - Run unit tests
- Built-in `bench` command for performance testing
- Debug commands: `movelist`, `eval`, `sizelist`
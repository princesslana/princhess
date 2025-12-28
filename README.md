# Princhess

Princhess is a chess engine written in Rust. It implements a Monte Carlo Tree Search (MCTS) algorithm designed to run on the CPU.

## Features

*   **CPU-Only MCTS:** Implements Monte Carlo Tree Search for CPU execution.
*   **UCI Protocol Support:** Compatible with the Universal Chess Interface (UCI).
*   **Bitboard Representation:** Uses bitboards for board representation and move generation.
*   **Multi-threading:** Supports multi-threaded operation.
*   **DAG-Based MCTS:** Uses a directed acyclic graph rather than a tree to support transpositions. Rewards and visit counts are stored on edges instead of nodes.
*   **Dynamic CPUCT Adjustments:** Multiple mechanisms for adjusting exploration including Gini impurity, per-thread jitter for search diversity, and trend-based adjustments for winning/losing positions.
*   **Custom Memory Allocator:** Includes an arena memory allocator with half flipping for managing the MCTS graph within a fixed memory size, allowing unbounded search depth with bounded memory usage.
*   **Phase-Aware Policy Networks:** Dynamically loads separate policy networks for middle-game and endgame phases.
*   **Endgame Tablebase Integration:** Integrates with Fathom tablebases for endgame evaluation.
*   **Self-Play & Training Infrastructure:** The `princhess-train` crate provides tools for self-play data generation. The policy and value networks are trained on this self-play data and are designed to be small enough to run efficiently on the CPU.

## Development Status

Princhess is under development. It is designed to be stable and can be used with UCI-compatible chess interfaces.

## Building

### Build Instructions

Clone the repository and build the project:

```bash
git clone https://github.com/princesslana/princhess.git
cd princhess
cargo build --release
```

The `build.rs` script compiles the vendored Fathom C library and generates Rust bindings.

### Pre-built Binaries

Pre-built binaries for released versions are available on the [GitHub Releases page](https://github.com/princesslana/princhess/releases).

## Usage

Princhess is a UCI-compatible chess engine. Run the executable and connect it to a UCI-compatible interface.

```bash
./target/release/princhess # Or the path to your downloaded binary
```

The engine responds to standard UCI commands.

## UCI Options

### Basic Engine Options
- **Hash** (default: 128): Memory in MB allowed for storage of the MCTS search tree
- **Threads** (default: 1): Number of search threads
- **MultiPV** (default: 1): Number of best moves to display in analysis output
- **SyzygyPath** (default: \<empty>): Path to Syzygy endgame tablebase files

### MCTS Search Parameters
- **CPuct** (default: 16): PUCT exploration constant
- **CPuctTau** (default: 84): Tau from the generalized PUCT formula
- **CPuctJitter** (default: 5): Random jitter applied to CPuct for helper threads to increase search diversity
- **CPuctTrendAdjustment** (default: 15): Dynamic CPUCT adjustment based on position trend (winning/losing)
- **CPuctGiniBase** (default: 68): Base value for Gini impurity scaling of exploration
- **CPuctGiniFactor** (default: 163): Logarithmic factor for Gini impurity scaling
- **CPuctGiniMax** (default: 210): Maximum value for Gini-scaled exploration coefficient
- **CVisitsSelection** (default: 1): How much to consider visits vs Q-value in final move selection
- **PolicyTemperature** (default: 100): Softmax temperature for policy network during node expansion
- **PolicyTemperatureRoot** (default: 1450): Separate policy temperature specifically for root node

### Time Management
Adaptive time allocation controls:
- **TMMinM** (default: 10): Minimum time multiplier
- **TMMaxM** (default: 500): Maximum time multiplier
- **TMVisitsBase** (default: 140): Base visits factor for time scaling based on best move visit count
- **TMVisitsM** (default: 139): Visits multiplier for time scaling based on best move visit count
- **TMPvDiffC** (default: 0): PV difference threshold for time scaling based on eval stability
- **TMPvDiffM** (default: 121): PV difference multiplier for time scaling based on eval stability

### Special Modes
- **UCI_Chess960** (default: false): Enable Chess960
- **PolicyOnly** (default: false): Skip search entirely, use only policy network for move selection
- **UCI_ShowMovesLeft** (default: false): Display estimated moves remaining in analysis
- **UCI_ShowWDL** (default: false): Show Win/Draw/Loss probabilities in analysis output

## Contributing

Contributions are welcome! Please follow Rust conventions and ensure code is properly formatted with `cargo fmt` and passes `cargo clippy`.

Commit messages use [Emoji Log](https://github.com/ahmadawais/Emoji-Log) format.

## License

Princhess is licensed under the **GNU General Public License v3.0 (GPL-3.0)**. See the [LICENSE](LICENSE) file for details.

The project's initial codebase was [Sashimi](https://github.com/zxqfl/sashimi), which is licensed under the [MIT License](LICENSE.sashimi).

The Fathom tablebase library (`deps/fathom/`) is licensed under the [MIT License](LICENSE.fathom).

The neural network training components (`crates/princhess-train/src/neural/`) are based on the [Goober](https://github.com/jw1912/goober/) library by Jamie Whiting, which is licensed under the [MIT License](LICENSE.goober).

## Links

*   **GitHub Repository:** [https://github.com/princesslana/princhess](https://github.com/princesslana/princhess)
*   **Engine Programming Discord:** Princhess channel on the [Engine Programming Discord](https://discord.gg/YctB2p4).
*   **Lichess Bot:** [https://lichess.org/@/princhess_bot](https://lichess.org/@/princhess_bot) (infrequently online).
*   **Policy-Only Lichess Bot:** [https://lichess.org/@/princhess_policy_bot](https://lichess.org/@/princhess_policy_bot) (plays using only the policy network, no search, infrequently online).

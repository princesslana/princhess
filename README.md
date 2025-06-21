# Princhess

Princhess is a chess engine written in Rust. It implements a Monte Carlo Tree Search (MCTS) algorithm designed to run on the CPU.

## Features

*   **CPU-Only MCTS:** Implements Monte Carlo Tree Search for CPU execution.
*   **UCI Protocol Support:** Compatible with the Universal Chess Interface (UCI).
*   **Bitboard Representation:** Uses bitboards for board representation and move generation.
*   **Multi-threading:** Supports multi-threaded operation.
*   **Custom Memory Allocator:** Includes an arena memory allocator for managing the MCTS graph within a fixed memory size.
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

./target/release/princhess # Or the path to your downloaded binary

The engine responds to standard UCI commands.

## Contributing

Refer to [CONVENTIONS.md](CONVENTIONS.md) for code style and commit message guidelines.
[Emoji Log](https://github.com/ahmadawais/Emoji-Log) is used for commit messages.

## License

Princhess is licensed under the **GNU General Public License v3.0 (GPL-3.0)**. See the [LICENSE](LICENSE) file for details.

The project's initial codebase was [Sashimi](https://github.com/zxqfl/sashimi), which is licensed under the [MIT License](LICENSE.sashimi).

The Fathom tablebase library (`deps/fathom/`) is licensed under the [MIT License](LICENSE.fathom).

## Links

*   **GitHub Repository:** [https://github.com/princesslana/princhess](https://github.com/princesslana/princhess)
*   **Engine Programming Discord:** Princhess channel on the [Engine Programming Discord](https://discord.gg/YctB2p4).
*   **Lichess Bot:** [https://lichess.org/@/princhess_bot](https://lichess.org/@/princhess_bot) (infrequently online).

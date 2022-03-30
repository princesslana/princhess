# Princhess

[![Discord](https://img.shields.io/discord/417389758470422538)](https://discord.gg/3aTVQtz)

Princhess is a UCI compatible chess engine written in Rust.

The initial code base was [Sashimi](https://github.com/zxqfl/sashimi)

# Play!

Princhess can be played against on Lichess at https://lichess.org/@/princhess_bot

# UCI Options

* **Threads** - The number of threads used during search. Defaults to 1.

* **Hash** - The amount of hash space to use in MB. Default 16

* **SyzygyPath** - Path to folder where the Syzygy tablebase files are.
  Currently only supports a single folder.

* **CPuct** - Exploration constant used by PUCT. Defaults to 2.15

* **CPuctBase** - Base for PUCT growth formula. Defaults to 18368.

* **CPuctFactor** - Multipler for the PUCT growth formula. Defaults to 2.82

* **MateScore** - Score used for checkmate. Defaults to 1.1

# Contributing

Look for the princhess channel on [Discord Projects Hub](https://discord.gg/3aTVQtz)

[Emoji Log](https://github.com/ahmadawais/Emoji-Log) is used for commit messages and pull requests.

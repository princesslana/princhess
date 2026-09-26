//! typed uci commands borrowing slices from the input line.

use std::time::Duration;

/// startpos or a raw fen slice from the incoming line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PositionSpec<'a> {
    /// the standard initial position.
    Startpos,
    /// fen fields sliced straight out of the input line.
    Fen(&'a str),
}

/// parsed setoption fields.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetOptionParams<'a> {
    /// option name, may contain spaces.
    pub name: &'a str,
    /// None for button-type options, which take no value.
    pub value: Option<&'a str>,
}

/// parsed go params.
///
/// depth/mate/ponder are kept so guis don't freak out; mcts only uses time and nodes.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct GoParams {
    /// search until stopped.
    pub infinite: bool,
    /// fixed time per move.
    pub movetime: Option<Duration>,
    /// white clock time remaining.
    pub wtime: Option<Duration>,
    /// black clock time remaining.
    pub btime: Option<Duration>,
    /// white increment per move.
    pub winc: Option<Duration>,
    /// black increment per move.
    pub binc: Option<Duration>,
    /// moves until next time control.
    pub movestogo: Option<u32>,
    /// node budget.
    pub nodes: Option<usize>,
    /// depth limit, unused by mcts.
    pub depth: Option<u32>,
    /// mate in n, unused by mcts.
    pub mate: Option<u32>,
    /// ponder mode, unused.
    pub ponder: bool,
}

/// a parsed uci command, borrowing string slices from the input line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UciCommand<'a> {
    /// uci handshake.
    Uci,
    /// debug flag.
    Debug(bool),
    /// isready ping.
    IsReady,
    /// setoption with parsed name/value.
    SetOption(SetOptionParams<'a>),
    /// start a new game.
    UciNewGame,
    /// set the root position and replay moves.
    Position {
        /// startpos or fen.
        spec: PositionSpec<'a>,
        /// everything after the moves keyword, borrowed from the input line.
        /// kept as a slice instead of a Vec so a hostile movelist can't force
        /// an allocation; split on whitespace downstream and resolved against
        /// legal moves where the position is known, since a bare uci string
        /// can't become a Move alone.
        moves: &'a str,
    },
    /// start searching.
    Go(GoParams),
    /// stop searching.
    Stop,
    /// opponent moved while pondering.
    PonderHit,
    /// quit the engine.
    Quit,
    /// borrowed movelist remainder, matched against the search tree.
    MoveList(&'a str),
    /// print search graph sizes.
    SizeList,
    /// print static eval of root.
    Eval,
    /// run the benchmark.
    Bench,
    /// generate a random opening.
    RandomOpen,
    /// print build metadata and net hashes.
    Fingerprint,
    /// first word of an unrecognized line.
    UnknownCommand(&'a str),
}

use std::time::Duration;

// startpos or raw fen slice from incoming line
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PositionSpec<'a> {
    Startpos,
    Fen(&'a str),
}

// parsed setoption fields, value is None for button types like clear hash
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetOptionParams<'a> {
    pub name: &'a str,
    pub value: Option<&'a str>,
}

// parsed params for go cmd, keep depth/mate/ponder so guis don't freak out but mcts only cares bout time and nodes
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct GoParams {
    pub infinite: bool,
    pub movetime: Option<Duration>,
    pub wtime: Option<Duration>,
    pub btime: Option<Duration>,
    pub winc: Option<Duration>,
    pub binc: Option<Duration>,
    pub movestogo: Option<u32>,
    pub nodes: Option<usize>,
    pub depth: Option<u32>,
    pub mate: Option<u32>,
    pub ponder: bool,
}

// all uci commands we care about, borrowing string slices directly from the input buffer
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UciCommand<'a> {
    Uci,
    Debug(bool),
    IsReady,
    SetOption(SetOptionParams<'a>),
    UciNewGame,
    Position {
        spec: PositionSpec<'a>,
        moves: Vec<&'a str>,
    },
    Go(GoParams),
    Stop,
    PonderHit,
    Quit,
    MoveList(Vec<&'a str>),
    SizeList,
    Eval,
    Bench,
    RandomOpen,
    Fingerprint,
    UnknownCommand(&'a str),
}

#[cfg(feature = "fathom")]
mod fathom;
mod probe;

// Without Fathom no tables load, so `max_pieces` is 0 and no probe reaches a table read.
#[cfg(not(feature = "fathom"))]
mod fathom {
    use crate::chess::Board;
    use crate::tablebase::Outcome;

    #[allow(dead_code)]
    pub enum DtzEntry {
        Value(i32),
        OtherSide,
    }

    pub fn init(_paths: &str) -> bool {
        false
    }

    pub fn max_pieces() -> usize {
        0
    }

    pub fn probe_wdl_table(_b: &Board) -> Option<Outcome> {
        None
    }

    pub fn probe_dtz_table(_b: &Board, _outcome: Outcome) -> Option<DtzEntry> {
        None
    }
}

use std::ops::Neg;

use crate::chess::{Board, Move};

pub enum Wdl {
    Win,
    Draw,
    Loss,
}

pub struct RootProbe {
    /// The legal moves that keep the best result available.
    pub moves: Vec<Move>,
    pub score: i64,
}

/// A position's result from the side to move. Cursed wins and blessed losses are wins and
/// losses that the 50-move rule turns into draws.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Outcome {
    Loss,
    BlessedLoss,
    Draw,
    CursedWin,
    Win,
}

impl Neg for Outcome {
    type Output = Self;

    fn neg(self) -> Self {
        match self {
            Self::Loss => Self::Win,
            Self::BlessedLoss => Self::CursedWin,
            Self::Draw => Self::Draw,
            Self::CursedWin => Self::BlessedLoss,
            Self::Win => Self::Loss,
        }
    }
}

pub fn set_tablebase_directory(paths: &str) -> bool {
    fathom::init(paths)
}

/// Ignores the 50-move rule: cursed wins and blessed losses are draws.
pub fn probe_wdl(b: &Board) -> Option<Wdl> {
    if !is_probeable(b) {
        return None;
    }

    match probe::probe_wdl(b)? {
        Outcome::Win => Some(Wdl::Win),
        Outcome::CursedWin | Outcome::Draw | Outcome::BlessedLoss => Some(Wdl::Draw),
        Outcome::Loss => Some(Wdl::Loss),
    }
}

/// `None` when the position is outside the tables or has no legal moves.
pub fn probe_root(b: &Board, has_repeated: bool) -> Option<RootProbe> {
    if !is_probeable(b) {
        return None;
    }

    probe::probe_root(b, has_repeated)
}

fn is_probeable(b: &Board) -> bool {
    b.occupied().count() <= fathom::max_pieces() && !b.is_castling_rights()
}

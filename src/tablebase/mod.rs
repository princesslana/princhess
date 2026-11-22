mod compression;
mod encoding;
mod probe;
mod reader;
mod table;
mod tables;

use std::ops::Neg;

use crate::chess::{Board, Move};

pub use crate::tablebase::tables::Tablebases;

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
pub enum Outcome {
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

impl Tablebases {
    /// Applies the 50-move rule from a fresh counter: cursed wins and blessed losses are draws.
    pub fn probe_wdl(&self, b: &Board) -> Option<Wdl> {
        if !self.is_probeable(b) {
            return None;
        }

        match self.probe_outcome(b)? {
            Outcome::Win => Some(Wdl::Win),
            Outcome::CursedWin | Outcome::Draw | Outcome::BlessedLoss => Some(Wdl::Draw),
            Outcome::Loss => Some(Wdl::Loss),
        }
    }

    /// `None` when the position is outside the tables or has no legal moves.
    pub fn probe_root(&self, b: &Board, has_repeated: bool) -> Option<RootProbe> {
        if !self.is_probeable(b) {
            return None;
        }

        self.rank_root(b, has_repeated)
    }

    fn is_probeable(&self, b: &Board) -> bool {
        b.occupied().count() <= self.max_pieces() && !b.has_castling_rights()
    }
}

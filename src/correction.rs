use std::sync::atomic::{AtomicI16, Ordering};

use crate::chess::Color;
use crate::quantized_value::QAB;
use crate::state::State;

const SIZE: usize = 65536;
const MASK: usize = SIZE - 1;

const MAX_CORRECTION: i32 = QAB / 10;
const RATE_DIVISOR: i32 = 64;

/// Learned eval corrections keyed by features of the position, shared by all search threads.
///
/// Entries are value-net logits (pre-tanh) quantized by `QAB`, from white's point of view. The
/// correction is the sum of the tables, which learn from a shared residual.
pub struct CorrectionTable {
    pawn: Box<[AtomicI16]>,
    non_pawn: [Box<[AtomicI16]>; Color::COUNT],
}

impl CorrectionTable {
    #[must_use]
    pub fn new() -> Self {
        Self {
            pawn: Self::table(),
            non_pawn: [Self::table(), Self::table()],
        }
    }

    fn table() -> Box<[AtomicI16]> {
        (0..SIZE).map(|_| AtomicI16::new(0)).collect()
    }

    fn entries(&self, state: &State) -> [&AtomicI16; 3] {
        let board = state.board();
        #[allow(clippy::cast_possible_truncation)]
        let index = |hash: u64| hash as usize & MASK;

        [
            &self.pawn[index(board.pawn_hash())],
            &self.non_pawn[Color::WHITE][index(board.non_pawn_hash(Color::WHITE))],
            &self.non_pawn[Color::BLACK][index(board.non_pawn_hash(Color::BLACK))],
        ]
    }

    fn sign(state: &State) -> i32 {
        state.side_to_move().fold(1, -1)
    }

    fn sum(entries: &[&AtomicI16]) -> i32 {
        entries
            .iter()
            .map(|e| i32::from(e.load(Ordering::Relaxed)))
            .sum()
    }

    /// The correction for this position, from the side to move.
    #[must_use]
    pub fn correction(&self, state: &State) -> i32 {
        let sum = Self::sum(&self.entries(state)).clamp(-MAX_CORRECTION, MAX_CORRECTION);
        Self::sign(state) * sum
    }

    /// Moves the correction towards `diff`, the error of the corrected eval from the side to move,
    /// faster with more visits behind it.
    pub fn update(&self, state: &State, diff: i32, visits: u32) {
        let entries = self.entries(state);
        let target = (Self::sign(state) * diff).clamp(-MAX_CORRECTION, MAX_CORRECTION);
        let residual = target - Self::sum(&entries);
        #[allow(clippy::cast_possible_wrap)]
        let weight = visits.ilog2() as i32;
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let divisor = RATE_DIVISOR * entries.len() as i32;

        for entry in entries {
            let current = i32::from(entry.load(Ordering::Relaxed));
            let updated =
                (current + weight * residual / divisor).clamp(-MAX_CORRECTION, MAX_CORRECTION);

            #[allow(clippy::cast_possible_truncation)]
            entry.store(updated as i16, Ordering::Relaxed);
        }
    }
}

impl Default for CorrectionTable {
    fn default() -> Self {
        Self::new()
    }
}

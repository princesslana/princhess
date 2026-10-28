use std::sync::atomic::{AtomicI16, Ordering};

use crate::quantized_value::QAB;
use crate::state::State;

const SIZE: usize = 65536;
const MASK: usize = SIZE - 1;

const MAX_CORRECTION: i32 = QAB / 10;
const RATE_DIVISOR: i32 = 64;

/// Learned eval corrections keyed by pawn structure, shared by all search threads.
///
/// Entries are value-net logits (pre-tanh) quantized by `QAB`, from white's point of view.
pub struct CorrectionTable {
    pawn: Box<[AtomicI16]>,
}

impl CorrectionTable {
    #[must_use]
    pub fn new() -> Self {
        Self {
            pawn: (0..SIZE).map(|_| AtomicI16::new(0)).collect(),
        }
    }

    fn pawn_entry(&self, state: &State) -> &AtomicI16 {
        #[allow(clippy::cast_possible_truncation)]
        let idx = state.board().pawn_hash() as usize & MASK;
        &self.pawn[idx]
    }

    fn sign(state: &State) -> i32 {
        state.side_to_move().fold(1, -1)
    }

    /// The correction for this position, from the side to move.
    #[must_use]
    pub fn correction(&self, state: &State) -> i32 {
        Self::sign(state) * i32::from(self.pawn_entry(state).load(Ordering::Relaxed))
    }

    /// Moves the correction towards `diff`, the error of the corrected eval from the side to move,
    /// faster with more visits behind it.
    pub fn update(&self, state: &State, diff: i32, visits: u32) {
        let entry = self.pawn_entry(state);
        let current = i32::from(entry.load(Ordering::Relaxed));
        let target = (Self::sign(state) * diff).clamp(-MAX_CORRECTION, MAX_CORRECTION);
        #[allow(clippy::cast_possible_wrap)]
        let weight = visits.ilog2() as i32;
        let updated = current + weight * (target - current) / RATE_DIVISOR;

        #[allow(clippy::cast_possible_truncation)]
        entry.store(updated as i16, Ordering::Relaxed);
    }
}

impl Default for CorrectionTable {
    fn default() -> Self {
        Self::new()
    }
}

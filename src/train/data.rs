use arrayvec::ArrayVec;
use bytemuck::{self, Pod, Zeroable};
use goober::SparseVector;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::mem;

use crate::chess::{Bitboard, Board, Castling, Color, Move, Piece, Square};
use crate::search::SCALE;
use crate::search_tree::SearchTree;
use crate::state::State;

#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct TrainingPosition {
    occupied: Bitboard,
    pieces: [u8; 16],

    evaluation: i32,
    result: i8,
    stm: u8,

    legal_moves: [Move; TrainingPosition::MAX_MOVES],
    visits: [u8; TrainingPosition::MAX_MOVES],
}

const _SIZE_CHECK: () = assert!(mem::size_of::<TrainingPosition>() == 192);

impl TrainingPosition {
    pub const MAX_MOVES: usize = 54;

    pub const SIZE: usize = mem::size_of::<Self>();
    pub const BATCH_SIZE: usize = 16384;
    pub const BUFFER_COUNT: usize = 1 << 16;
    pub const BUFFER_SIZE: usize = Self::BUFFER_COUNT * Self::SIZE;

    pub fn write_buffer(out: &mut BufWriter<File>, data: &[TrainingPosition; Self::BUFFER_COUNT]) {
        out.write_all(bytemuck::bytes_of(data)).unwrap();
    }

    #[must_use]
    pub fn read_buffer(buffer: &[u8]) -> &[TrainingPosition; Self::BUFFER_COUNT] {
        bytemuck::from_bytes(buffer)
    }

    pub fn read_buffer_mut(buffer: &mut [u8]) -> &mut [TrainingPosition; Self::BUFFER_COUNT] {
        bytemuck::from_bytes_mut(buffer)
    }

    #[must_use]
    pub fn evaluation(&self) -> f32 {
        self.evaluation as f32 / SCALE
    }

    pub fn stm(&self) -> Color {
        Color::from(self.stm)
    }

    #[must_use]
    pub fn stm_relative_evaluation(&self) -> f32 {
        let e = self.evaluation();
        self.stm().fold(e, -e)
    }

    #[must_use]
    pub fn stm_relative_result(&self) -> i8 {
        self.stm().fold(self.result, -self.result)
    }

    #[must_use]
    pub fn moves(&self) -> ArrayVec<(Move, u8), { Self::MAX_MOVES }> {
        self.legal_moves
            .iter()
            .zip(self.visits.iter())
            .take_while(|(m, _)| **m != Move::NONE)
            .map(|(m, v)| (*m, *v))
            .collect()
    }

    pub fn set_result(&mut self, result: i8) {
        self.result = result;
    }

    #[must_use]
    pub fn get_value_features(&self) -> SparseVector {
        let mut features = SparseVector::with_capacity(64);
        let state = State::from(self);

        state.value_features_map(|idx| features.push(idx));

        features
    }

    #[must_use]
    pub fn get_policy_features(&self) -> SparseVector {
        let mut features = SparseVector::with_capacity(64);
        let state = State::from(self);

        state.policy_features_map(|idx| features.push(idx));

        features
    }
}

impl From<&SearchTree> for TrainingPosition {
    fn from(tree: &SearchTree) -> Self {
        let board = tree.root_state().board();

        let occupied = board.occupied();
        let stm = board.side_to_move();

        let mut pieces = [0; 16];

        for (idx, sq) in occupied.into_iter().enumerate() {
            let color = u8::from(board.color_at(sq)) << 3;
            let piece = board.piece_at(sq);

            let pc = color | u8::from(piece);

            pieces[idx / 2] |= pc << (4 * (idx & 1));
        }

        let mut nodes = [(Move::NONE, 0); Self::MAX_MOVES];
        let mut max_visits = 0;

        for (node, hot) in nodes.iter_mut().zip(tree.root_node().hots().iter()) {
            let vs = hot.visits();
            max_visits = max_visits.max(vs);
            *node = (*hot.get_move(), vs);
        }

        let mut legal_moves = [Move::NONE; Self::MAX_MOVES];
        let mut visits = [0; Self::MAX_MOVES];

        for (idx, (mv, vs)) in nodes.iter().enumerate() {
            legal_moves[idx] = *mv;

            let scaled_visits = (*vs * u32::from(u8::MAX)).div_ceil(max_visits.max(1));
            assert!(u8::try_from(scaled_visits).is_ok());
            visits[idx] = scaled_visits as u8;
        }

        let pv = tree.best_edge();
        let mut evaluation = pv.reward().average;

        // white relative evaluation
        evaluation = stm
            .fold(evaluation, -evaluation)
            .clamp(-SCALE as i32, SCALE as i32);

        // zero'd to be filled in later
        let result = 0;

        TrainingPosition {
            occupied,
            pieces,
            legal_moves,
            visits,
            evaluation,
            result,
            stm: u8::from(stm),
        }
    }
}

impl From<&TrainingPosition> for State {
    fn from(position: &TrainingPosition) -> Self {
        let mut colors = [Bitboard::EMPTY; 2];
        let mut pieces = [Bitboard::EMPTY; 6];

        for (idx, sq) in position.occupied.into_iter().enumerate() {
            let packed_piece = position.pieces[idx / 2] >> (4 * (idx & 1));
            let color = Color::from((packed_piece >> 3) & 1);
            let piece = Piece::from(packed_piece & 7);

            colors[color].toggle(sq);
            pieces[piece].toggle(sq);
        }

        let board = Board::from_bitboards(
            colors,
            pieces,
            position.stm(),
            Square::NONE,
            Castling::none(),
        );

        State::from_board(board)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::options::SearchOptions;
    use crate::search::Search;
    use crate::transposition_table::LRTable;

    const STARTPOS_NO_CASTLING: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1";
    const KIWIPETE_NO_CASTLING: &str =
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w - - 0 1";
    const KIWIPETE_NO_CASTLING_STMB: &str =
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b - - 0 1";

    #[test]
    fn test_startpos_conversion() {
        let state_before = State::from_fen(STARTPOS_NO_CASTLING);
        let search = Search::new(
            state_before.clone(),
            LRTable::empty(16),
            SearchOptions::default(),
        );

        let training_position = TrainingPosition::from(search.tree());

        let state_after = State::from(&training_position);

        assert_eq!(state_before, state_after);
    }

    #[test]
    fn test_kiwipete_conversion() {
        let state_before = State::from_fen(KIWIPETE_NO_CASTLING);
        let search = Search::new(
            state_before.clone(),
            LRTable::empty(16),
            SearchOptions::default(),
        );

        let training_position = TrainingPosition::from(search.tree());

        let state_after = State::from(&training_position);

        assert_eq!(state_before, state_after);
    }

    #[test]
    fn test_kiwipete_stm_black_conversion() {
        let state_before = State::from_fen(KIWIPETE_NO_CASTLING_STMB);
        let search = Search::new(
            state_before.clone(),
            LRTable::empty(16),
            SearchOptions::default(),
        );

        let training_position = TrainingPosition::from(search.tree());

        let state_after = State::from(&training_position);

        assert_eq!(state_before, state_after);
    }
}

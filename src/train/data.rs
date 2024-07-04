use crate::chess::{Bitboard, Board, Castling, Color, Move, Piece, Square};
use crate::search::SCALE;
use crate::search_tree::SearchTree;
use crate::state::State;

use arrayvec::ArrayVec;
use goober::SparseVector;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::{io, mem, slice};

#[derive(Clone, Copy, Debug)]
pub struct TrainingPosition {
    occupied: Bitboard,
    pieces: [u8; 16],
    stm: Color,

    result: i8,
    evaluation: i32,

    previous_moves: [Move; 4],

    #[allow(dead_code)]
    best_move: Move,

    legal_moves: [Move; TrainingPosition::MAX_MOVES],
    visits: [u8; TrainingPosition::MAX_MOVES],
}

const _SIZE_CHECK: () = assert!(mem::size_of::<TrainingPosition>() == 256);

impl TrainingPosition {
    pub const MAX_MOVES: usize = 72;
    pub const MAX_VISITS: u32 = 1024;
    pub const SIZE: usize = mem::size_of::<Self>();

    pub fn write_batch(out: &mut BufWriter<File>, data: &[TrainingPosition]) -> io::Result<()> {
        let src_size = mem::size_of_val(data);
        let data_slice = unsafe { slice::from_raw_parts(data.as_ptr().cast(), src_size) };
        out.write_all(data_slice)?;
        Ok(())
    }

    #[must_use]
    pub fn read_batch(buffer: &[u8]) -> &[TrainingPosition] {
        let len = buffer.len() / TrainingPosition::SIZE;
        unsafe { slice::from_raw_parts(buffer.as_ptr().cast(), len) }
    }

    pub fn read_batch_mut(buffer: &mut [u8]) -> &mut [TrainingPosition] {
        let len = buffer.len() / TrainingPosition::SIZE;
        unsafe { slice::from_raw_parts_mut(buffer.as_mut_ptr().cast(), len) }
    }

    #[must_use]
    pub fn evaluation(&self) -> f32 {
        self.evaluation as f32 / SCALE
    }

    #[must_use]
    pub fn stm_relative_evaluation(&self) -> f32 {
        let e = self.evaluation();
        self.stm.fold(e, -e)
    }

    #[must_use]
    pub fn stm_relative_result(&self) -> i8 {
        self.stm.fold(self.result, -self.result)
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

    pub fn set_previous_moves(&mut self, moves: [Move; 4]) {
        self.previous_moves = moves;
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

        for (node, hot) in nodes.iter_mut().zip(tree.root_node().hots().iter()) {
            *node = (*hot.get_move(), hot.visits());
        }

        let mut legal_moves = [Move::NONE; Self::MAX_MOVES];
        let mut visits = [0; Self::MAX_MOVES];

        let mut max_visits = 0;

        for (idx, (mv, vs)) in nodes
            .iter()
            .take_while(|(m, _)| *m != Move::NONE)
            .enumerate()
        {
            let vs = vs.min(&Self::MAX_VISITS);

            assert!(*vs <= Self::MAX_VISITS);
            assert!(u8::try_from(vs * u32::from(u8::MAX) / Self::MAX_VISITS).is_ok());

            if *vs > max_visits {
                max_visits = *vs;
            }

            legal_moves[idx] = *mv;
            visits[idx] = (*vs * u32::from(u8::MAX) / Self::MAX_VISITS) as u8;
        }

        assert!(max_visits == Self::MAX_VISITS);

        let pv = tree.best_edge();
        let mut evaluation = match pv.visits() {
            0 => 0,
            v => pv.sum_rewards() / i64::from(v),
        } as i32;

        let best_move = *pv.get_move();

        // white relative evaluation
        evaluation = stm
            .fold(evaluation, -evaluation)
            .clamp(-SCALE as i32, SCALE as i32);

        // zero'd to be filled in later
        let result = 0;
        let previous_moves = [Move::NONE; 4];

        TrainingPosition {
            occupied,
            pieces,
            stm,
            result,
            evaluation,
            previous_moves,
            best_move,
            legal_moves,
            visits,
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

            colors[color.index()].toggle(sq);
            pieces[piece.index()].toggle(sq);
        }

        let board =
            Board::from_bitboards(colors, pieces, position.stm, Square::NONE, Castling::none());

        State::from_board(board)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::chess::Square;
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
        let search = Search::new(state_before.clone(), LRTable::empty());

        let training_position = TrainingPosition::from(search.tree());

        let state_after = State::from(&training_position);

        assert_eq!(state_before, state_after);
    }

    #[test]
    fn test_startpos_and_moves_conversion() {
        let e2e4 = Move::new(Square::E2, Square::E4);
        let e7e6 = Move::new(Square::E7, Square::E6);

        let mut state_before = State::from_fen(STARTPOS_NO_CASTLING);
        state_before.make_move(e2e4);
        state_before.make_move(e7e6);

        let search = Search::new(state_before.clone(), LRTable::empty());

        let mut training_position = TrainingPosition::from(search.tree());
        training_position.set_previous_moves([e7e6, e2e4, Move::NONE, Move::NONE]);

        let state_after = State::from(&training_position);

        assert_eq!(state_before, state_after);
    }

    #[test]
    fn test_kiwipete_conversion() {
        let state_before = State::from_fen(KIWIPETE_NO_CASTLING);
        let search = Search::new(state_before.clone(), LRTable::empty());

        let training_position = TrainingPosition::from(search.tree());

        let state_after = State::from(&training_position);

        assert_eq!(state_before, state_after);
    }

    #[test]
    fn test_kiwipete_stm_black_conversion() {
        let state_before = State::from_fen(KIWIPETE_NO_CASTLING_STMB);
        let search = Search::new(state_before.clone(), LRTable::empty());

        let training_position = TrainingPosition::from(search.tree());

        let state_after = State::from(&training_position);

        assert_eq!(state_before, state_after);
    }
}

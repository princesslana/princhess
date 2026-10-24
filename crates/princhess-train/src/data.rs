use arrayvec::ArrayVec;
use bytemuck::{self, Pod, Zeroable};
use princhess::chess::{Bitboard, Board, Castling, Color, Move, Piece, Square};
use princhess::engine::SCALE;
use princhess::math;
use princhess::mcts::Mcts;
use princhess::state::State;
use std::fs::File;
use std::io::{self, ErrorKind, Read, Seek, SeekFrom, Write};
use std::mem;
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::thread;

#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct TrainingPosition {
    occupied: Bitboard,
    pieces: [u8; 16], // Packed representation of pieces

    evaluation: i32,
    result: i8,
    stm: u8,

    legal_moves: [Move; TrainingPosition::MAX_MOVES],
    policy_target: [u8; TrainingPosition::MAX_MOVES],
}

// Updated size check: 8 (occupied) + 16 (pieces) + 4 (evaluation) + 1 (result) + 1 (stm) + 108 (legal_moves) + 54 (policy_target) = 192
const _SIZE_CHECK: () = assert!(mem::size_of::<TrainingPosition>() == 192);

impl TrainingPosition {
    pub const MAX_MOVES: usize = 54;

    pub const SIZE: usize = mem::size_of::<Self>();
    pub const BATCH_SIZE: usize = 16384;
    pub const BUFFER_COUNT: usize = 1 << 16;
    pub const BUFFER_SIZE: usize = Self::BUFFER_COUNT * Self::SIZE;

    /// # Errors
    ///
    /// Returns an error if writing to the output stream fails.
    pub fn write_buffer<W: Write>(out: &mut W, data: &[TrainingPosition]) -> io::Result<()> {
        out.write_all(bytemuck::cast_slice(data))
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
    pub fn piece_count(&self) -> usize {
        self.occupied.count()
    }

    #[must_use]
    pub fn phase(&self) -> usize {
        let mut phase = 0;

        for idx in 0..self.occupied.count() {
            let packed_piece = self.pieces[idx / 2] >> (4 * (idx & 1));
            let piece_type = packed_piece & 7;
            match Piece::from(piece_type) {
                Piece::KNIGHT | Piece::BISHOP => phase += 1,
                Piece::ROOK => phase += 2,
                Piece::QUEEN => phase += 4,
                _ => {}
            }
        }

        phase.clamp(0, 24)
    }

    #[must_use]
    pub fn moves(&self) -> ArrayVec<(Move, u8), { Self::MAX_MOVES }> {
        self.legal_moves
            .iter()
            .zip(self.policy_target.iter())
            .take_while(|(m, _)| **m != Move::NONE)
            .map(|(m, v)| (*m, *v))
            .collect()
    }

    pub fn set_result(&mut self, result: i8) {
        self.result = result;
    }
}

impl TrainingPosition {
    #[must_use]
    pub fn from_tree_and_priors(tree: &Mcts, priors: &[f32]) -> Self {
        let board = tree.root_state().board();

        let occupied = board.occupied();
        let stm = board.side_to_move();

        let mut pieces_for_tp = [0; 16]; // Packed representation of pieces

        for (idx, sq) in occupied.into_iter().enumerate() {
            let color_val = u8::from(board.color_at(sq));
            let piece_val = u8::from(board.piece_at(sq));

            let pc = (color_val << 3) | piece_val;

            pieces_for_tp[idx / 2] |= pc << (4 * (idx & 1));
        }

        let root_edges = tree.root_edges();
        let total_visits = tree.root_visits();
        #[allow(clippy::redundant_closure_for_method_calls)]
        let gini = math::gini(root_edges.iter().map(|e| e.visits()), total_visits);
        let explore_coef =
            tree.exploration_coefficient(&tree.mcts_options(), total_visits, false, gini);
        let lambda_n = explore_coef as f32 / (root_edges.len() as f32 + total_visits as f32);

        // Grill et al. (2020) Eq. 8: pi_bar(a) = lambda_n * prior(a) / (alpha - Q(a)),
        // alpha found by bisection (Prop. 4 / Eq. 21-22) so the distribution sums to 1.
        // Cache each edge's Q (reward() costs two atomic loads) since bisection reads
        // it on every iteration, not just once.
        let mut qs = [0.0_f32; Self::MAX_MOVES];
        let mut max_q = f32::NEG_INFINITY;
        let mut alpha_min = f32::NEG_INFINITY;

        for ((q, edge), &prior) in qs.iter_mut().zip(root_edges.iter()).zip(priors) {
            *q = edge.reward().average as f32;
            max_q = max_q.max(*q);
            alpha_min = alpha_min.max(*q + lambda_n * prior);
        }
        let mut alpha_max = max_q + lambda_n;
        let inv_lambda_n = 1.0 / lambda_n;

        for _ in 0..25 {
            let mid = f32::midpoint(alpha_min, alpha_max);
            let sum_p: f32 = qs[..root_edges.len()]
                .iter()
                .zip(priors)
                .map(|(&q, &prior)| prior / (mid - q))
                .sum();

            if sum_p > inv_lambda_n {
                alpha_min = mid;
            } else {
                alpha_max = mid;
            }
        }
        let alpha = f32::midpoint(alpha_min, alpha_max);

        let mut nodes = [(Move::NONE, 0.0_f32); Self::MAX_MOVES];
        let mut max_weight = 0.0_f32;

        for (((node, &q), &prior), edge) in nodes
            .iter_mut()
            .zip(qs.iter())
            .zip(priors)
            .zip(root_edges.iter())
        {
            let weight = (lambda_n * prior / (alpha - q)).max(0.0);
            max_weight = max_weight.max(weight);
            *node = (*edge.get_move(), weight);
        }

        let mut legal_moves = [Move::NONE; Self::MAX_MOVES];
        let mut policy_target = [0; Self::MAX_MOVES];

        for (idx, (mv, weight)) in nodes.iter().enumerate() {
            legal_moves[idx] = *mv;

            let scaled = if max_weight > 0.0 {
                (*weight / max_weight * f32::from(u8::MAX)).round()
            } else {
                0.0
            };
            policy_target[idx] = scaled as u8;
        }

        let pv = tree.best_edge();
        let mut evaluation_i64 = pv.reward().average;

        evaluation_i64 = stm
            .fold(evaluation_i64, -evaluation_i64)
            .clamp(-(SCALE as i64), SCALE as i64);

        let evaluation = evaluation_i64 as i32;

        let result = 0;

        TrainingPosition {
            occupied,
            pieces: pieces_for_tp,
            legal_moves,
            policy_target,
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

pub struct TrainingData {
    file_positions: usize,
    full_rx: Receiver<(Vec<u8>, u64)>,
    empty_tx: SyncSender<Vec<u8>>,
    current_buf: Vec<u8>,
    positions_consumed: u64,
    _reader: thread::JoinHandle<()>,
}

impl TrainingData {
    /// # Panics
    ///
    /// Panics if the file cannot be opened or is smaller than one full buffer.
    #[must_use]
    pub fn new(path: &str) -> Self {
        let file = File::open(path).expect("Failed to open training data file");
        let file_positions = file.metadata().unwrap().len() as usize / TrainingPosition::SIZE;
        assert!(
            file_positions >= TrainingPosition::BUFFER_COUNT,
            "Training file has {file_positions} positions, need at least {} (BUFFER_COUNT)",
            TrainingPosition::BUFFER_COUNT
        );

        let (full_tx, full_rx) = sync_channel(1);
        let (empty_tx, empty_rx) = sync_channel(1);

        // Seed the reader with one empty buffer so it starts prefetching immediately.
        empty_tx
            .send(vec![0u8; TrainingPosition::BUFFER_SIZE])
            .unwrap();

        let path = path.to_string();
        let reader = thread::spawn(move || {
            let mut file = File::open(&path).expect("Failed to open training data file");
            let mut positions_consumed: u64 = 0;

            while let Ok(mut buf) = empty_rx.recv() {
                loop {
                    match file.read_exact(&mut buf) {
                        Ok(()) => {
                            positions_consumed += TrainingPosition::BUFFER_COUNT as u64;
                            if full_tx.send((buf, positions_consumed)).is_err() {
                                return;
                            }
                            break;
                        }
                        Err(e) if e.kind() == ErrorKind::UnexpectedEof => {
                            file.seek(SeekFrom::Start(0)).unwrap();
                            positions_consumed = 0;
                        }
                        Err(e) => panic!("Data read error: {e}"),
                    }
                }
            }
        });

        Self {
            file_positions,
            full_rx,
            empty_tx,
            current_buf: vec![0u8; TrainingPosition::BUFFER_SIZE],
            positions_consumed: 0,
            _reader: reader,
        }
    }

    #[must_use]
    pub fn positions(&self) -> usize {
        self.file_positions
    }

    #[must_use]
    pub fn positions_consumed(&self) -> u64 {
        self.positions_consumed
    }

    /// # Panics
    ///
    /// Panics on unexpected I/O errors reading the data file.
    pub fn next_buffer(&mut self) -> &[TrainingPosition; TrainingPosition::BUFFER_COUNT] {
        let (new_buf, consumed) = self.full_rx.recv().expect("Reader thread died");
        // Return the processed buffer to the reader to be refilled, take the new one.
        let old_buf = mem::replace(&mut self.current_buf, new_buf);
        let _ = self.empty_tx.send(old_buf);
        self.positions_consumed = consumed;
        TrainingPosition::read_buffer(&self.current_buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use princhess::engine::Engine;
    use princhess::options::EngineOptions;

    const STARTPOS_NO_CASTLING: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1";
    const KIWIPETE_NO_CASTLING: &str =
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w - - 0 1";
    const KIWIPETE_NO_CASTLING_STMB: &str =
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b - - 0 1";

    // Priors can be zeros — these tests only check board-state round-tripping.
    fn training_position_for(fen: &str) -> (State, TrainingPosition) {
        let state_before = State::from_fen(fen);
        let engine = Engine::new(state_before.clone(), EngineOptions::default());
        let tree = engine.mcts();

        let priors = vec![0.0; tree.root_edges().len()];

        (
            state_before,
            TrainingPosition::from_tree_and_priors(tree, &priors),
        )
    }

    #[test]
    fn test_startpos_conversion() {
        let (state_before, training_position) = training_position_for(STARTPOS_NO_CASTLING);
        let state_after = State::from(&training_position);

        assert_eq!(state_before, state_after);
    }

    #[test]
    fn test_kiwipete_conversion() {
        let (state_before, training_position) = training_position_for(KIWIPETE_NO_CASTLING);
        let state_after = State::from(&training_position);

        assert_eq!(state_before, state_after);
    }

    #[test]
    fn test_kiwipete_stm_black_conversion() {
        let (state_before, training_position) = training_position_for(KIWIPETE_NO_CASTLING_STMB);
        let state_after = State::from(&training_position);

        assert_eq!(state_before, state_after);
    }
}

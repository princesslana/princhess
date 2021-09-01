use chess::*;
use state::State;
use std::cmp::min;
use std::io::Write;

use features_common::*;

include!(concat!(env!("OUT_DIR"), "/feature_const.rs"));
const MAX_PATTERNS_IN_POSITION: usize = 64 + 7 * 7;
const NUM_RANKS: usize = 8;
const NUM_FILES: usize = 8;
const NUM_PIECES: usize = NUM_COLORS * NUM_ROLES;
const NUM_PATTERNS: usize = NUM_PIECES * (NUM_RANKS + NUM_FILES) * NUM_PHASES;
const NUM_2X2_ELTS: usize = NUM_PIECES + 1;
const NUM_2X2_PATTERNS: usize = NUM_2X2_ELTS * NUM_2X2_ELTS * NUM_2X2_ELTS * NUM_2X2_ELTS;
pub const NUM_FEATURES: usize = NUM_DENSE_FEATURES + NUM_PATTERNS + NUM_2X2_PATTERNS;

struct FullPiece {
    role: Piece,
    color: Color,
}

fn feature_index(c: usize, p: Phase, idx: usize) -> usize {
    let x = 0;
    let x = x * NUM_PHASES + p as usize;
    let x = x * NUM_COLORS + c;
    x * NUM_NAMES + idx
}

fn encode_piece(pc: FullPiece) -> usize {
    let x = 0;
    let x = x * NUM_COLORS + pc.color as usize;
    x * NUM_ROLES + pc.role as usize
}

fn pattern_index(ph: Phase, pc: FullPiece, sq: Square, is_rank: bool) -> usize {
    let rr = if is_rank {
        sq.get_rank() as usize
    } else {
        sq.get_file() as usize + NUM_RANKS
    };
    let x = 0;
    let x = x * NUM_PHASES + ph as usize;
    let x = x * NUM_PIECES + encode_piece(pc);
    let x = x * (NUM_RANKS + NUM_FILES) + rr;
    x + NUM_DENSE_FEATURES
}

fn index_2x2_pattern(pattern: &[usize; 4]) -> usize {
    let mut x = 0;
    for y in pattern {
        x = x * NUM_2X2_ELTS + *y;
    }
    x + NUM_DENSE_FEATURES + NUM_PATTERNS
}

pub struct FeatureVec {
    pub arr: Vec<u8>,
    pub patterns: Vec<usize>,
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum GameResult {
    WhiteWin,
    BlackWin,
    Draw,
}
const NUM_OUTCOMES: usize = 3;
impl GameResult {
    pub fn flip(&self) -> Self {
        match *self {
            GameResult::WhiteWin => GameResult::BlackWin,
            GameResult::BlackWin => GameResult::WhiteWin,
            GameResult::Draw => GameResult::Draw,
        }
    }
}

impl FeatureVec {
    pub fn write_libsvm<W: Write, Pred: Fn(usize) -> bool>(
        &mut self,
        f: &mut W,
        label: usize,
        whitelist: Pred,
    ) {
        write!(f, "{}", label).unwrap();
        for (index, value) in self.arr.iter().enumerate() {
            if *value != 0 && whitelist(index) {
                write!(f, " {}:{}", index + 1, value).unwrap();
            }
        }
        self.patterns.sort_unstable();
        let mut cnt = 0;
        for i in 0..self.patterns.len() {
            let x = self.patterns[i];
            cnt += 1;
            if i + 1 == self.patterns.len() || x != self.patterns[i + 1] {
                if whitelist(x) {
                    write!(f, " {}:{}", x + 1, cnt).unwrap();
                }
                cnt = 0;
            }
        }
        write!(f, "\n").unwrap();
    }
    pub fn write_frequency(&self, freq: &mut [u64; NUM_FEATURES]) {
        for (index, &value) in self.arr.iter().enumerate() {
            freq[index] += value as u64;
        }
        for &x in &self.patterns {
            freq[x] += 1;
        }
    }
}

fn flip_side(s: Square) -> Square {
    let r = s.get_rank();
    let f = s.get_file();
    let r = r.to_index();
    let r = 7 - r;
    let r = Rank::from_index(r);
    Square::make_square(r, f)
}

fn encode_move(c: Color, p: Piece, to: Square) -> usize {
    let to = if c == Color::Black { flip_side(to) } else { to };
    let x = 0;
    let x = x * NUM_PIECES + p.to_index();
    let x = x * NUM_SQUARES + to.to_index();
    x + CAN_DO_PAWN_TO_A1
}

fn foreach_feature<F>(state: &State, mut f: F)
where
    F: FnMut(usize, u8),
{
    let board = state.board();
    let colors = &[board.side_to_move(), !board.side_to_move()];
    let non_king_pieces = &[
        Piece::Pawn,
        Piece::Knight,
        Piece::Bishop,
        Piece::Rook,
        Piece::Queen,
    ];
    let all_pieces = &[
        Piece::Pawn,
        Piece::Knight,
        Piece::Bishop,
        Piece::Rook,
        Piece::Queen,
        Piece::King,
    ];
    let p = phase(state);
    for (color_index, &c) in colors.iter().enumerate() {
        let moves = &state.move_lists()[color_index];
        let color_board = board.color_combined(c);
        for &piece in non_king_pieces {
            let cnt = (board.pieces(piece) & color_board).popcnt();
            let feat = match (piece, cnt) {
                (Piece::Pawn, 0) => PAWN_NUM_0,
                (Piece::Pawn, 1) => PAWN_NUM_1,
                (Piece::Pawn, 2) => PAWN_NUM_2,
                (Piece::Pawn, 3) => PAWN_NUM_3,
                (Piece::Pawn, 4) => PAWN_NUM_4,
                (Piece::Pawn, 5) => PAWN_NUM_5,
                (Piece::Pawn, 6) => PAWN_NUM_6,
                (Piece::Pawn, 7) => PAWN_NUM_7,
                (Piece::Pawn, _) => PAWN_NUM_8,
                (Piece::Knight, 0) => KNIGHT_NUM_0,
                (Piece::Knight, 1) => KNIGHT_NUM_1,
                (Piece::Knight, _) => KNIGHT_NUM_2,
                (Piece::Bishop, 0) => BISHOP_NUM_0,
                (Piece::Bishop, 1) => BISHOP_NUM_1,
                (Piece::Bishop, _) => BISHOP_NUM_2,
                (Piece::Rook, 0) => ROOK_NUM_0,
                (Piece::Rook, 1) => ROOK_NUM_1,
                (Piece::Rook, _) => ROOK_NUM_2,
                (Piece::Queen, 0) => QUEEN_NUM_0,
                (Piece::Queen, _) => QUEEN_NUM_1,
                (Piece::King, _) => unreachable!(),
            };
            f(feature_index(color_index, p, feat), 1);
        }
        let prev_dest = state
            .prev_move()
            .map(|x| x.get_dest())
            .unwrap_or_else(|| Square::make_square(Rank::First, File::A));
        let mut best_recapture = Piece::King;
        for mov in moves {
            if let Some(a) = board.piece_on(mov.get_source()) {
                let b = board.piece_on(mov.get_dest()).unwrap_or(Piece::King);
                f(
                    feature_index(color_index, p, CAN_DO_PAWN_TAKES_PAWN + encode_pair(a, b)),
                    1,
                );
                if mov.get_dest() == prev_dest {
                    best_recapture = min(best_recapture, a);
                }
                f(
                    feature_index(color_index, p, encode_move(c, a, mov.get_dest())),
                    1,
                );
            }
        }
        if let Some(captured) = state.prev_capture() {
            f(
                feature_index(
                    color_index,
                    p,
                    JUST_CAPTURED_PAWN_RECAPTURE_WITH_PAWN + encode_pair(captured, best_recapture),
                ),
                1,
            );
        }
        if color_index == 0 {
            if board.checkers().0 != 0 {
                f(feature_index(color_index, p, IS_CHECK), 1);
                if board.checkers().popcnt() != 1 {
                    f(feature_index(color_index, p, IS_DOUBLE_CHECK), 1);
                }
            }
            let pinned = board.pinned();
            for &piece in all_pieces {
                if (board.pieces(piece) & pinned).0 != 0 {
                    f(
                        feature_index(color_index, p, PAWN_PINNED + piece as usize),
                        1,
                    );
                }
            }
        }
        f(feature_index(color_index, p, ONE), 1);
    }
    for &color in colors {
        let color_board = board.color_combined(color);
        for &piece in all_pieces {
            for sq in board.pieces(piece) & color_board {
                for &is_rank in &[false, true] {
                    if piece != Piece::Pawn
                        && piece != Piece::Rook
                        && piece != Piece::King
                        && is_rank
                        && sq.get_rank() != Rank::First
                        && sq.get_rank() != Rank::Eighth
                    {
                        continue;
                    }
                    if piece != Piece::Pawn
                        && piece != Piece::King
                        && piece != Piece::Knight
                        && !is_rank
                    {
                        continue;
                    }
                    f(
                        pattern_index(p, FullPiece { color, role: piece }, sq, is_rank),
                        1,
                    );
                }
            }
        }
    }
    if p != Phase::Endgame {
        for file in 0..7 {
            for rank in 0..7 {
                let pattern =
                    extract_2x2_pattern(board, File::from_index(file), Rank::from_index(rank));
                f(index_2x2_pattern(&pattern), 1);
            }
        }
    }
}

pub fn featurize(state: &State) -> FeatureVec {
    let mut arr = [0u8; NUM_DENSE_FEATURES];
    let mut patterns = Vec::with_capacity(MAX_PATTERNS_IN_POSITION);
    foreach_feature(state, |i, v| {
        assert_eq!(v, 1);
        assert!(i < NUM_FEATURES);
        if i < NUM_DENSE_FEATURES {
            arr[i] += v;
        } else {
            patterns.push(i as usize);
        }
    });
    assert!(patterns.len() <= MAX_PATTERNS_IN_POSITION);
    FeatureVec {
        arr: arr.to_vec(),
        patterns,
    }
}

pub struct Model;

impl Model {
    pub fn new() -> Self {
        Model
    }
    pub fn predict(&self, state: &State) -> [f32; NUM_OUTCOMES] {
        let mut result = [0f32; NUM_OUTCOMES];
        foreach_feature(state, |i, _| {
            if i < NUM_MODEL_FEATURES {
                #[allow(clippy::needless_range_loop)]
                for j in 0..NUM_OUTCOMES {
                    // result[j] += COEF[i][j] * (v as f32);
                    result[j] += COEF[i][j];
                }
            }
        });
        for x in &mut result {
            *x = x.exp();
        }
        let s = 1.0 / result.iter().sum::<f32>();
        for x in &mut result {
            *x *= s;
        }
        if state.board().side_to_move() == Color::Black {
            result.swap(0, 1);
        }
        result
    }
    pub fn score(&self, state: &State) -> f32 {
        let probs = self.predict(state);
        probs[GameResult::WhiteWin as usize] - probs[GameResult::BlackWin as usize]
    }
}

fn phase(s: &State) -> Phase {
    if s.is_endgame() {
        Phase::Endgame
    } else {
        Phase::Midgame
    }
}

pub fn name_feature(fidx: usize) -> String {
    assert!(fidx < NUM_DENSE_FEATURES);
    for (c, side) in ["OUR", "OPPONENT"].iter().enumerate() {
        for p in &[Phase::Midgame, Phase::Endgame] {
            for (idx, name) in INDEX_NAMES.iter().enumerate() {
                if feature_index(c, *p, idx) == fidx {
                    return format!("{}_{:?}_{}", side, p, name).to_lowercase();
                }
            }
        }
    }
    unreachable!()
}

fn full_piece_on(board: &Board, sq: Square) -> Option<FullPiece> {
    let role = board.piece_on(sq)?;
    let color = if (board.color_combined(Color::White) & BitBoard::from_square(sq)).0 != 0 {
        Color::White
    } else {
        Color::Black
    };
    Some(FullPiece { color, role })
}

fn extract_2x2_pattern(board: &Board, file: File, rank: Rank) -> [usize; 4] {
    [
        full_piece_on(board, Square::make_square(rank, file))
            .map(encode_piece)
            .unwrap_or(NUM_PIECES),
        full_piece_on(board, Square::make_square(rank, file.right()))
            .map(encode_piece)
            .unwrap_or(NUM_PIECES),
        full_piece_on(board, Square::make_square(rank.up(), file))
            .map(encode_piece)
            .unwrap_or(NUM_PIECES),
        full_piece_on(board, Square::make_square(rank.up(), file.right()))
            .map(encode_piece)
            .unwrap_or(NUM_PIECES),
    ]
}

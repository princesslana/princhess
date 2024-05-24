use arrayvec::ArrayVec;

use crate::chess::{Board, Color, File, Move, MoveIndex, MoveList, Piece, Square};
use crate::uci::Tokens;

const NUMBER_KING_BUCKETS: usize = 3;
const NUMBER_THREAT_BUCKETS: usize = 4;

const VALUE_NUMBER_POSITION: usize = 768;

pub const VALUE_NUMBER_FEATURES: usize =
    VALUE_NUMBER_POSITION * NUMBER_KING_BUCKETS * NUMBER_THREAT_BUCKETS;

pub const POLICY_NUMBER_FEATURES: usize = VALUE_NUMBER_POSITION * NUMBER_THREAT_BUCKETS;

#[must_use]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct State {
    board: Board,
    prev_state_hashes: ArrayVec<u64, 100>,
}

impl State {
    pub fn from_board(board: Board) -> Self {
        Self {
            board,
            prev_state_hashes: ArrayVec::new(),
        }
    }

    #[must_use]
    pub fn from_tokens(mut tokens: Tokens) -> Option<Self> {
        let mut result = match tokens.next()? {
            "startpos" => Self::default(),
            "fen" => {
                let mut s = String::new();
                for i in 0..6 {
                    if i != 0 {
                        s.push(' ');
                    }
                    s.push_str(tokens.next()?);
                }
                Self::from_fen(&s)
            }
            _ => return None,
        };
        match tokens.next() {
            Some("moves") | None => (),
            Some(_) => return None,
        };
        for mov_str in tokens {
            for mov in result.available_moves() {
                if mov.to_uci() == mov_str {
                    result.make_move(mov);
                    break;
                }
            }
        }
        Some(result)
    }

    pub fn from_fen(fen: &str) -> Self {
        let board = Board::from_fen(fen);

        Self {
            board,
            prev_state_hashes: ArrayVec::new(),
        }
    }

    pub fn board(&self) -> &Board {
        &self.board
    }

    pub fn side_to_move(&self) -> Color {
        self.board.side_to_move()
    }

    #[must_use]
    pub fn hash(&self) -> u64 {
        self.board.hash()
    }

    #[must_use]
    pub fn is_check(&self) -> bool {
        self.board.is_check()
    }

    #[must_use]
    pub fn is_available_move(&self) -> bool {
        self.board.is_legal_move()
    }

    #[must_use]
    pub fn available_moves(&self) -> MoveList {
        self.board.legal_moves()
    }

    pub fn make_move(&mut self, mov: Move) {
        let b = &self.board;
        let piece = b.piece_at(mov.from()).unwrap();

        let is_pawn_move = piece == Piece::PAWN;
        let capture = b.piece_at(mov.to());

        if is_pawn_move || capture.is_some() {
            self.prev_state_hashes.clear();
        } else {
            self.prev_state_hashes.push(self.hash());
        }

        self.board.make_move(mov);
    }

    #[must_use]
    pub fn halfmove_counter(&self) -> usize {
        self.prev_state_hashes.len()
    }

    #[must_use]
    pub fn drawn_by_fifty_move_rule(&self) -> bool {
        self.prev_state_hashes.len() >= 100
    }

    #[must_use]
    pub fn is_repetition(&self) -> bool {
        let crnt_hash = self.hash();

        self.prev_state_hashes.iter().rev().any(|h| *h == crnt_hash)
    }

    fn feature_flip(&self) -> (bool, bool) {
        let stm = self.side_to_move();
        let b = &self.board;

        let ksq = b.king_of(stm);

        let flip_rank = stm == Color::BLACK;
        let flip_file = ksq.file() <= File::D;

        (flip_rank, flip_file)
    }

    pub fn value_features_map<F>(&self, mut f: F)
    where
        F: FnMut(usize),
    {
        let stm = self.side_to_move();
        let b = &self.board;
        let occ = b.occupied();

        let (flip_rank, flip_file) = self.feature_flip();

        let flip_square = |sq: Square| match (flip_rank, flip_file) {
            (true, true) => sq.flip_rank().flip_file(),
            (true, false) => sq.flip_rank(),
            (false, true) => sq.flip_file(),
            (false, false) => sq,
        };

        let king_bucket = KING_BUCKETS[flip_square(b.king_of(stm)).index()];

        for sq in b.occupied() {
            let piece = b.piece_at(sq).unwrap();
            let color = b.color_at(sq).unwrap();

            let sq_idx = flip_square(sq).index();
            let piece_idx = piece.index();
            let side_idx = usize::from(color != stm);

            let threatened = b.is_attacked(sq, !color, occ);
            let defended = b.is_attacked(sq, color, occ);

            let threat_bucket = usize::from(threatened) * 2 + usize::from(defended);

            let bucket = threat_bucket * NUMBER_KING_BUCKETS + king_bucket;
            let position = [0, 384][side_idx] + piece_idx * 64 + sq_idx;
            let index = bucket * 768 + position;

            f(index);
        }
    }

    pub fn policy_features_map<F>(&self, mut f: F)
    where
        F: FnMut(usize),
    {
        let stm = self.side_to_move();
        let b = &self.board;

        let (flip_rank, flip_file) = self.feature_flip();

        let flip_square = |sq: Square| match (flip_rank, flip_file) {
            (true, true) => sq.flip_rank().flip_file(),
            (true, false) => sq.flip_rank(),
            (false, true) => sq.flip_file(),
            (false, false) => sq,
        };

        for sq in b.occupied() {
            let piece = b.piece_at(sq).unwrap();
            let color = b.color_at(sq).unwrap();

            let sq_idx = flip_square(sq).index();
            let piece_idx = piece.index();
            let side_idx = usize::from(color != stm);

            let threatened = b.is_attacked(sq, !color, occ);
            let defended = b.is_attacked(sq, color, occ);

            let bucket = usize::from(threatened) * 2 + usize::from(defended);
            let position = [0, 384][side_idx] + piece_idx * 64 + sq_idx;

            let index = bucket * 768 + position;

            f(index);
        }
    }

    pub fn move_to_index(&self, mv: Move) -> MoveIndex {
        let piece = self.board.piece_at(mv.from()).unwrap();
        let from_sq = mv.from();
        let to_sq = mv.to();

        let (flip_rank, flip_file) = self.feature_flip();

        let flip_square = |sq: Square| match (flip_rank, flip_file) {
            (true, true) => sq.flip_rank().flip_file(),
            (true, false) => sq.flip_rank(),
            (false, true) => sq.flip_file(),
            (false, false) => sq,
        };

        let flip_to = flip_square(to_sq);
        let flip_from = flip_square(from_sq);

        MoveIndex::new(flip_from, flip_to, piece)
    }
}

impl Default for State {
    fn default() -> Self {
        Self::from_board(Board::startpos())
    }
}

#[rustfmt::skip]
const KING_BUCKETS: [usize; Square::COUNT] = [
    0, 0, 0, 1, 1, 0, 0, 0,
    0, 0, 1, 1, 1, 1, 0, 0,
    2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2,
];

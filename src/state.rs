use arrayvec::ArrayVec;

use crate::chess::{Board, Color, File, Move, MoveList, Piece, Rank, Square};
use crate::uci::Tokens;

const OFFSET_POSITION: usize = 0;
const OFFSET_THREATS: usize = 768;
const OFFSET_DEFENDS: usize = 768 * 2;

const NUMBER_KING_BUCKETS: usize = 3;
const NUMBER_THREAT_BUCKETS: usize = 4;

const VALUE_NUMBER_POSITION: usize = 768;

pub const VALUE_NUMBER_FEATURES: usize =
    VALUE_NUMBER_POSITION * NUMBER_KING_BUCKETS * NUMBER_THREAT_BUCKETS;

pub const POLICY_NUMBER_FEATURES: usize = 768 * 3;

#[must_use]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct State {
    board: Board,
    prev_state_hashes: ArrayVec<u64, 100>,
    prev_moves: [Option<Move>; 2],
}

impl State {
    pub fn from_board(board: Board) -> Self {
        Self::from_board_with_prev_moves(board, [None, None])
    }

    pub fn from_board_with_prev_moves(board: Board, prev_moves: [Option<Move>; 2]) -> Self {
        Self {
            board,
            prev_state_hashes: ArrayVec::new(),
            prev_moves,
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
            prev_moves: [None, None],
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

        self.prev_moves[0] = self.prev_moves[1];
        self.prev_moves[1] = Some(mov);

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

    pub fn fifty_move_counter(&self) -> usize {
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

        let feature_idx = |sq: Square, p: Piece, c: Color| {
            let sq_idx = flip_square(sq).index();
            let piece_idx = p.index();
            let side_idx = usize::from(c != stm);

            (side_idx * 6 + piece_idx) * 64 + sq_idx
        };

        for sq in b.occupied() {
            let piece = b.piece_at(sq).unwrap();
            let color = b.color_at(sq).unwrap();

            f(OFFSET_POSITION + feature_idx(sq, piece, color));

            if piece != Piece::KING {
                // Threats
                if b.is_attacked(sq, !color, b.occupied()) {
                    f(OFFSET_THREATS + feature_idx(sq, piece, color));
                }

                // Defenses
                if b.is_attacked(sq, color, b.occupied()) {
                    f(OFFSET_DEFENDS + feature_idx(sq, piece, color));
                }
            }
        }

        // We use the king threats and defenses squares for previous moves
        if let Some(m) = &self.prev_moves[0] {
            f(OFFSET_DEFENDS + feature_idx(m.to(), Piece::KING, stm));
            f(OFFSET_DEFENDS + feature_idx(m.from(), Piece::KING, !stm));
        }

        if let Some(m) = &self.prev_moves[1] {
            f(OFFSET_THREATS + feature_idx(m.to(), Piece::KING, stm));
            f(OFFSET_THREATS + feature_idx(m.from(), Piece::KING, !stm));
        }
    }

    pub fn training_features_map<F>(&self, f: F)
    where
        F: FnMut(usize),
    {
        self.value_features_map(f);
    }

    #[must_use]
    pub fn move_to_index(&self, mv: Move) -> usize {
        let piece = self.board.piece_at(mv.from()).unwrap();
        let to_sq = mv.to();

        let (flip_rank, flip_file) = self.feature_flip();

        let flip_square = |sq: Square| match (flip_rank, flip_file) {
            (true, true) => sq.flip_rank().flip_file(),
            (true, false) => sq.flip_rank(),
            (false, true) => sq.flip_file(),
            (false, false) => sq,
        };

        let piece_idx = piece.index();

        let flip_to = flip_square(to_sq);

        let adj_to = match mv.promotion() {
            Some(Piece::KNIGHT) => Square::from_coords(flip_to.file(), Rank::_1),
            Some(Piece::BISHOP | Piece::ROOK) => Square::from_coords(flip_to.file(), Rank::_2),
            _ => flip_to,
        };

        let to_idx = adj_to.index();

        piece_idx * 64 + to_idx
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

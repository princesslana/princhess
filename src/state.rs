use arrayvec::ArrayVec;

use crate::chess::{Board, Color, File, Move, MoveList, Piece, Rank, Square};
use crate::uci::Tokens;

const OFFSET_POSITION: usize = 0;
const OFFSET_THREATS: usize = 768;
const OFFSET_DEFENDS: usize = 768 * 2;

pub const NUMBER_FEATURES: usize = 768 * 3;

#[derive(Clone)]
pub struct State {
    board: Board,

    // 101 should be enough to track 50-move rule, but some games in the dataset
    // can go above this. Hence we add a little space
    prev_state_hashes: ArrayVec<u64, 128>,

    prev_moves: [Option<Move>; 2],
}

impl State {
    pub fn from_board(board: Board) -> Self {
        Self {
            board,
            prev_state_hashes: ArrayVec::new(),
            prev_moves: [None, None],
        }
    }

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

    pub fn hash(&self) -> u64 {
        self.board.hash()
    }

    pub fn is_check(&self) -> bool {
        self.board.is_check()
    }

    pub fn is_available_move(&self) -> bool {
        self.board.is_legal_move()
    }

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
        }
        self.prev_state_hashes.push(self.hash());

        self.board.make_move(mov);
    }

    pub fn halfmove_counter(&self) -> usize {
        self.prev_state_hashes.len() - 1
    }

    pub fn drawn_by_fifty_move_rule(&self) -> bool {
        self.prev_state_hashes.len() > 100
    }

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

    fn features_map<F>(&self, mut f: F)
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

    pub fn state_features_map<F>(&self, f: F)
    where
        F: FnMut(usize),
    {
        self.features_map(f);
    }

    pub fn policy_features_map<F>(&self, f: F)
    where
        F: FnMut(usize),
    {
        self.features_map(f);
    }

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
        Self::from_board(Board::default())
    }
}

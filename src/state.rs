use arrayvec::ArrayVec;
use shakmaty::fen::Fen;
use shakmaty::uci::Uci;
use shakmaty::zobrist::{Zobrist64, ZobristHash, ZobristValue};
use shakmaty::{
    self, CastlingMode, CastlingSide, Chess, Color, EnPassantMode, File, Piece, Position, Rank,
    Role, Square,
};
use std::convert::Into;

use crate::chess;
use crate::options::is_chess960;
use crate::uci::Tokens;

const OFFSET_POSITION: usize = 0;
const OFFSET_THREATS: usize = 768;
const OFFSET_DEFENDS: usize = 768 * 2;

pub const NUMBER_FEATURES: usize = 768 * 3;

#[derive(Clone)]
pub struct State {
    board: Chess,

    // 101 should be enough to track 50-move rule, but some games in the dataset
    // can go above this. Hence we add a little space
    prev_state_hashes: ArrayVec<u64, 128>,

    prev_moves: [Option<chess::Move>; 2],

    repetitions: usize,
    hash: Zobrist64,
}
impl State {
    pub fn from_board(board: Chess) -> Self {
        let hash = board.zobrist_hash(EnPassantMode::Always);

        Self {
            board,
            prev_state_hashes: ArrayVec::new(),
            prev_moves: [None, None],
            repetitions: 0,
            hash,
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
                Self::from_fen(&s)?
            }
            _ => return None,
        };
        match tokens.next() {
            Some("moves") | None => (),
            Some(_) => return None,
        };
        for mov_str in tokens {
            let uci = mov_str.parse::<Uci>().ok()?;
            let mov = uci.to_move(result.board()).ok()?;
            result.make_move(mov.into());
        }
        Some(result)
    }

    pub fn from_fen(fen: &str) -> Option<Self> {
        let board = fen
            .parse::<Fen>()
            .ok()?
            .into_position::<Chess>(CastlingMode::from_chess960(is_chess960()))
            .ok()?;

        let hash = board.zobrist_hash(EnPassantMode::Always);

        Some(Self {
            board,
            prev_state_hashes: ArrayVec::new(),
            prev_moves: [None, None],
            repetitions: 0,
            hash,
        })
    }

    pub fn board(&self) -> &Chess {
        &self.board
    }

    pub fn side_to_move(&self) -> Color {
        self.board.turn()
    }

    pub fn hash(&self) -> u64 {
        self.hash.0
    }

    pub fn available_moves(&self) -> chess::MoveList {
        let mut moves = chess::MoveList::new();

        for m in self.board.legal_moves() {
            moves.push(m.into());
        }

        moves
    }

    pub fn make_move(&mut self, mov: chess::Move) {
        let b = self.board.board();
        let role = b.role_at(mov.from()).unwrap();

        let is_pawn_move = role == Role::Pawn;
        let capture = b.role_at(mov.to());

        self.prev_moves[0] = self.prev_moves[1];
        self.prev_moves[1] = Some(mov);

        if is_pawn_move || capture.is_some() {
            self.prev_state_hashes.clear();
        }
        self.prev_state_hashes.push(self.hash());

        self.update_hash_pre();
        self.board
            .play_unchecked(&mov.to_shakmaty(self.board.board()));
        self.update_hash(!self.side_to_move(), role, capture, mov);

        self.check_for_repetition();
    }

    fn update_hash_pre(&mut self) {
        if let Some(ep_sq) = self.board.ep_square(EnPassantMode::Always) {
            self.hash ^= Zobrist64::zobrist_for_en_passant_file(ep_sq.file());
        }

        let castles = self.board.castles();

        if !castles.is_empty() {
            for color in Color::ALL {
                for side in CastlingSide::ALL {
                    if castles.has(color, side) {
                        self.hash ^= Zobrist64::zobrist_for_castling_right(color, side);
                    }
                }
            }
        }
    }

    fn update_hash(
        &mut self,
        color: Color,
        role: shakmaty::Role,
        capture: Option<shakmaty::Role>,
        mv: chess::Move,
    ) {
        if !mv.is_normal() {
            self.hash = self.board.zobrist_hash(EnPassantMode::Always);
            return;
        }

        let from = mv.from();
        let to = mv.to();

        let pc = Piece { color, role };
        self.hash ^= Zobrist64::zobrist_for_piece(from, pc);
        self.hash ^= Zobrist64::zobrist_for_piece(to, pc);

        if let Some(captured) = capture {
            self.hash ^= Zobrist64::zobrist_for_piece(
                to,
                Piece {
                    color: !color,
                    role: captured,
                },
            );
        }

        if let Some(ep_sq) = self.board.ep_square(EnPassantMode::Always) {
            self.hash ^= Zobrist64::zobrist_for_en_passant_file(ep_sq.file());
        }

        let castles = self.board.castles();

        if !castles.is_empty() {
            for color in Color::ALL {
                for side in CastlingSide::ALL {
                    if castles.has(color, side) {
                        self.hash ^= Zobrist64::zobrist_for_castling_right(color, side);
                    }
                }
            }
        }

        self.hash ^= Zobrist64::zobrist_for_white_turn();
    }

    fn check_for_repetition(&mut self) {
        let crnt_hash = self.hash();
        self.repetitions = self
            .prev_state_hashes
            .iter()
            .filter(|h| **h == crnt_hash)
            .count();
    }

    pub fn halfmove_counter(&self) -> usize {
        self.prev_state_hashes.len() - 1
    }

    pub fn drawn_by_fifty_move_rule(&self) -> bool {
        self.prev_state_hashes.len() > 100
    }

    pub fn is_repetition(&self) -> bool {
        self.repetitions > 0
    }

    fn feature_flip(&self) -> (bool, bool) {
        let stm = self.side_to_move();
        let b = self.board.board();

        let ksq = b.king_of(stm).unwrap();

        let flip_vertical = stm == Color::Black;
        let flip_horizontal = ksq.file() <= File::D;

        (flip_vertical, flip_horizontal)
    }

    fn features_map<F>(&self, mut f: F)
    where
        F: FnMut(usize),
    {
        let stm = self.side_to_move();
        let b = self.board.board();

        let (flip_vertical, flip_horizontal) = self.feature_flip();

        let flip_square = |sq: Square| match (flip_vertical, flip_horizontal) {
            (true, true) => sq.flip_vertical().flip_horizontal(),
            (true, false) => sq.flip_vertical(),
            (false, true) => sq.flip_horizontal(),
            (false, false) => sq,
        };

        let feature_idx = |sq: Square, r: Role, c: Color| {
            let sq_idx = flip_square(sq) as usize;
            let role_idx = r as usize - 1;
            let side_idx = usize::from(c != stm);

            (side_idx * 6 + role_idx) * 64 + sq_idx
        };

        for sq in b.occupied() {
            let role = b.role_at(sq).unwrap();
            let color = b.color_at(sq).unwrap();

            f(OFFSET_POSITION + feature_idx(sq, role, color));

            if role != Role::King {
                // Threats
                if b.attacks_to(sq, !color, b.occupied()).any() {
                    f(OFFSET_THREATS + feature_idx(sq, role, color));
                }

                // Defenses
                if b.attacks_to(sq, color, b.occupied()).any() {
                    f(OFFSET_DEFENDS + feature_idx(sq, role, color));
                }
            }
        }

        // We use the king threats and defenses squares for previous moves
        if let Some(m) = &self.prev_moves[0] {
            f(OFFSET_DEFENDS + feature_idx(m.to(), Role::King, stm));
            f(OFFSET_DEFENDS + feature_idx(m.from(), Role::King, !stm));
        }

        if let Some(m) = &self.prev_moves[1] {
            f(OFFSET_THREATS + feature_idx(m.to(), Role::King, stm));
            f(OFFSET_THREATS + feature_idx(m.from(), Role::King, !stm));
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

    pub fn move_to_index(&self, mv: chess::Move) -> usize {
        let role = self.board.board().role_at(mv.from()).unwrap();
        let to_sq = mv.to();

        let (flip_vertical, flip_horizontal) = self.feature_flip();

        let flip_square = |sq: shakmaty::Square| match (flip_vertical, flip_horizontal) {
            (true, true) => sq.flip_vertical().flip_horizontal(),
            (true, false) => sq.flip_vertical(),
            (false, true) => sq.flip_horizontal(),
            (false, false) => sq,
        };

        let role_idx = role as usize - 1;

        let flip_to = flip_square(to_sq);

        let adj_to = match mv.promotion() {
            Some(Role::Knight) => Square::from_coords(flip_to.file(), Rank::First),
            Some(Role::Bishop | Role::Rook) => Square::from_coords(flip_to.file(), Rank::Second),
            _ => flip_to,
        };

        let to_idx = adj_to as usize;

        role_idx * 64 + to_idx
    }
}

impl Default for State {
    fn default() -> Self {
        Self::from_board(Chess::default())
    }
}

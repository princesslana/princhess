use arrayvec::ArrayVec;
use shakmaty::attacks;
use shakmaty::fen::Fen;
use shakmaty::uci::Uci;
use shakmaty::zobrist::{Zobrist64, ZobristHash, ZobristValue};
use shakmaty::{
    self, CastlingMode, CastlingSide, Chess, Color, EnPassantMode, File, Move, MoveList, Piece,
    Position, Role, Square,
};
use std::convert::Into;

use crate::options::is_chess960;
use crate::uci::Tokens;

const OFFSET_POSITION: usize = 0;
const OFFSET_THREATS: usize = 768;
const OFFSET_DEFENDS: usize = 768 * 2;

pub const NUMBER_FEATURES: usize = 768 * 3;
pub const NUMBER_MOVE_IDX: usize = 384;

pub struct Builder {
    initial_state: Chess,
    crnt_state: Chess,
    moves: Vec<Move>,
}

impl Builder {
    pub fn chess(&self) -> &Chess {
        &self.crnt_state
    }

    pub fn make_move(&mut self, mov: Move) {
        self.crnt_state = self.crnt_state.clone().play(&mov).unwrap();
        self.moves.push(mov);
    }

    pub fn from_fen(fen: &str) -> Option<Self> {
        Some(
            fen.parse::<Fen>()
                .ok()?
                .into_position::<Chess>(CastlingMode::from_chess960(is_chess960()))
                .ok()?
                .into(),
        )
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
            let mov = uci.to_move(result.chess()).ok()?;
            result.make_move(mov);
        }
        Some(result)
    }

    pub fn extract(&self) -> (State, Vec<Move>) {
        let state = Self::from(self.initial_state.clone()).into();
        let moves = self.moves.clone();
        (state, moves)
    }
}

#[derive(Clone)]
pub struct State {
    board: Chess,

    // 101 should be enough to track 50-move rule, but some games in the dataset
    // can go above this. Hence we add a little space
    prev_state_hashes: ArrayVec<u64, 128>,

    prev_move_sq: Option<Square>,
    prev_capture_sq: Option<Square>,

    repetitions: usize,
    hash: Zobrist64,
}
impl State {
    pub fn from_tokens(tokens: Tokens) -> Option<Self> {
        Builder::from_tokens(tokens).map(Into::into)
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

    pub fn available_moves(&self) -> MoveList {
        self.board.legal_moves()
    }

    pub fn make_move(&mut self, mov: &Move) {
        let is_pawn_move = mov.role() == Role::Pawn;

        (self.prev_move_sq, self.prev_capture_sq) = match mov.capture() {
            Some(_) => (None, Some(mov.to())),
            None => (Some(mov.to()), None),
        };

        if is_pawn_move || self.prev_capture_sq.is_some() {
            self.prev_state_hashes.clear();
        }
        self.prev_state_hashes.push(self.hash());

        self.update_hash_pre();
        self.board.play_unchecked(mov);
        self.update_hash(!self.side_to_move(), mov);

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

    fn update_hash(&mut self, color: Color, mv: &Move) {
        match mv {
            Move::Normal {
                role,
                from,
                to,
                capture,
                promotion: None,
            } => {
                let pc = Piece { color, role: *role };
                self.hash ^= Zobrist64::zobrist_for_piece(*from, pc);
                self.hash ^= Zobrist64::zobrist_for_piece(*to, pc);

                if let Some(captured) = capture {
                    self.hash ^= Zobrist64::zobrist_for_piece(
                        *to,
                        Piece {
                            color: !color,
                            role: *captured,
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
            _ => self.hash = self.board.zobrist_hash(EnPassantMode::Always),
        };
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

            // King Virtual Moblity
            if role == Role::King {
                let us = b.by_color(color);
                let blockers = us | b.pawns();
                let virtual_mobility = attacks::queen_attacks(sq, blockers) & !us;

                for sq in virtual_mobility {
                    f(OFFSET_THREATS + feature_idx(sq, role, color));
                }
            } else {
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

        // We use the king defends squares for previous move
        if let Some(prev_move_sq) = self.prev_move_sq {
            f(OFFSET_DEFENDS + feature_idx(prev_move_sq, Role::King, stm));
        }

        if let Some(prev_capture_sq) = self.prev_capture_sq {
            f(OFFSET_DEFENDS + feature_idx(prev_capture_sq, Role::King, !stm));
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

    pub fn training_features_map<F>(&self, f: F)
    where
        F: FnMut(usize),
    {
        self.features_map(f);
    }

    pub fn move_to_index(&self, mv: &Move) -> usize {
        let to_sq = mv.to();

        let (flip_vertical, flip_horizontal) = self.feature_flip();

        let flip_square = |sq: shakmaty::Square| match (flip_vertical, flip_horizontal) {
            (true, true) => sq.flip_vertical().flip_horizontal(),
            (true, false) => sq.flip_vertical(),
            (false, true) => sq.flip_horizontal(),
            (false, false) => sq,
        };

        let role_idx = mv.role() as usize - 1;
        let to_idx = flip_square(to_sq) as usize;

        role_idx * 64 + to_idx
    }
}

impl Default for Builder {
    fn default() -> Self {
        shakmaty::Chess::default().into()
    }
}

impl Default for State {
    fn default() -> Self {
        Builder::default().into()
    }
}

impl From<shakmaty::Chess> for Builder {
    fn from(chess: shakmaty::Chess) -> Self {
        Self {
            initial_state: chess.clone(),
            crnt_state: chess,
            moves: Vec::new(),
        }
    }
}

impl From<Builder> for State {
    fn from(sb: Builder) -> Self {
        let hash = sb.initial_state.zobrist_hash(EnPassantMode::Always);

        let mut state = State {
            board: sb.initial_state,
            prev_state_hashes: ArrayVec::new(),
            prev_move_sq: None,
            prev_capture_sq: None,
            repetitions: 0,
            hash,
        };

        for mov in sb.moves {
            state.make_move(&mov);
        }

        state
    }
}

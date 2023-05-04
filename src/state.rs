use arrayvec::ArrayVec;
use shakmaty::fen::Fen;
use shakmaty::uci::Uci;
use shakmaty::zobrist::{ZobristHash, ZobristValue};
use shakmaty::{
    self, CastlingMode, CastlingSide, Chess, Color, File, Move, MoveList, Piece, Position, Role,
    Setup,
};

use crate::options::is_chess960;
use crate::uci::Tokens;

const NF_PIECES: usize = 2 * 6 * 64; // color * roles * squares
const NF_LAST_CAPTURE: usize = 5 * 64; // roles (-king) * squares
const NF_THREATS: usize = 2 * 6 * 64; // color * roles * suquares

const OFFSET_LAST_CAPTURE: usize = NF_PIECES;
const OFFSET_THREATS: usize = NF_PIECES + NF_LAST_CAPTURE;

pub const NUMBER_FEATURES: usize = NF_PIECES + NF_LAST_CAPTURE + NF_THREATS;

pub struct StateBuilder {
    initial_state: Chess,
    crnt_state: Chess,
    moves: Vec<Move>,
}

impl StateBuilder {
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
                .position::<Chess>(CastlingMode::from_chess960(is_chess960()))
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
            Some("moves") => (),
            Some(_) => return None,
            None => (),
        };
        for mov_str in tokens {
            let uci = mov_str.parse::<Uci>().ok()?;
            let mov = uci.to_move(result.chess()).ok()?;
            result.make_move(mov);
        }
        Some(result)
    }

    pub fn extract(&self) -> (State, Vec<Move>) {
        let state = StateBuilder::from(self.initial_state.clone()).into();
        let moves = self.moves.to_vec();
        (state, moves)
    }
}

#[derive(Clone)]
pub struct State {
    board: Chess,
    prev_capture: Option<shakmaty::Role>,
    prev_capture_sq: Option<shakmaty::Square>,

    // 101 should be enough to track 50-move rule, but some games in the dataset
    // can go above this. Hence we add a little space
    prev_state_hashes: ArrayVec<u64, 128>,

    repetitions: usize,
    hash: u64,
}
impl State {
    pub fn from_tokens(tokens: Tokens) -> Option<Self> {
        StateBuilder::from_tokens(tokens).map(|x| x.into())
    }

    pub fn board(&self) -> &Chess {
        &self.board
    }

    pub fn side_to_move(&self) -> Color {
        self.board.turn()
    }

    pub fn hash(&self) -> u64 {
        self.hash
    }

    pub fn available_moves(&self) -> MoveList {
        self.board.legal_moves()
    }

    pub fn make_move(&mut self, mov: &Move) {
        self.prev_capture = mov.capture();
        self.prev_capture_sq = self.prev_capture_sq.map(|_| mov.to());

        let is_pawn_move = mov.role() == Role::Pawn;

        if is_pawn_move || self.prev_capture.is_some() {
            self.prev_state_hashes.clear();
        }
        self.prev_state_hashes.push(self.hash());

        self.update_hash_pre();
        self.board.play_unchecked(mov);
        self.update_hash(!self.side_to_move(), mov);

        self.check_for_repetition();
    }

    fn update_hash_pre(&mut self) {
        if let Some(ep_sq) = self.board.ep_square() {
            self.hash ^= u64::zobrist_for_en_passant_file(ep_sq.file());
        }

        let castles = self.board.castles();

        if !castles.is_empty() {
            for color in Color::ALL {
                for side in CastlingSide::ALL {
                    if castles.has(color, side) {
                        self.hash ^= u64::zobrist_for_castling_right(color, side);
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
                self.hash ^= u64::zobrist_for_piece(*from, pc);
                self.hash ^= u64::zobrist_for_piece(*to, pc);

                if let Some(captured) = capture {
                    self.hash ^= u64::zobrist_for_piece(
                        *to,
                        Piece {
                            color: !color,
                            role: *captured,
                        },
                    );
                }

                if let Some(ep_sq) = self.board.ep_square() {
                    self.hash ^= u64::zobrist_for_en_passant_file(ep_sq.file());
                }

                let castles = self.board.castles();

                if !castles.is_empty() {
                    for color in Color::ALL {
                        for side in CastlingSide::ALL {
                            if castles.has(color, side) {
                                self.hash ^= u64::zobrist_for_castling_right(color, side);
                            }
                        }
                    }
                }

                self.hash ^= u64::zobrist_for_white_turn();
            }
            _ => self.hash = self.board.zobrist_hash(),
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

    pub fn features_map<F>(&self, mut f: F)
    where
        F: FnMut(usize),
    {
        let stm = self.side_to_move();
        let b = self.board.board();

        let (flip_vertical, flip_horizontal) = self.feature_flip();

        let flip_square = |sq: shakmaty::Square| match (flip_vertical, flip_horizontal) {
            (true, true) => sq.flip_vertical().flip_horizontal(),
            (true, false) => sq.flip_vertical(),
            (false, true) => sq.flip_horizontal(),
            (false, false) => sq,
        };

        for sq in b.occupied() {
            let role = b.role_at(sq).unwrap();
            let color = b.color_at(sq).unwrap();

            let adj_sq = flip_square(sq);

            let sq_idx = adj_sq as usize;
            let role_idx = role as usize - 1;
            let side_idx = usize::from(color != stm);

            let feature_idx = (side_idx * 6 + role_idx) * 64 + sq_idx;

            f(feature_idx);

            if b.attacks_to(sq, !stm, b.occupied()).any() {
                f(OFFSET_THREATS + feature_idx);
            }
        }

        if let Some((sq, pc)) = self.prev_capture_sq.zip(self.prev_capture) {
            let adj_sq = flip_square(sq);
            let role_idx = pc as usize - 1;

            f(OFFSET_LAST_CAPTURE + role_idx * 64 + adj_sq as usize);
        }
    }

    pub fn features(&self) -> [f32; NUMBER_FEATURES] {
        let mut features = [0f32; NUMBER_FEATURES];

        self.features_map(|idx| features[idx] = 1f32);

        features
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

impl Default for StateBuilder {
    fn default() -> Self {
        shakmaty::Chess::default().into()
    }
}

impl Default for State {
    fn default() -> Self {
        StateBuilder::default().into()
    }
}

impl From<shakmaty::Chess> for StateBuilder {
    fn from(chess: shakmaty::Chess) -> Self {
        Self {
            initial_state: chess.clone(),
            crnt_state: chess,
            moves: Vec::new(),
        }
    }
}

impl From<StateBuilder> for State {
    fn from(sb: StateBuilder) -> Self {
        let hash = sb.initial_state.zobrist_hash();

        let mut state = State {
            board: sb.initial_state,
            prev_capture: None,
            prev_capture_sq: None,
            prev_state_hashes: ArrayVec::new(),
            repetitions: 0,
            hash,
        };

        for mov in sb.moves {
            state.make_move(&mov);
        }

        state
    }
}

use chess;
use mcts::GameState;
use move_index;
use shakmaty;
use shakmaty::{File, Position, Setup};
use smallvec::SmallVec;
use std::str::FromStr;
use transposition_table::TranspositionHash;
use uci::Tokens;

pub type Player = chess::Color;
pub type Move = shakmaty::Move;
pub type MoveList = shakmaty::MoveList;

const NF_PIECES: usize = 2 * 6 * 64; // color * roles * squares
const NF_LAST_CAPTURE: usize = 5 * 64; // roles (-king) * squares
const NF_THREATS: usize = 2 * 6 * 64; // color * roles * suquares

const OFFSET_LAST_CAPTURE: usize = NF_PIECES;
const OFFSET_THREATS: usize = NF_PIECES + NF_LAST_CAPTURE;

pub const NUMBER_FEATURES: usize = NF_PIECES + NF_LAST_CAPTURE + NF_THREATS;

pub struct StateBuilder {
    initial_state: shakmaty::Chess,
    crnt_state: shakmaty::Chess,
    moves: Vec<shakmaty::Move>,
}

impl StateBuilder {
    pub fn chess(&self) -> &shakmaty::Chess {
        &self.crnt_state
    }

    pub fn make_move(&mut self, mov: shakmaty::Move) {
        self.crnt_state = self.crnt_state.clone().play(&mov).unwrap();
        self.moves.push(mov);
    }

    pub fn from_fen(fen: &str) -> Option<Self> {
        Some(
            fen.parse::<shakmaty::fen::Fen>()
                .ok()?
                .position::<shakmaty::Chess>(shakmaty::CastlingMode::Standard)
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
            let uci = mov_str.parse::<shakmaty::uci::Uci>().ok()?;
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
    shakmaty_board: shakmaty::Chess,
    board: chess::Board,
    prev_capture: Option<shakmaty::Role>,
    prev_capture_sq: Option<shakmaty::Square>,
    prev_state_hashes: SmallVec<[u64; 64]>,
    repetitions: usize,
}
impl State {
    pub fn from_tokens(tokens: Tokens) -> Option<Self> {
        StateBuilder::from_tokens(tokens).map(|x| x.into())
    }
    pub fn board(&self) -> &chess::Board {
        &self.board
    }
    pub fn shakmaty_board(&self) -> &shakmaty::Chess {
        &self.shakmaty_board
    }

    fn check_for_repetition(&mut self) {
        let crnt_hash = self.board.get_hash();
        self.repetitions = self
            .prev_state_hashes
            .iter()
            .filter(|h| **h == crnt_hash)
            .count();
    }

    pub fn drawn_by_fifty_move_rule(&self) -> bool {
        self.prev_state_hashes.len() >= 100
    }

    pub fn drawn_by_repetition(&self) -> bool {
        self.repetitions >= 2
    }

    pub fn feature_flip(&self) -> (bool, bool) {
        let turn = self.shakmaty_board().turn();
        let b = self.shakmaty_board().board();

        let ksq = b.king_of(turn).unwrap();

        let flip_vertical = turn == shakmaty::Color::Black;
        let flip_horizontal = ksq.file() <= File::D;

        (flip_vertical, flip_horizontal)
    }

    pub fn features(&self) -> [f32; NUMBER_FEATURES] {
        let mut features = [0f32; NUMBER_FEATURES];

        let turn = self.shakmaty_board().turn();
        let b = self.shakmaty_board().board();

        let (flip_vertical, flip_horizontal) = self.feature_flip();

        let flip_square = |sq: shakmaty::Square| match (flip_vertical, flip_horizontal) {
            (true, true) => sq.flip_vertical().flip_horizontal(),
            (true, false) => sq.flip_vertical(),
            (false, true) => sq.flip_horizontal(),
            (false, false) => sq,
        };

        for (sq, pc) in b.pieces() {
            let adj_sq = flip_square(sq);

            let sq_idx = adj_sq as usize;
            let role_idx = pc.role as usize - 1;
            let side_idx = usize::from(pc.color != turn);

            let feature_idx = (side_idx * 6 + role_idx) * 64 + sq_idx;

            features[feature_idx] = 1.;

            if b.attacks_to(sq, !turn, b.occupied()).any() {
                features[OFFSET_THREATS + feature_idx] = 1.;
            }
        }

        if let Some((sq, pc)) = self.prev_capture_sq.zip(self.prev_capture) {
            let adj_sq = flip_square(sq);
            let role_idx = pc as usize - 1;

            features[OFFSET_LAST_CAPTURE + role_idx * 64 + adj_sq as usize] = 1.
        }
        features
    }

    pub fn move_to_index(&self, mv: &Move) -> usize {
        let from_sq = mv.from().unwrap();
        let to_sq = mv.to();

        let (flip_vertical, flip_horizontal) = self.feature_flip();

        let flip_square = |sq: shakmaty::Square| match (flip_vertical, flip_horizontal) {
            (true, true) => sq.flip_vertical().flip_horizontal(),
            (true, false) => sq.flip_vertical(),
            (false, true) => sq.flip_horizontal(),
            (false, false) => sq,
        };

        move_index::move_to_index(flip_square(from_sq), flip_square(to_sq))
    }
}

impl TranspositionHash for State {
    fn hash(&self) -> u64 {
        match self.repetitions {
            0 => self.board().get_hash(),
            1 => self.board().get_hash() ^ 0xDEADBEEF,
            _ => 1,
        }
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

fn convert_square(sq: shakmaty::Square) -> chess::Square {
    chess::Square::make_square(
        chess::Rank::from_index(sq.rank() as usize),
        chess::File::from_index(sq.file() as usize),
    )
}

fn convert_role(role: shakmaty::Role) -> chess::Piece {
    match role {
        shakmaty::Role::Pawn => chess::Piece::Pawn,
        shakmaty::Role::Knight => chess::Piece::Knight,
        shakmaty::Role::Bishop => chess::Piece::Bishop,
        shakmaty::Role::Rook => chess::Piece::Rook,
        shakmaty::Role::Queen => chess::Piece::Queen,
        shakmaty::Role::King => chess::Piece::King,
    }
}

fn convert_move(mov: &shakmaty::Move) -> chess::ChessMove {
    match mov {
        &shakmaty::Move::Castle { ref king, ref rook } => {
            let from = convert_square(mov.from().unwrap());
            let to = if king.file() < rook.file() {
                from.right().unwrap().right().unwrap()
            } else {
                from.left().unwrap().left().unwrap()
            };
            chess::ChessMove::new(from, to, None)
        }
        mov => chess::ChessMove::new(
            convert_square(mov.from().unwrap()),
            convert_square(mov.to()),
            mov.promotion().map(convert_role),
        ),
    }
}

impl From<StateBuilder> for State {
    fn from(sb: StateBuilder) -> Self {
        let fen = shakmaty::fen::fen(&sb.initial_state);
        let board = chess::Board::from_str(&fen).unwrap();

        let mut state = State {
            shakmaty_board: sb.initial_state,
            board,
            prev_capture: None,
            prev_capture_sq: None,
            prev_state_hashes: SmallVec::new(),
            repetitions: 0,
        };

        for mov in sb.moves {
            state.make_move(&mov);
        }

        state
    }
}

impl GameState for State {
    type Player = Player;
    type MoveList = MoveList;

    fn current_player(&self) -> Player {
        self.board.side_to_move()
    }

    fn available_moves(&self) -> MoveList {
        self.shakmaty_board().legal_moves()
    }

    fn make_move(&mut self, mov: &shakmaty::Move) {
        let b = self.shakmaty_board.board();

        self.prev_capture = b.role_at(mov.to());
        self.prev_capture_sq = self.prev_capture_sq.map(|_| mov.to());

        let is_pawn_move = b.pawns().contains(mov.from().unwrap());

        if is_pawn_move || self.prev_capture.is_some() {
            self.prev_state_hashes.clear();
        }
        self.prev_state_hashes.push(self.board.get_hash());

        self.shakmaty_board.play_unchecked(mov);
        self.board = self.board.make_move_new(convert_move(mov));
        self.check_for_repetition();
    }
}

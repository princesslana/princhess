use chess;
use mcts::GameState;
use shakmaty;
use shakmaty::Position;
use smallvec::SmallVec;
use std;
use std::cmp::max;
use std::iter::IntoIterator;
use transposition_table::TranspositionHash;
use uci::Tokens;

pub type Player = chess::Color;
pub type Move = chess::ChessMove;

pub const NUM_OCCUPIED_KEPT: usize = 4;

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
                .position::<shakmaty::Chess>()
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
        let moves = self.moves.iter().map(|m| convert_move(m)).collect();
        (state, moves)
    }
}

#[derive(Clone)]
pub struct State {
    board: chess::Board,
    prev_move: Option<chess::ChessMove>,
    prev_capture: Option<chess::Piece>,
    prev_state_hashes: SmallVec<[u64; 64]>,
    repetitions: usize,
    formerly_occupied: [chess::BitBoard; NUM_OCCUPIED_KEPT],
    frozen: bool,
    queens_off: bool,
    move_lists: [Vec<chess::ChessMove>; 2],
}

impl State {
    pub fn from_tokens(tokens: Tokens) -> Option<Self> {
        StateBuilder::from_tokens(tokens).map(|x| x.into())
    }
    #[cfg(test)]
    pub fn from_fen(fen: &str) -> Option<Self> {
        StateBuilder::from_fen(fen).map(|x| x.into())
    }
    pub fn prev_move(&self) -> Option<chess::ChessMove> {
        self.prev_move
    }
    pub fn prev_capture(&self) -> Option<chess::Piece> {
        self.prev_capture
    }
    pub fn board(&self) -> &chess::Board {
        &self.board
    }
    pub fn outcome(&self) -> chess::BoardStatus {
        if self.drawn_by_repetition() {
            chess::BoardStatus::Stalemate // close enough
        } else {
            self.board.status()
        }
    }
    pub fn formerly_occupied(&self) -> &[chess::BitBoard; NUM_OCCUPIED_KEPT] {
        &self.formerly_occupied
    }
    fn check_for_repetition(&mut self) {
        let crnt_hash = self.board.get_hash();
        self.repetitions = max(
            self.repetitions,
            self.prev_state_hashes
                .iter()
                .filter(|h| **h == crnt_hash)
                .count(),
        );
    }
    fn drawn_by_repetition(&self) -> bool {
        self.repetitions >= 2
    }
    pub fn freeze(self) -> Self {
        Self {
            frozen: true,
            ..self
        }
    }
    pub fn queens_off(&self) -> bool {
        self.queens_off
    }
    pub fn move_lists(&self) -> &[Vec<chess::ChessMove>; 2] {
        &self.move_lists
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
            mov.promotion().map(|x| convert_role(x)),
        ),
    }
}

impl From<chess::Board> for State {
    fn from(board: chess::Board) -> Self {
        Self {
            board,
            prev_move: None,
            prev_capture: None,
            prev_state_hashes: SmallVec::new(),
            repetitions: 0,
            formerly_occupied: [board.combined(); NUM_OCCUPIED_KEPT],
            frozen: false,
            queens_off: false,
            move_lists: [Vec::new(), Vec::new()],
        }
    }
}

impl From<StateBuilder> for State {
    fn from(sb: StateBuilder) -> Self {
        let fen = shakmaty::fen::fen(&sb.initial_state, &shakmaty::fen::FenOpts::default());
        let board = chess::Board::from_fen(fen).unwrap();
        let mut state = State::from(board);
        for mov in sb.moves {
            let mov = convert_move(&mov);
            assert!(
                state.board().legal(mov),
                "{} is illegal on the following board:\n{}",
                mov,
                state.board()
            );
            state.make_move(&mov);
        }
        state
    }
}

pub struct MoveList {
    arr: [chess::ChessMove; 256],
    len: usize,
}

impl MoveList {
    pub fn as_slice(&self) -> &[chess::ChessMove] {
        &self.arr[..self.len]
    }
    pub fn len(&self) -> usize {
        self.len
    }
}

pub struct MoveListIterator {
    arr: [chess::ChessMove; 256],
    len: usize,
    pos: usize,
}

impl Iterator for MoveListIterator {
    type Item = chess::ChessMove;
    fn next(&mut self) -> Option<chess::ChessMove> {
        self.pos += 1;
        if self.pos <= self.len {
            unsafe { Some(*self.arr.get_unchecked(self.pos - 1)) }
        } else {
            None
        }
    }
}

impl IntoIterator for MoveList {
    type Item = chess::ChessMove;
    type IntoIter = MoveListIterator;
    fn into_iter(self) -> MoveListIterator {
        MoveListIterator {
            pos: 0,
            arr: self.arr,
            len: self.len,
        }
    }
}

impl GameState for State {
    type Move = Move;
    type Player = Player;
    type MoveList = MoveList;

    fn current_player(&self) -> Player {
        self.board.side_to_move()
    }
    fn available_moves(&self) -> MoveList {
        let mut arr = unsafe { std::mem::uninitialized() };
        let len = if self.drawn_by_repetition() {
            0
        } else {
            self.board.enumerate_moves(&mut arr)
        };
        MoveList { arr, len }
    }
    fn make_move(&mut self, mov: &chess::ChessMove) {
        if (self.board.pieces(chess::Piece::Pawn) & chess::BitBoard::from_square(mov.get_source()))
            .0
            != 0
        {
            self.prev_state_hashes.clear();
        } else if !self.frozen {
            self.prev_state_hashes.push(self.board.get_hash());
        }
        self.prev_capture = self.board.piece_on(mov.get_dest());
        self.prev_move = Some(*mov);
        for i in (0..(NUM_OCCUPIED_KEPT - 1)).rev() {
            self.formerly_occupied[i + 1] = self.formerly_occupied[i];
        }
        self.formerly_occupied[0] = self.board.combined();
        self.board = self.board.make_move(*mov);
        self.check_for_repetition();
        self.queens_off = self.queens_off || self.board.pieces(chess::Piece::Queen).0 == 0;
        self.move_lists.swap(0, 1);
        if self.board.checkers().0 == 0 {
            self.move_lists[0] = self.available_moves().as_slice().to_vec();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shakmaty::san::San;
    use std::str::FromStr;

    #[test]
    fn threefold_repetition() {
        let mut state = StateBuilder::default();
        let moves = &["Nf3", "Nf6", "Ng1", "Ng8", "Nf3", "Nf6", "Ng1", "Ng8"];
        for m in moves {
            let m = San::from_str(m).expect("make san");
            let m = m.to_move(state.chess()).expect("convert san");
            state.make_move(m);
        }
        let state = State::from(state);
        assert!(state.outcome() == chess::BoardStatus::Stalemate);
    }
}

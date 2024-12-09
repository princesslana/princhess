use arrayvec::ArrayVec;

use crate::chess::{Board, Color, File, Move, MoveList, Square};
use crate::policy::MoveIndex;
use crate::uci::Tokens;

const NUMBER_KING_BUCKETS: usize = 3;
const NUMBER_THREAT_BUCKETS: usize = 4;
const NUMBER_POSITIONS: usize = 768;

pub const VALUE_NUMBER_FEATURES: usize =
    NUMBER_POSITIONS * NUMBER_KING_BUCKETS * NUMBER_THREAT_BUCKETS;

pub const POLICY_NUMBER_FEATURES: usize = NUMBER_POSITIONS;

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
    pub fn from_tokens(mut tokens: Tokens, is_chess960: bool) -> Option<Self> {
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
                if mov.to_uci(is_chess960) == mov_str {
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
    pub fn phase(&self) -> usize {
        let b = self.board;

        (4 * b.queens().count() + 2 * b.rooks().count() + b.bishops().count() + b.knights().count())
            .clamp(0, 24)
    }

    #[must_use]
    pub fn available_moves(&self) -> MoveList {
        self.board.legal_moves()
    }

    pub fn make_move(&mut self, mov: Move) {
        self.prev_state_hashes.push(self.hash());

        self.board.make_move(mov);

        if self.board.halfmove_clock() == 0 {
            self.prev_state_hashes.clear();
        }
    }

    #[must_use]
    #[allow(clippy::cast_sign_loss)]
    pub fn moves_left(&self) -> u16 {
        let p = f32::from(
            (self.board.fullmove_counter() - 1) * 2
                + u16::from(self.side_to_move() == Color::BLACK),
        );
        (59.3 + (72830.0 - p * 2330.0) / (p * p + p * 10.0 + 2644.0)) as u16 / 2
    }

    #[must_use]
    pub fn halfmove_clock(&self) -> u8 {
        self.board.halfmove_clock()
    }

    #[must_use]
    pub fn drawn_by_fifty_move_rule(&self) -> bool {
        self.halfmove_clock() >= 100
    }

    #[must_use]
    pub fn is_repetition(&self) -> bool {
        let crnt_hash = self.hash();

        self.prev_state_hashes.iter().rev().any(|h| *h == crnt_hash)
    }

    #[allow(clippy::similar_names)]
    pub fn value_features_map<F>(&self, mut f: F)
    where
        F: FnMut(usize),
    {
        let stm = self.side_to_move();
        let b = &self.board;
        let occ = b.occupied();

        let stm_ksq = b.king_of(stm);
        let nstm_ksq = b.king_of(!stm);

        let flip_stm = |sq: Square| match (stm == Color::BLACK, stm_ksq.file() <= File::D) {
            (true, true) => sq.flip_rank().flip_file(),
            (true, false) => sq.flip_rank(),
            (false, true) => sq.flip_file(),
            (false, false) => sq,
        };

        let flip_nstm = |sq: Square| match (stm == Color::WHITE, nstm_ksq.file() <= File::D) {
            (true, true) => sq.flip_rank().flip_file(),
            (true, false) => sq.flip_rank(),
            (false, true) => sq.flip_file(),
            (false, false) => sq,
        };

        let stm_king_bucket = KING_BUCKETS[flip_stm(stm_ksq)];
        let nstm_king_bucket = KING_BUCKETS[flip_nstm(nstm_ksq)];

        for sq in b.occupied() {
            let piece = b.piece_at(sq);
            let color = b.color_at(sq);

            let piece_idx = piece.index();
            let side_idx = usize::from(color != stm);

            let threatened = b.is_attacked(sq, !color, occ);
            let defended = b.is_attacked(sq, color, occ);

            let threat_bucket = usize::from(threatened) * 2 + usize::from(defended);

            // stm
            {
                let sq_idx = flip_stm(sq).index();

                let bucket = threat_bucket * NUMBER_KING_BUCKETS + stm_king_bucket;
                let position = [0, 384][side_idx] + piece_idx * 64 + sq_idx;
                let index = bucket * 768 + position;

                f(index);
            }

            //nstm
            {
                let sq_idx = flip_nstm(sq).index();

                let bucket = threat_bucket * NUMBER_KING_BUCKETS + nstm_king_bucket;
                let position = [384, 0][side_idx] + piece_idx * 64 + sq_idx;
                let index = bucket * 768 + position;

                f(index + VALUE_NUMBER_FEATURES);
            }
        }
    }

    pub fn policy_features_map<F>(&self, mut f: F)
    where
        F: FnMut(usize),
    {
        let stm = self.side_to_move();
        let b = &self.board;
        let occ = b.occupied();

        let flip_square = match (stm == Color::BLACK, b.king_of(stm).file() <= File::D) {
            (true, true) => |sq: Square| sq.flip_rank().flip_file(),
            (true, false) => |sq: Square| sq.flip_rank(),
            (false, true) => |sq: Square| sq.flip_file(),
            (false, false) => |sq: Square| sq,
        };

        for sq in occ {
            let piece = b.piece_at(sq);
            let color = b.color_at(sq);

            let sq_idx = flip_square(sq).index();
            let piece_idx = piece.index();
            let side_idx = usize::from(color != stm);

            let index = [0, 384][side_idx] + piece_idx * 64 + sq_idx;

            f(index);
        }
    }

    pub fn moves_to_indexes<'a>(
        &'a self,
        mvs: &'a MoveList,
    ) -> impl Iterator<Item = MoveIndex> + '_ {
        let b = self.board;
        let color = self.side_to_move();

        let flip_square = match (color == Color::BLACK, b.king_of(color).file() <= File::D) {
            (true, true) => |sq: Square| sq.flip_rank().flip_file(),
            (true, false) => |sq: Square| sq.flip_rank(),
            (false, true) => |sq: Square| sq.flip_file(),
            (false, false) => |sq: Square| sq,
        };

        mvs.iter().map(move |mv| {
            let piece = b.piece_at(mv.from());

            let from_sq = mv.from();
            let to_sq = mv.to();

            let flip_from = flip_square(from_sq);
            let flip_to = flip_square(to_sq);

            let adj_to = if mv.is_castle() {
                flip_to
                    .with_file([File::A, File::B][usize::from(flip_to.file() < flip_from.file())])
            } else {
                flip_to
            };

            MoveIndex::new(piece, mv.promotion(), flip_from, adj_to, b.see(*mv, 8))
        })
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

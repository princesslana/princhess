use crate::chess::{zobrist, Board, Color, File, Rank, Square};

#[must_use]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Castling {
    white_king: Square,
    white_queen: Square,
    black_king: Square,
    black_queen: Square,
}

impl Castling {
    const HASH_WKS: u64 = zobrist::castling_rights(0);
    const HASH_WQS: u64 = zobrist::castling_rights(1);
    const HASH_BKS: u64 = zobrist::castling_rights(2);
    const HASH_BQS: u64 = zobrist::castling_rights(3);

    pub fn none() -> Self {
        Self {
            white_king: Square::NONE,
            white_queen: Square::NONE,
            black_king: Square::NONE,
            black_queen: Square::NONE,
        }
    }

    pub fn from_squares(wk: Square, wq: Square, bk: Square, bq: Square) -> Self {
        Self {
            white_king: wk,
            white_queen: wq,
            black_king: bk,
            black_queen: bq,
        }
    }

    pub fn from_fen(board: &Board, fen: &str) -> Self {
        let mut castling = Self::none();

        let chars = fen.chars().map(|c| c as u8).collect::<Vec<_>>();

        for c in chars {
            match c {
                b'K' => {
                    castling.white_king = (board.white() & board.rooks() & Rank::_1).last_square();
                }
                b'Q' => {
                    castling.white_queen =
                        (board.white() & board.rooks() & Rank::_1).first_square();
                }
                b'k' => {
                    castling.black_king = (board.black() & board.rooks() & Rank::_8).last_square();
                }
                b'q' => {
                    castling.black_queen =
                        (board.black() & board.rooks() & Rank::_8).first_square();
                }
                b'A'..=b'H' => {
                    let king_file = board.king_of(Color::WHITE).file();
                    let rook_file = File::from(c - b'A');

                    if rook_file < king_file {
                        castling.white_queen = Square::from_coords(rook_file, Rank::_1);
                    } else {
                        castling.white_king = Square::from_coords(rook_file, Rank::_1);
                    }
                }
                b'a'..=b'h' => {
                    let king_file = board.king_of(Color::BLACK).file();
                    let rook_file = File::from(c - b'a');

                    if rook_file < king_file {
                        castling.black_queen = Square::from_coords(rook_file, Rank::_8);
                    } else {
                        castling.black_king = Square::from_coords(rook_file, Rank::_8);
                    }
                }
                _ => {}
            }
        }

        castling
    }

    #[must_use]
    pub fn any(self) -> bool {
        self.white_king != Square::NONE
            || self.white_queen != Square::NONE
            || self.black_king != Square::NONE
            || self.black_queen != Square::NONE
    }

    pub fn by_color(self, color: Color) -> (Square, Square) {
        match color {
            Color::WHITE => (self.white_king, self.white_queen),
            Color::BLACK => (self.black_king, self.black_queen),
        }
    }

    pub fn discard_color(&mut self, color: Color) {
        if color == Color::WHITE {
            self.white_king = Square::NONE;
            self.white_queen = Square::NONE;
        } else {
            self.black_king = Square::NONE;
            self.black_queen = Square::NONE;
        }
    }

    pub fn discard_rook(&mut self, sq: Square) {
        if self.white_king == sq {
            self.white_king = Square::NONE;
        }
        if self.white_queen == sq {
            self.white_queen = Square::NONE;
        }
        if self.black_king == sq {
            self.black_king = Square::NONE;
        }
        if self.black_queen == sq {
            self.black_queen = Square::NONE;
        }
    }

    #[must_use]
    pub fn hash(self) -> u64 {
        let mut hash = 0;

        if self.white_king != Square::NONE {
            hash ^= Self::HASH_WKS;
        }
        if self.white_queen != Square::NONE {
            hash ^= Self::HASH_WQS;
        }
        if self.black_king != Square::NONE {
            hash ^= Self::HASH_BKS;
        }
        if self.black_queen != Square::NONE {
            hash ^= Self::HASH_BQS;
        }

        hash
    }
}

impl Default for Castling {
    fn default() -> Self {
        Self {
            white_king: Square::H1,
            white_queen: Square::A1,
            black_king: Square::H8,
            black_queen: Square::A8,
        }
    }
}

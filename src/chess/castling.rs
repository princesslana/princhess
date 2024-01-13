use crate::chess::{zobrist, Board, Color, File, Rank, Square};

#[must_use]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Castling {
    white_king: Option<Square>,
    white_queen: Option<Square>,
    black_king: Option<Square>,
    black_queen: Option<Square>,
}

impl Castling {
    const HASH_WKS: u64 = zobrist::castling_rights(0);
    const HASH_WQS: u64 = zobrist::castling_rights(1);
    const HASH_BKS: u64 = zobrist::castling_rights(2);
    const HASH_BQS: u64 = zobrist::castling_rights(3);

    pub fn none() -> Self {
        Self {
            white_king: None,
            white_queen: None,
            black_king: None,
            black_queen: None,
        }
    }

    pub fn from_fen(board: &Board, fen: &str) -> Self {
        let mut castling = Self::none();

        let chars = fen.chars().map(|c| c as u8).collect::<Vec<_>>();

        for c in chars {
            match c {
                b'K' => castling.white_king = Some(Square::H1),
                b'Q' => castling.white_queen = Some(Square::A1),
                b'k' => castling.black_king = Some(Square::H8),
                b'q' => castling.black_queen = Some(Square::A8),
                b'A'..=b'H' => {
                    let king_file = board.king_of(Color::WHITE).file();
                    let rook_file = File::from(c - b'A');

                    if rook_file < king_file {
                        castling.white_queen = Some(Square::from_coords(rook_file, Rank::_1));
                    } else {
                        castling.white_king = Some(Square::from_coords(rook_file, Rank::_1));
                    }
                }
                b'a'..=b'h' => {
                    let king_file = board.king_of(Color::BLACK).file();
                    let rook_file = File::from(c - b'a');

                    if rook_file < king_file {
                        castling.black_queen = Some(Square::from_coords(rook_file, Rank::_8));
                    } else {
                        castling.black_king = Some(Square::from_coords(rook_file, Rank::_8));
                    }
                }
                _ => {}
            }
        }

        castling
    }

    #[must_use]
    pub fn any(self) -> bool {
        self.white_king.is_some()
            || self.white_queen.is_some()
            || self.black_king.is_some()
            || self.black_queen.is_some()
    }

    #[must_use]
    pub fn by_color(self, color: Color) -> (Option<Square>, Option<Square>) {
        match color {
            Color::WHITE => (self.white_king, self.white_queen),
            Color::BLACK => (self.black_king, self.black_queen),
        }
    }

    pub fn discard_color(&mut self, color: Color) {
        match color {
            Color::WHITE => {
                self.white_king = None;
                self.white_queen = None;
            }
            Color::BLACK => {
                self.black_king = None;
                self.black_queen = None;
            }
        }
    }

    pub fn discard_rook(&mut self, sq: Square) {
        if self.white_king == Some(sq) {
            self.white_king = None;
        }
        if self.white_queen == Some(sq) {
            self.white_queen = None;
        }
        if self.black_king == Some(sq) {
            self.black_king = None;
        }
        if self.black_queen == Some(sq) {
            self.black_queen = None;
        }
    }

    #[must_use]
    pub fn hash(self) -> u64 {
        let mut hash = 0;

        if self.white_king.is_some() {
            hash ^= Self::HASH_WKS;
        }
        if self.white_queen.is_some() {
            hash ^= Self::HASH_WQS;
        }
        if self.black_king.is_some() {
            hash ^= Self::HASH_BKS;
        }
        if self.black_queen.is_some() {
            hash ^= Self::HASH_BQS;
        }

        hash
    }
}

impl Default for Castling {
    fn default() -> Self {
        Self {
            white_king: Some(Square::H1),
            white_queen: Some(Square::A1),
            black_king: Some(Square::H8),
            black_queen: Some(Square::A8),
        }
    }
}

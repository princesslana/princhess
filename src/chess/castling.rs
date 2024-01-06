use std::convert::Into;

use crate::chess::{zobrist, Color, Square};

#[derive(Clone, Copy, Debug)]
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

    pub fn any(self) -> bool {
        self.white_king.is_some()
            || self.white_queen.is_some()
            || self.black_king.is_some()
            || self.black_queen.is_some()
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

impl From<shakmaty::Castles> for Castling {
    fn from(castles: shakmaty::Castles) -> Self {
        let white_king = castles
            .rook(shakmaty::Color::White, shakmaty::CastlingSide::KingSide)
            .map(Into::into);

        let white_queen = castles
            .rook(shakmaty::Color::White, shakmaty::CastlingSide::QueenSide)
            .map(Into::into);

        let black_king = castles
            .rook(shakmaty::Color::Black, shakmaty::CastlingSide::KingSide)
            .map(Into::into);

        let black_queen = castles
            .rook(shakmaty::Color::Black, shakmaty::CastlingSide::QueenSide)
            .map(Into::into);

        Self {
            white_king,
            white_queen,
            black_king,
            black_queen,
        }
    }
}

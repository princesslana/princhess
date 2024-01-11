use arrayvec::ArrayVec;

use crate::chess::{Piece, Square};
use crate::options::is_chess960;

#[must_use]
#[derive(Debug, Copy, Clone)]
pub struct Move(u16);

pub type MoveList = ArrayVec<Move, 256>;

impl Move {
    pub const NONE: Self = Self(0);

    const SQ_MASK: u16 = 0b11_1111;
    const TO_SHIFT: u16 = 6;
    const PROMO_MASK: u16 = 0b11;
    const PROMO_SHIFT: u16 = 12;

    const PROMO_FLAG: u16 = 0b1100_0000_0000_0000;
    const ENPASSANT_FLAG: u16 = 0b0100_0000_0000_0000;
    const CASTLE_FLAG: u16 = 0b1000_0000_0000_0000;

    pub fn new(from: Square, to: Square) -> Self {
        Self(u16::from(from) | (u16::from(to) << Self::TO_SHIFT))
    }

    pub fn new_promotion(from: Square, to: Square, promotion: Piece) -> Self {
        Self(
            Self::new(from, to).0
                | Self::PROMO_FLAG
                | ((piece_to_promotion_idx(promotion)) << Self::PROMO_SHIFT),
        )
    }

    pub fn new_en_passant(from: Square, to: Square) -> Self {
        Self(Self::new(from, to).0 | Self::ENPASSANT_FLAG)
    }

    pub fn new_castle(king: Square, rook: Square) -> Self {
        Self(Self::new(king, rook).0 | Self::CASTLE_FLAG)
    }

    pub fn from(self) -> Square {
        Square::from(self.0 & Self::SQ_MASK)
    }

    pub fn to(self) -> Square {
        Square::from((self.0 >> Self::TO_SHIFT) & Self::SQ_MASK)
    }

    #[must_use]
    pub fn is_enpassant(self) -> bool {
        self.0 & Self::ENPASSANT_FLAG != 0 && self.0 & Self::CASTLE_FLAG == 0
    }

    #[must_use]
    pub fn is_castle(self) -> bool {
        self.0 & Self::CASTLE_FLAG != 0 && self.0 & Self::ENPASSANT_FLAG == 0
    }

    #[must_use]
    pub fn is_promotion(self) -> bool {
        self.0 & Self::PROMO_FLAG == Self::PROMO_FLAG
    }

    #[must_use]
    pub fn promotion(self) -> Option<Piece> {
        if self.is_promotion() {
            Some(promotion_idx_to_piece(
                (self.0 >> Self::PROMO_SHIFT) & Self::PROMO_MASK,
            ))
        } else {
            None
        }
    }

    pub fn to_uci(self) -> String {
        let from = self.from();
        let to = if self.is_castle() && !is_chess960() {
            match self.to() {
                Square::H1 => Square::G1,
                Square::A1 => Square::C1,
                Square::H8 => Square::G8,
                Square::A8 => Square::C8,
                _ => panic!("Invalid castle move: {self:?}"),
            }
        } else {
            self.to()
        };

        let promotion = self
            .promotion()
            .map_or(String::new(), piece_to_promotion_char);

        format!("{from}{to}{promotion}")
    }
}

fn piece_to_promotion_char(piece: Piece) -> String {
    match piece {
        Piece::QUEEN => "q",
        Piece::ROOK => "r",
        Piece::BISHOP => "b",
        Piece::KNIGHT => "n",
        _ => panic!("Invalid promotion piece: {piece:?}"),
    }
    .to_string()
}

fn piece_to_promotion_idx(piece: Piece) -> u16 {
    match piece {
        Piece::QUEEN => 0,
        Piece::ROOK => 1,
        Piece::BISHOP => 2,
        Piece::KNIGHT => 3,
        _ => panic!("Invalid promotion piece: {piece:?}"),
    }
}

fn promotion_idx_to_piece(idx: u16) -> Piece {
    match idx {
        0 => Piece::QUEEN,
        1 => Piece::ROOK,
        2 => Piece::BISHOP,
        3 => Piece::KNIGHT,
        _ => panic!("Invalid promotion index: {idx}"),
    }
}

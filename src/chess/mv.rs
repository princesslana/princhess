use arrayvec::ArrayVec;
use bytemuck::{Pod, Zeroable};

use crate::chess::{Piece, Square};

#[must_use]
#[derive(Debug, Copy, Clone, Eq, PartialEq, Pod, Zeroable)]
#[repr(transparent)]
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
                | (promotion.to_promotion_idx() << Self::PROMO_SHIFT),
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

    pub fn promotion(self) -> Piece {
        if self.is_promotion() {
            Piece::from_promotion_idx((self.0 >> Self::PROMO_SHIFT) & Self::PROMO_MASK)
        } else {
            Piece::NONE
        }
    }

    /// Converts the move to UCI notation.
    ///
    /// # Panics
    ///
    /// Panics if this is a castle move to an invalid square.
    #[must_use]
    pub fn to_uci(self, is_chess960: bool) -> String {
        let from = self.from();
        let to = if self.is_castle() && !is_chess960 {
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

        let promotion = self.promotion().to_promotion_char();

        format!("{from}{to}{promotion}")
    }

    // match uci string on the stack without allocating any heap strings :)
    #[must_use]
    pub fn matches_uci(self, mov_str: &str, is_chess960: bool) -> bool {
        let bytes = mov_str.as_bytes();
        if bytes.len() != 4 && bytes.len() != 5 {
            return false;
        }

        let from = self.from();
        let to = if self.is_castle() && !is_chess960 {
            match self.to() {
                Square::H1 => Square::G1,
                Square::A1 => Square::C1,
                Square::H8 => Square::G8,
                Square::A8 => Square::C8,
                _ => return false,
            }
        } else {
            self.to()
        };

        let from_idx = from.index() as u8;
        let to_idx = to.index() as u8;

        if bytes[0] != b'a' + (from_idx % 8) || bytes[1] != b'1' + (from_idx / 8) {
            return false;
        }

        if bytes[2] != b'a' + (to_idx % 8) || bytes[3] != b'1' + (to_idx / 8) {
            return false;
        }

        let promo = self.promotion().to_promotion_char();
        if promo.is_empty() {
            bytes.len() == 4
        } else {
            bytes.len() == 5 && bytes[4] == promo.as_bytes()[0]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matches_uci_normal() {
        let m = Move::new(Square::E2, Square::E4);
        assert!(m.matches_uci("e2e4", false));
        assert!(!m.matches_uci("e2e5", false));
        assert!(!m.matches_uci("e3e4", false));
        assert!(!m.matches_uci("e2e4q", false));
    }

    #[test]
    fn test_matches_uci_promotion() {
        let m = Move::new_promotion(Square::E7, Square::E8, Piece::QUEEN);
        assert!(m.matches_uci("e7e8q", false));
        assert!(!m.matches_uci("e7e8r", false));
        assert!(!m.matches_uci("e7e8", false));
    }

    #[test]
    fn test_matches_uci_castling() {
        let m = Move::new_castle(Square::E1, Square::H1);
        // standard chess: king lands on g1
        assert!(m.matches_uci("e1g1", false));
        assert!(!m.matches_uci("e1h1", false));
        // chess960: king lands on rook square h1
        assert!(m.matches_uci("e1h1", true));
        assert!(!m.matches_uci("e1g1", true));
    }
}

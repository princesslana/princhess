use arrayvec::ArrayVec;

use crate::chess::{Piece, Square};

#[must_use]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
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
}

#[must_use]
#[derive(Debug, Copy, Clone)]
pub struct MoveIndex {
    piece: Piece,
    from_sq: Square,
    to_sq: Square,
    from_threats: u8,
    to_threats: u8,
}

impl MoveIndex {
    const FROM_BUCKETS: usize = 4;
    const TO_BUCKETS: usize = 10;

    pub const FROM_COUNT: usize = 64 * Self::FROM_BUCKETS;
    pub const TO_COUNT: usize = 64 * Self::TO_BUCKETS;

    const THREAT_SHIFT: u8 = 0;
    const DEFEND_SHIFT: u8 = 1;
    const SEE_SHIFT: u8 = 0;

    pub fn new(piece: Piece, from: Square, to: Square) -> Self {
        Self {
            piece,
            from_sq: from,
            to_sq: to,
            from_threats: 0,
            to_threats: 0,
        }
    }

    pub fn set_from_threat(&mut self, is_threat: bool) {
        self.from_threats |= u8::from(is_threat) << Self::THREAT_SHIFT;
    }

    pub fn set_from_defend(&mut self, is_defend: bool) {
        self.from_threats |= u8::from(is_defend) << Self::DEFEND_SHIFT;
    }

    pub fn set_to_good_see(&mut self, is_good_see: bool) {
        self.to_threats |= u8::from(is_good_see) << Self::SEE_SHIFT;
    }

    #[must_use]
    pub fn from_index(&self) -> usize {
        let bucket = usize::from(self.from_threats);
        bucket * 64 + self.from_sq.index()
    }

    #[must_use]
    pub fn to_index(&self) -> usize {
        let bucket = match self.piece {
            Piece::KING => 0,
            Piece::PAWN => 1,
            p => 2 + usize::from(self.to_threats) * 4 + p.index() - 1,
        };
        bucket * 64 + self.to_sq.index()
    }
}

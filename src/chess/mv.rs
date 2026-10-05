use arrayvec::ArrayVec;
use bytemuck::{Pod, Zeroable};

use crate::chess::{Piece, Square};

#[must_use]
#[derive(Debug, Copy, Clone, Eq, PartialEq, Pod, Zeroable)]
#[repr(transparent)]
pub struct Move(u16);

pub type MoveList = ArrayVec<Move, 256>;

/// A bitmask over move indices, sized to match `MoveList`'s capacity.
#[must_use]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct MoveListMask([u64; 4]);

impl MoveListMask {
    pub const NONE: Self = Self([0; 4]);

    pub fn from_predicate<F: FnMut(usize) -> bool>(len: usize, mut predicate: F) -> Self {
        let mut mask = Self::NONE;
        for i in 0..len {
            if predicate(i) {
                mask.set(i);
            }
        }
        mask
    }

    pub fn contains(self, index: usize) -> bool {
        // `& 3` lets the compiler prove the index is in range, eliding a bounds check.
        self.0[(index / 64) & 3] & (1 << (index % 64)) != 0
    }

    pub fn set(&mut self, index: usize) {
        self.0[(index / 64) & 3] |= 1 << (index % 64);
    }

    pub fn count(self) -> u32 {
        self.0.iter().map(|w| w.count_ones()).sum()
    }
}

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
}

use std::iter::FusedIterator;
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, Not};

use crate::chess::Square;

#[derive(Copy, Clone, Debug)]
pub struct Bitboard(pub u64);

impl Bitboard {
    pub const EMPTY: Self = Self(0);
    pub const FULL: Self = Self(0xFFFF_FFFF_FFFF_FFFF);

    pub const fn new(bb: u64) -> Self {
        Self(bb)
    }

    pub fn any(self) -> bool {
        self.0 != 0
    }

    pub fn contains(self, square: Square) -> bool {
        self.0 & (1 << square.index()) != 0
    }

    pub fn count(self) -> usize {
        self.0.count_ones() as usize
    }

    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    pub fn toggle(&mut self, square: Square) {
        self.0 ^= 1 << square.index();
    }

    pub fn or_square(self, square: Square) -> Self {
        Self(self.0 | (1 << square.index()))
    }

    pub fn xor_square(self, square: Square) -> Self {
        Self(self.0 ^ (1 << square.index()))
    }
}

impl BitAnd for Bitboard {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self(self.0 & rhs.0)
    }
}

impl BitAndAssign for Bitboard {
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}

impl BitOr for Bitboard {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl BitOrAssign for Bitboard {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl BitXor for Bitboard {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self::Output {
        Self(self.0 ^ rhs.0)
    }
}

impl Not for Bitboard {
    type Output = Self;

    fn not(self) -> Self::Output {
        Self(!self.0)
    }
}

pub struct Bitloop(u64);

impl IntoIterator for Bitboard {
    type Item = Square;
    type IntoIter = Bitloop;

    fn into_iter(self) -> Self::IntoIter {
        Bitloop(self.0)
    }
}

impl Iterator for Bitloop {
    type Item = Square;

    fn next(&mut self) -> Option<Self::Item> {
        if self.0 == 0 {
            return None;
        }

        let square = self.0.trailing_zeros() as u8;
        self.0 &= self.0 - 1;

        Some(Square::from(square))
    }

    fn count(self) -> usize {
        self.0.count_ones() as usize
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let count = self.0.count_ones() as usize;
        (count, Some(count))
    }
}

impl ExactSizeIterator for Bitloop {
    fn len(&self) -> usize {
        self.0.count_ones() as usize
    }
}

impl FusedIterator for Bitloop {}

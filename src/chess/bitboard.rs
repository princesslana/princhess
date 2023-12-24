use std::iter::FusedIterator;

use crate::chess::Square;

pub struct Bitboard(pub u64);

impl Bitboard {
    pub fn any(&self) -> bool {
        self.0 != 0
    }

    pub fn count(&self) -> usize {
        self.0.count_ones() as usize
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

impl From<shakmaty::Bitboard> for Bitboard {
    fn from(b: shakmaty::Bitboard) -> Self {
        Self(b.0)
    }
}

impl From<Bitboard> for shakmaty::Bitboard {
    fn from(b: Bitboard) -> Self {
        Self(b.0)
    }
}

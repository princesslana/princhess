use std::fmt::{self, Display, Formatter};
use std::ops::{Add, Sub};

use crate::chess::Bitboard;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Square(u8);

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct File(u8);

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Rank(u8);

impl Square {
    pub const A1: Square = Square(0);
    pub const C1: Square = Square(2);
    pub const G1: Square = Square(6);
    pub const H1: Square = Square(7);

    pub const A8: Square = Square(56);
    pub const C8: Square = Square(58);
    pub const G8: Square = Square(62);
    pub const H8: Square = Square(63);

    pub fn from_coords(file: File, rank: Rank) -> Square {
        Square((rank.0 * 8) + file.0)
    }

    pub fn file(self) -> File {
        File(self.0 & 7)
    }

    pub fn flip_rank(self) -> Square {
        Square(self.0 ^ 56)
    }

    pub fn flip_file(self) -> Square {
        Square(self.0 ^ 7)
    }

    pub const fn index(self) -> usize {
        self.0 as usize
    }

    pub fn rank(self) -> Rank {
        Rank(self.0 / 8)
    }

    pub fn with_file(self, file: File) -> Square {
        Square((self.0 & !7) + file.0)
    }

    pub fn with_rank(self, rank: Rank) -> Square {
        Square((self.0 & 7) + rank.0 * 8)
    }
}

impl File {
    pub const C: File = File(2);
    pub const D: File = File(3);
    pub const F: File = File(5);
    pub const G: File = File(6);

    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

impl Rank {
    pub const _1: Rank = Rank(0);
    pub const _2: Rank = Rank(1);
    pub const _3: Rank = Rank(2);
    pub const _4: Rank = Rank(3);
    pub const _5: Rank = Rank(4);
    pub const _6: Rank = Rank(5);
    pub const _7: Rank = Rank(6);
    pub const _8: Rank = Rank(7);
}

impl From<shakmaty::Square> for Square {
    fn from(square: shakmaty::Square) -> Self {
        Square::from(square as u8)
    }
}

impl From<Square> for shakmaty::Square {
    fn from(square: Square) -> Self {
        unsafe { shakmaty::Square::new_unchecked(u32::from(square.0)) }
    }
}

impl From<u8> for Square {
    fn from(square: u8) -> Self {
        Square(square)
    }
}

impl From<u16> for Square {
    fn from(square: u16) -> Self {
        Square(square as u8)
    }
}

impl From<u32> for Square {
    fn from(square: u32) -> Self {
        Square(square as u8)
    }
}

impl From<Square> for u16 {
    fn from(square: Square) -> Self {
        u16::from(square.0)
    }
}

impl From<Bitboard> for Square {
    fn from(square: Bitboard) -> Self {
        Square(square.0.trailing_zeros() as u8)
    }
}

impl Add<u8> for Square {
    type Output = Square;

    fn add(self, rhs: u8) -> Self::Output {
        Square(self.0 + rhs)
    }
}

impl Sub<u8> for Square {
    type Output = Square;

    fn sub(self, rhs: u8) -> Self::Output {
        Square(self.0 - rhs)
    }
}

impl Display for Square {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let file = ((self.0 & 7) + b'a') as char;
        let rank = (self.0 / 8) + 1;

        write!(f, "{file}{rank}")
    }
}

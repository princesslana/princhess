use std::ops::{BitOr, BitXorAssign, Index, IndexMut};

#[must_use]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Piece(u8);

impl Piece {
    pub const PAWN: Piece = Piece(0);
    pub const KNIGHT: Piece = Piece(1);
    pub const BISHOP: Piece = Piece(2);
    pub const ROOK: Piece = Piece(3);
    pub const QUEEN: Piece = Piece(4);
    pub const KING: Piece = Piece(5);

    pub const NONE: Piece = Piece(u8::MAX);

    pub const COUNT: usize = 6;

    #[must_use]
    pub const fn index(self) -> usize {
        self.0 as usize
    }

    #[must_use]
    pub const fn see_value(self) -> i32 {
        match self {
            Piece::PAWN => 100,
            Piece::KNIGHT | Piece::BISHOP => 450,
            Piece::ROOK => 650,
            Piece::QUEEN => 1250,
            _ => 0,
        }
    }

    /// Creates a piece from a promotion index (0-3).
    ///
    /// # Panics
    ///
    /// Panics if the index is not in the range 0-3.
    pub fn from_promotion_idx(idx: u16) -> Piece {
        match idx {
            0 => Piece::KNIGHT,
            1 => Piece::BISHOP,
            2 => Piece::ROOK,
            3 => Piece::QUEEN,
            _ => panic!("Invalid promotion index: {idx}"),
        }
    }

    /// Converts a piece to its promotion index.
    ///
    /// # Panics
    ///
    /// Panics if the piece is not a valid promotion piece (Knight, Bishop, Rook, Queen).
    #[must_use]
    pub fn to_promotion_idx(self) -> u16 {
        match self {
            Piece::KNIGHT => 0,
            Piece::BISHOP => 1,
            Piece::ROOK => 2,
            Piece::QUEEN => 3,
            _ => panic!("Invalid promotion piece: {self:?}"),
        }
    }

    #[must_use]
    pub fn to_promotion_char(self) -> String {
        match self {
            Piece::QUEEN => "q",
            Piece::ROOK => "r",
            Piece::BISHOP => "b",
            Piece::KNIGHT => "n",
            _ => "",
        }
        .to_string()
    }
}

impl From<Piece> for u8 {
    fn from(piece: Piece) -> Self {
        piece.0
    }
}

impl From<Piece> for u16 {
    fn from(piece: Piece) -> Self {
        u16::from(piece.0)
    }
}

impl From<u8> for Piece {
    fn from(piece: u8) -> Self {
        Piece(piece)
    }
}

impl From<usize> for Piece {
    fn from(index: usize) -> Self {
        Piece(index as u8)
    }
}

impl BitOr for Piece {
    type Output = Self;

    fn bitor(self, other: Self) -> Self {
        match self {
            Piece::NONE => other,
            _ => self,
        }
    }
}

impl BitXorAssign for Piece {
    fn bitxor_assign(&mut self, other: Self) {
        *self = match *self {
            Piece::NONE => other,
            _ => Piece::NONE,
        };
    }
}

impl<T> Index<Piece> for [T; Piece::COUNT] {
    type Output = T;

    fn index(&self, piece: Piece) -> &Self::Output {
        let idx = piece.index();

        if idx >= Piece::COUNT {
            unsafe { std::hint::unreachable_unchecked() }
        }

        &self[idx]
    }
}

impl<T> IndexMut<Piece> for [T; Piece::COUNT] {
    fn index_mut(&mut self, piece: Piece) -> &mut Self::Output {
        let idx = piece.index();

        if idx >= Piece::COUNT {
            unsafe { std::hint::unreachable_unchecked() }
        }

        &mut self[idx]
    }
}

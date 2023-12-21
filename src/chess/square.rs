use std::fmt::{self, Display, Formatter};

#[derive(Copy, Clone, PartialEq, Eq)]
pub struct Square(u8);

impl Square {
    pub const A1: Square = Square(0);
    pub const C1: Square = Square(2);
    pub const G1: Square = Square(6);
    pub const H1: Square = Square(7);

    pub const A8: Square = Square(56);
    pub const C8: Square = Square(58);
    pub const G8: Square = Square(62);
    pub const H8: Square = Square(63);
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

impl Display for Square {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let file = ((self.0 & 7) + b'a') as char;
        let rank = (self.0 / 8) + 1;

        write!(f, "{file}{rank}")
    }
}

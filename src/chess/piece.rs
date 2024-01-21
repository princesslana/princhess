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

    pub const COUNT: usize = 6;

    #[must_use]
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

impl From<Piece> for u8 {
    fn from(piece: Piece) -> Self {
        piece.0
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

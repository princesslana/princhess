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

    pub fn from_promotion_idx(idx: u16) -> Piece {
        match idx {
            0 => Piece::KNIGHT,
            1 => Piece::BISHOP,
            2 => Piece::ROOK,
            3 => Piece::QUEEN,
            _ => panic!("Invalid promotion index: {idx}"),
        }
    }

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

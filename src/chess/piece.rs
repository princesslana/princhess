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

    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

impl From<usize> for Piece {
    fn from(index: usize) -> Self {
        Piece(index as u8)
    }
}

impl From<shakmaty::Role> for Piece {
    fn from(role: shakmaty::Role) -> Self {
        match role {
            shakmaty::Role::Pawn => Piece::PAWN,
            shakmaty::Role::Knight => Piece::KNIGHT,
            shakmaty::Role::Bishop => Piece::BISHOP,
            shakmaty::Role::Rook => Piece::ROOK,
            shakmaty::Role::Queen => Piece::QUEEN,
            shakmaty::Role::King => Piece::KING,
        }
    }
}

impl From<Piece> for shakmaty::Role {
    fn from(piece: Piece) -> Self {
        match piece {
            Piece::PAWN => shakmaty::Role::Pawn,
            Piece::KNIGHT => shakmaty::Role::Knight,
            Piece::BISHOP => shakmaty::Role::Bishop,
            Piece::ROOK => shakmaty::Role::Rook,
            Piece::QUEEN => shakmaty::Role::Queen,
            Piece::KING => shakmaty::Role::King,
            _ => unreachable!(),
        }
    }
}

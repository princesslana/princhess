use crate::chess::Square;
use crate::options::is_chess960;
use arrayvec::ArrayVec;

#[derive(Debug, Copy, Clone)]
pub struct Move(u16);

pub type MoveList = ArrayVec<Move, 256>;

impl Move {
    const SQ_MASK: u16 = 0b11_1111;
    const TO_SHIFT: u16 = 6;
    const PROMO_MASK: u16 = 0b11;
    const PROMO_SHIFT: u16 = 12;

    const PROMO_FLAG: u16 = 0b1100_0000_0000_0000;
    const ENPASSANT_FLAG: u16 = 0b0100_0000_0000_0000;
    const CASTLE_FLAG: u16 = 0b1000_0000_0000_0000;

    pub fn new(from: shakmaty::Square, to: shakmaty::Square) -> Self {
        Self(from as u16 | ((to as u16) << Self::TO_SHIFT))
    }

    pub fn new_promotion(
        from: shakmaty::Square,
        to: shakmaty::Square,
        promotion: shakmaty::Role,
    ) -> Self {
        Self(
            Self::new(from, to).0
                | Self::PROMO_FLAG
                | ((role_to_promotion_idx(promotion)) << Self::PROMO_SHIFT),
        )
    }

    pub fn new_enpassant(from: shakmaty::Square, to: shakmaty::Square) -> Self {
        Self(Self::new(from, to).0 | Self::ENPASSANT_FLAG)
    }

    pub fn new_castle(king: shakmaty::Square, rook: shakmaty::Square) -> Self {
        Self(Self::new(king, rook).0 | Self::CASTLE_FLAG)
    }

    pub fn from(self) -> Square {
        Square::from(self.0 & Self::SQ_MASK)
    }

    pub fn to(self) -> Square {
        Square::from((self.0 >> Self::TO_SHIFT) & Self::SQ_MASK)
    }

    pub fn is_normal(self) -> bool {
        self.0 & Self::ENPASSANT_FLAG == 0 && self.0 & Self::CASTLE_FLAG == 0
    }

    pub fn is_enpassant(self) -> bool {
        self.0 & Self::ENPASSANT_FLAG != 0 && self.0 & Self::CASTLE_FLAG == 0
    }

    pub fn is_castle(self) -> bool {
        self.0 & Self::CASTLE_FLAG != 0 && self.0 & Self::ENPASSANT_FLAG == 0
    }

    fn is_promotion(self) -> bool {
        self.0 & Self::PROMO_FLAG == Self::PROMO_FLAG
    }

    pub fn promotion(self) -> Option<shakmaty::Role> {
        if self.is_promotion() {
            Some(promotion_idx_to_role(
                (self.0 >> Self::PROMO_SHIFT) & Self::PROMO_MASK,
            ))
        } else {
            None
        }
    }

    pub fn to_uci(self) -> String {
        let from = self.from();
        let to = if self.is_castle() && !is_chess960() {
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

        let promotion = self
            .promotion()
            .map_or(String::new(), role_to_promotion_char);

        format!("{from}{to}{promotion}")
    }

    pub fn to_shakmaty(self, board: &shakmaty::Board) -> shakmaty::Move {
        let from = shakmaty::Square::from(self.from());
        let to = shakmaty::Square::from(self.to());

        if self.is_enpassant() {
            return shakmaty::Move::EnPassant { from, to };
        }

        if self.is_castle() {
            return shakmaty::Move::Castle {
                king: from,
                rook: to,
            };
        }

        let promotion = self.promotion();
        let role = board.role_at(from).unwrap();
        let capture = board.role_at(to);

        shakmaty::Move::Normal {
            from,
            to,
            promotion,
            role,
            capture,
        }
    }
}

impl From<shakmaty::Move> for Move {
    fn from(m: shakmaty::Move) -> Self {
        match m {
            shakmaty::Move::Normal {
                from,
                to,
                promotion: None,
                ..
            } => Self::new(from, to),
            shakmaty::Move::Normal {
                from,
                to,
                promotion: Some(promo),
                ..
            } => Self::new_promotion(from, to, promo),
            shakmaty::Move::EnPassant { from, to } => Self::new_enpassant(from, to),
            shakmaty::Move::Castle { king, rook } => Self::new_castle(king, rook),
            _ => panic!("Invalid move: {m:?}"),
        }
    }
}

fn role_to_promotion_char(role: shakmaty::Role) -> String {
    match role {
        shakmaty::Role::Queen => "q",
        shakmaty::Role::Rook => "r",
        shakmaty::Role::Bishop => "b",
        shakmaty::Role::Knight => "n",
        _ => panic!("Invalid promotion role: {role:?}"),
    }
    .to_string()
}

fn role_to_promotion_idx(role: shakmaty::Role) -> u16 {
    match role {
        shakmaty::Role::Queen => 0,
        shakmaty::Role::Rook => 1,
        shakmaty::Role::Bishop => 2,
        shakmaty::Role::Knight => 3,
        _ => panic!("Invalid promotion role: {role:?}"),
    }
}

fn promotion_idx_to_role(idx: u16) -> shakmaty::Role {
    match idx {
        0 => shakmaty::Role::Queen,
        1 => shakmaty::Role::Rook,
        2 => shakmaty::Role::Bishop,
        3 => shakmaty::Role::Knight,
        _ => panic!("Invalid promotion index: {idx}"),
    }
}

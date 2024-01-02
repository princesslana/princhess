use core::mem;
use shakmaty::fen::Fen;
use shakmaty::{CastlingMode, Chess, EnPassantMode, Position};
use std::convert::Into;

use crate::chess::{
    attacks, zobrist, Bitboard, Castling, Color, File, Move, MoveList, Piece, Rank, Square,
};
use crate::options::is_chess960;

#[derive(Clone, Debug)]
pub struct Board {
    shakmaty: shakmaty::Chess,
    stm: Color,
    ep: Option<Square>,
    castling: Castling,
    hash: u64,
}

impl Board {
    pub fn new(
        shakmaty: shakmaty::Chess,
        stm: Color,
        ep: Option<Square>,
        castling: Castling,
    ) -> Self {
        let hash = 0;

        let mut b = Self {
            shakmaty,
            stm,
            ep,
            castling,
            hash,
        };

        b.hash = b.generate_zobrist_hash();

        b
    }

    pub fn from_fen(fen: &str) -> Option<Self> {
        let shakmaty = fen
            .parse::<Fen>()
            .ok()?
            .into_position::<Chess>(CastlingMode::from_chess960(is_chess960()))
            .ok()?;

        let stm = shakmaty.turn().into();
        let ep = shakmaty.ep_square(EnPassantMode::Always).map(Into::into);
        let castling = shakmaty.castles().clone().into();

        Some(Self::new(shakmaty, stm, ep, castling))
    }

    pub fn white(&self) -> Bitboard {
        self.shakmaty.board().white().into()
    }

    pub fn black(&self) -> Bitboard {
        self.shakmaty.board().black().into()
    }

    pub fn kings(&self) -> Bitboard {
        self.shakmaty.board().kings().into()
    }

    pub fn queens(&self) -> Bitboard {
        self.shakmaty.board().queens().into()
    }

    pub fn rooks(&self) -> Bitboard {
        self.shakmaty.board().rooks().into()
    }

    pub fn bishops(&self) -> Bitboard {
        self.shakmaty.board().bishops().into()
    }

    pub fn knights(&self) -> Bitboard {
        self.shakmaty.board().knights().into()
    }

    pub fn pawns(&self) -> Bitboard {
        self.shakmaty.board().pawns().into()
    }

    pub fn color_at(&self, sq: Square) -> Option<Color> {
        self.shakmaty.board().color_at(sq.into()).map(Into::into)
    }

    pub fn ep_square(&self) -> Option<Square> {
        self.ep
    }

    pub fn hash(&self) -> u64 {
        self.hash
    }

    pub fn king_of(&self, color: Color) -> Square {
        self.shakmaty.board().king_of(color.into()).unwrap().into()
    }

    #[allow(clippy::similar_names)]
    pub fn is_attacked(&self, sq: Square, attacker: Color, occ: Bitboard) -> bool {
        let them = attacker.fold(self.white(), self.black());
        let bishop_and_queens = self.bishops() | self.queens();
        let rooks_and_queens = self.rooks() | self.queens();

        (attacks::knight(sq) & self.knights() & them
            | (attacks::king(sq) & self.kings() & them)
            | (attacks::pawn(!attacker, sq) & self.pawns() & them)
            | (attacks::bishop(sq, occ) & bishop_and_queens & them)
            | (attacks::rook(sq, occ) & rooks_and_queens & them))
            .any()
    }

    pub fn is_castling_rights(&self) -> bool {
        self.castling.any()
    }

    pub fn is_check(&self) -> bool {
        self.shakmaty.is_check()
    }

    pub fn is_insufficient_material(&self) -> bool {
        self.shakmaty.is_insufficient_material()
    }

    pub fn legal_moves(&self) -> MoveList {
        let mut moves = MoveList::new();

        for m in self.shakmaty.legal_moves() {
            moves.push(m.into());
        }

        moves
    }

    pub fn occupied(&self) -> Bitboard {
        self.shakmaty.board().occupied().into()
    }

    pub fn make_move(&mut self, mov: Move) {
        let color = self.stm;
        let piece = self.piece_at(mov.from()).unwrap();
        let capture = self.piece_at(mov.to());

        self.flip_side_to_move();

        self.update_en_passant(color, piece, mov);
        self.update_castling(color, piece, mov, capture);

        self.shakmaty.play_unchecked(&mov.to_shakmaty(self));

        if mov.is_enpassant() {
            self.toggle(color, piece, mov.from());
            self.toggle(color, piece, mov.to());
            self.toggle(!color, Piece::PAWN, mov.to().with_rank(mov.from().rank()));
        } else if mov.is_castle() {
            let (king_to, rook_to) = if mov.from().file() < mov.to().file() {
                (File::G, File::F)
            } else {
                (File::C, File::D)
            };

            self.toggle(color, Piece::KING, mov.from());
            self.toggle(color, Piece::KING, mov.from().with_file(king_to));

            self.toggle(color, Piece::ROOK, mov.to());
            self.toggle(color, Piece::ROOK, mov.to().with_file(rook_to));
        } else if mov.is_promotion() {
            let promotion = mov.promotion().unwrap();

            self.toggle(color, piece, mov.from());
            self.toggle(color, promotion, mov.to());

            if let Some(capture) = capture {
                self.toggle(!color, capture, mov.to());
            }
        } else {
            self.toggle(color, piece, mov.from());
            self.toggle(color, piece, mov.to());

            if let Some(capture) = capture {
                self.toggle(!color, capture, mov.to());
            }
        }
    }

    pub fn piece_at(&self, sq: Square) -> Option<Piece> {
        self.shakmaty.board().role_at(sq.into()).map(Into::into)
    }

    pub fn side_to_move(&self) -> Color {
        self.stm
    }

    fn flip_side_to_move(&mut self) {
        self.stm = !self.stm;
        self.hash ^= zobrist::white_to_move();
    }

    fn update_en_passant(&mut self, color: Color, piece: Piece, mov: Move) {
        let new_ep = match piece {
            Piece::PAWN => {
                let from_rank = mov.from().rank();
                let to_rank = mov.to().rank();
                let is_double_push = from_rank == Rank::_2 && to_rank == Rank::_4
                    || from_rank == Rank::_7 && to_rank == Rank::_5;

                if is_double_push {
                    Some(color.fold(
                        mov.from().with_rank(Rank::_3),
                        mov.from().with_rank(Rank::_6),
                    ))
                } else {
                    None
                }
            }
            _ => None,
        };

        if let Some(ep) = mem::replace(&mut self.ep, new_ep) {
            self.hash ^= zobrist::ep(ep.file());
        }

        if let Some(ep) = self.ep {
            self.hash ^= zobrist::ep(ep.file());
        }
    }

    fn update_castling(&mut self, color: Color, piece: Piece, mov: Move, capture: Option<Piece>) {
        if piece != Piece::KING && piece != Piece::ROOK && capture != Some(Piece::ROOK) {
            return;
        }

        self.hash ^= self.castling.hash();

        match piece {
            Piece::KING => self.castling.discard_color(color),
            Piece::ROOK => self.castling.discard_rook(mov.from()),
            _ => (),
        }

        if let Some(Piece::ROOK) = capture {
            self.castling.discard_rook(mov.to());
        }

        self.hash ^= self.castling.hash();
    }

    fn toggle(&mut self, color: Color, piece: Piece, square: Square) {
        self.hash ^= zobrist::piece(color, piece, square);
    }

    fn generate_zobrist_hash(&self) -> u64 {
        let mut hash = 0;

        for sq in self.occupied() {
            let piece = self.piece_at(sq).unwrap();
            let color = self.color_at(sq).unwrap();

            hash ^= zobrist::piece(color, piece, sq);
        }

        if let Some(ep) = self.ep {
            hash ^= zobrist::ep(ep.file());
        }

        hash ^= self.castling.hash();

        if self.stm == Color::WHITE {
            hash ^= zobrist::white_to_move();
        }

        hash
    }
}

impl Default for Board {
    fn default() -> Self {
        let shakmaty = shakmaty::Chess::default();
        let stm = Color::WHITE;
        let ep = None;
        let castling = Castling::default();

        Self::new(shakmaty, stm, ep, castling)
    }
}

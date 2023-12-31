use shakmaty::fen::Fen;
use shakmaty::zobrist::{Zobrist64, ZobristHash, ZobristValue};
use shakmaty::{CastlingMode, CastlingSide, Chess, EnPassantMode, Position, Role};
use std::convert::Into;

use crate::chess::{attacks, Bitboard, Color, Move, MoveList, Piece, Rank, Square};
use crate::options::is_chess960;

#[derive(Clone, Debug)]
pub struct Board {
    shakmaty: shakmaty::Chess,
    stm: Color,
    ep: Option<Square>,
    hash: Zobrist64,
}

impl Board {
    pub fn from_fen(fen: &str) -> Option<Self> {
        let shakmaty = fen
            .parse::<Fen>()
            .ok()?
            .into_position::<Chess>(CastlingMode::from_chess960(is_chess960()))
            .ok()?;

        let stm = shakmaty.turn().into();
        let ep = shakmaty.ep_square(EnPassantMode::Always).map(Into::into);
        let hash = shakmaty.zobrist_hash(EnPassantMode::Always);

        Some(Self {
            shakmaty,
            stm,
            ep,
            hash,
        })
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
        self.hash.0
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
        self.shakmaty.castles().any()
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

        let b = self.shakmaty.board();
        let role = b.role_at(mov.from().into()).unwrap();
        let capture = b.role_at(mov.to().into());

        self.stm = !self.stm;
        self.ep = None;

        self.update_hash_pre();
        self.shakmaty.play_unchecked(&mov.to_shakmaty(self));
        self.update_hash(!self.side_to_move(), role, capture, mov);

        if piece == Piece::PAWN && (mov.to().rank() - mov.from().rank()).abs() == 2 {
            self.ep = color.fold(
                Some(mov.from().with_rank(Rank::_3)),
                Some(mov.from().with_rank(Rank::_6)),
            );
        }
    }

    pub fn piece_at(&self, sq: Square) -> Option<Piece> {
        self.shakmaty.board().role_at(sq.into()).map(Into::into)
    }

    pub fn side_to_move(&self) -> Color {
        self.stm
    }

    fn update_hash_pre(&mut self) {
        if let Some(ep_sq) = self.shakmaty.ep_square(EnPassantMode::Always) {
            self.hash ^= Zobrist64::zobrist_for_en_passant_file(ep_sq.file());
        }

        let castles = self.shakmaty.castles();

        if !castles.is_empty() {
            for color in shakmaty::Color::ALL {
                for side in CastlingSide::ALL {
                    if castles.has(color, side) {
                        self.hash ^= Zobrist64::zobrist_for_castling_right(color, side);
                    }
                }
            }
        }
    }

    fn update_hash(&mut self, color: Color, role: Role, capture: Option<Role>, mv: Move) {
        if !mv.is_normal() {
            self.hash = self.shakmaty.zobrist_hash(EnPassantMode::Always);
            return;
        }

        let from = shakmaty::Square::from(mv.from());
        let to = shakmaty::Square::from(mv.to());

        let pc = shakmaty::Piece {
            color: color.into(),
            role,
        };
        self.hash ^= Zobrist64::zobrist_for_piece(from, pc);
        self.hash ^= Zobrist64::zobrist_for_piece(to, pc);

        if let Some(captured) = capture {
            self.hash ^= Zobrist64::zobrist_for_piece(
                to,
                shakmaty::Piece {
                    color: shakmaty::Color::from(!color),
                    role: captured,
                },
            );
        }

        if let Some(ep_sq) = self.shakmaty.ep_square(EnPassantMode::Always) {
            self.hash ^= Zobrist64::zobrist_for_en_passant_file(ep_sq.file());
        }

        let castles = self.shakmaty.castles();

        if !castles.is_empty() {
            for color in shakmaty::Color::ALL {
                for side in CastlingSide::ALL {
                    if castles.has(color, side) {
                        self.hash ^= Zobrist64::zobrist_for_castling_right(color, side);
                    }
                }
            }
        }

        self.hash ^= Zobrist64::zobrist_for_white_turn();
    }
}

impl Default for Board {
    fn default() -> Self {
        let shakmaty = shakmaty::Chess::default();
        let stm = Color::WHITE;
        let ep = None;
        let hash = shakmaty.zobrist_hash(EnPassantMode::Always);

        Self {
            shakmaty,
            stm,
            ep,
            hash,
        }
    }
}

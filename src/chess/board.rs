use core::mem;
use std::ops::ControlFlow;

use crate::chess::movegen::MoveGen;
use crate::chess::{
    attacks, zobrist, Bitboard, Castling, Color, File, Move, MoveList, Piece, Rank, Square,
};

const STARTPOS_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

#[must_use]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Board {
    colors: [Bitboard; Color::COUNT],
    pieces: [Bitboard; Piece::COUNT],
    stm: Color,
    ep: Option<Square>,
    castling: Castling,
    hash: u64,
}

impl Board {
    fn empty() -> Self {
        Self {
            colors: [Bitboard::EMPTY; Color::COUNT],
            pieces: [Bitboard::EMPTY; Piece::COUNT],
            stm: Color::WHITE,
            ep: None,
            castling: Castling::default(),
            hash: 0,
        }
    }

    pub fn from_bitboards(
        colors: [Bitboard; Color::COUNT],
        pieces: [Bitboard; Piece::COUNT],
        stm: Color,
        ep: Option<Square>,
        castling: Castling,
    ) -> Self {
        let mut board = Self::empty();

        board.colors = colors;
        board.pieces = pieces;

        board.stm = stm;
        board.ep = ep;
        board.castling = castling;

        board.hash = board.generate_zobrist_hash();

        board
    }

    pub fn from_fen(fen: &str) -> Self {
        let mut board = Self::empty();

        let parts = fen.split_whitespace().collect::<Vec<_>>();

        let [fen_pos, fen_color, fen_castling, fen_ep] = parts[0..4] else {
            println!("info string invalid fen");
            return board;
        };

        let pos = fen_pos.chars().collect::<Vec<_>>();

        let (mut file, mut rank) = (0, 7);

        for c in pos {
            match c {
                '/' => {
                    file = 0;
                    rank -= 1;
                }
                '1'..='8' => {
                    file += c as u8 - b'0';
                }
                _ => {
                    let sq = Square::from_coords(File::from(file), Rank::from(rank));
                    let idx = "PNBRQKpnbrqk".find(c).unwrap();

                    let piece = Piece::from(idx % 6);
                    let color = Color::from(idx > 5);

                    board.colors[color.index()].toggle(sq);
                    board.pieces[piece.index()].toggle(sq);

                    file += 1;
                }
            }
        }

        board.stm = Color::from(fen_color == "b");

        board.castling = Castling::from_fen(&board, fen_castling);

        board.ep = match fen_ep {
            "-" => None,
            s => Some(Square::from_uci(s)),
        };

        board.hash = board.generate_zobrist_hash();

        board
    }

    pub fn white(&self) -> Bitboard {
        self.colors[Color::WHITE.index()]
    }

    pub fn black(&self) -> Bitboard {
        self.colors[Color::BLACK.index()]
    }

    pub fn kings(&self) -> Bitboard {
        self.pieces[Piece::KING.index()]
    }

    pub fn queens(&self) -> Bitboard {
        self.pieces[Piece::QUEEN.index()]
    }

    pub fn rooks(&self) -> Bitboard {
        self.pieces[Piece::ROOK.index()]
    }

    pub fn bishops(&self) -> Bitboard {
        self.pieces[Piece::BISHOP.index()]
    }

    pub fn knights(&self) -> Bitboard {
        self.pieces[Piece::KNIGHT.index()]
    }

    pub fn pawns(&self) -> Bitboard {
        self.pieces[Piece::PAWN.index()]
    }

    pub fn by_piece(&self, piece: Piece) -> Bitboard {
        self.pieces[piece.index()]
    }

    pub fn attackers(&self, sq: Square, attacker: Color, occ: Bitboard) -> Bitboard {
        let them = attacker.fold(self.white(), self.black());
        let bishop_and_queens = self.bishops() | self.queens();
        let rooks_and_queens = self.rooks() | self.queens();

        attacks::knight(sq) & self.knights() & them
            | (attacks::king(sq) & self.kings() & them)
            | (attacks::pawn(!attacker, sq) & self.pawns() & them)
            | (attacks::bishop(sq, occ) & bishop_and_queens & them)
            | (attacks::rook(sq, occ) & rooks_and_queens & them)
    }

    pub fn castling_rights(&self) -> Castling {
        self.castling
    }

    #[must_use]
    pub fn color_at(&self, sq: Square) -> Option<Color> {
        if self.colors[0].contains(sq) {
            Some(Color::WHITE)
        } else if self.colors[1].contains(sq) {
            Some(Color::BLACK)
        } else {
            None
        }
    }

    #[must_use]
    pub fn ep_square(&self) -> Option<Square> {
        self.ep
    }

    #[must_use]
    pub fn hash(&self) -> u64 {
        self.hash
    }

    pub fn king_of(&self, color: Color) -> Square {
        Square::from(self.kings() & self.colors[color.index()])
    }

    #[must_use]
    #[allow(clippy::similar_names)]
    pub fn is_attacked(&self, sq: Square, attacker: Color, occ: Bitboard) -> bool {
        self.attackers(sq, attacker, occ).any()
    }

    #[must_use]
    pub fn is_castling_rights(&self) -> bool {
        self.castling.any()
    }

    #[must_use]
    pub fn is_check(&self) -> bool {
        self.is_attacked(self.king_of(self.stm), !self.stm, self.occupied())
    }

    #[must_use]
    pub fn is_insufficient_material(&self) -> bool {
        if (self.pawns() | self.queens() | self.rooks()).any() {
            return false;
        }

        if (self.knights() | self.bishops()).count() > 1 {
            return false;
        }

        true
    }

    #[must_use]
    pub fn is_legal_move(&self) -> bool {
        MoveGen::new(self)
            .gen(|_| ControlFlow::Break(true))
            .unwrap_or(false)
    }

    #[must_use]
    pub fn legal_moves(&self) -> MoveList {
        let mut moves = MoveList::new();

        MoveGen::new(self).gen(|m| {
            moves.push(m);
            ControlFlow::<()>::Continue(())
        });

        moves
    }

    pub fn occupied(&self) -> Bitboard {
        self.white() | self.black()
    }

    pub fn make_move(&mut self, mov: Move) {
        let color = self.stm;
        let piece = self.piece_at(mov.from()).unwrap();
        let capture = self.piece_at(mov.to());

        self.flip_side_to_move();

        self.update_en_passant(color, piece, mov);
        self.update_castling(color, piece, mov, capture);

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

    #[must_use]
    pub fn piece_at(&self, sq: Square) -> Option<Piece> {
        for idx in 0..self.pieces.len() {
            if self.pieces[idx].contains(sq) {
                return Some(Piece::from(idx));
            }
        }
        None
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
        self.colors[color.index()].toggle(square);
        self.pieces[piece.index()].toggle(square);

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
        Self::from_fen(STARTPOS_FEN)
    }
}

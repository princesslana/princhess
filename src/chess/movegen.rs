use std::ops::ControlFlow::{self, Break, Continue};

use crate::chess::{attacks, Bitboard, Board, Color, File, Move, Piece, Rank, Square};

pub struct MoveGen<'a> {
    board: &'a Board,
    threats: Bitboard,
    checkers: Bitboard,
    pinned: Bitboard,
}

macro_rules! shortcircuit {
    ($($expr:expr),*) => {
        $(if let Break(r) = $expr {
            return Break(r);
        })*
    }
}

impl MoveGen<'_> {
    pub fn new(board: &Board) -> MoveGen {
        let king_sq = board.king_of(board.side_to_move());
        let threats = threats(board);
        let pinned = pinned(board);

        let checkers = if threats.contains(king_sq) {
            board.attackers(king_sq, !board.side_to_move(), board.occupied())
        } else {
            Bitboard::EMPTY
        };

        MoveGen {
            board,
            threats,
            checkers,
            pinned,
        }
    }

    pub fn gen<F, R>(&self, mut f: F) -> Option<R>
    where
        F: FnMut(Move) -> ControlFlow<R>,
    {
        if let Break(r) = self.gen_safe_king(&mut f) {
            return Some(r);
        }

        if self.checkers.count() > 1 {
            return None;
        }

        if let Break(r) = self.gen_pawn(&mut f) {
            return Some(r);
        }

        for pc in &[Piece::KNIGHT, Piece::BISHOP, Piece::ROOK, Piece::QUEEN] {
            if let Break(r) = self.gen_for_piece(*pc, &mut f) {
                return Some(r);
            }
        }

        if !self.is_check() {
            if let Break(r) = self.gen_castling(&mut f) {
                return Some(r);
            }
        }

        None
    }

    fn us(&self) -> Color {
        self.board.side_to_move()
    }

    fn us_pieces(&self) -> Bitboard {
        self.us().fold(self.board.white(), self.board.black())
    }

    fn is_check(&self) -> bool {
        self.checkers.any()
    }

    fn gen_safe_king<F, R>(&self, mut f: F) -> ControlFlow<R>
    where
        F: FnMut(Move) -> ControlFlow<R>,
    {
        let from = self.board.king_of(self.us());

        let attacks = attacks::king(from) & !self.us_pieces() & !self.threats;

        for to in attacks {
            shortcircuit!(f(Move::new(from, to)));
        }

        Continue(())
    }

    fn gen_pawn<F, R>(&self, mut f: F) -> ControlFlow<R>
    where
        F: FnMut(Move) -> ControlFlow<R>,
    {
        let pieces = self.board.pawns() & self.us_pieces();
        let targets = self.target_squares();

        let us = self.us();
        let them = !us;
        let them_pieces = them.fold(self.board.white(), self.board.black());
        let occ = self.board.occupied();

        for from in pieces & !self.pinned {
            let quiets = pawn_pushes(us, from, occ);
            let attacks = attacks::pawn(us, from) & them_pieces;
            let moves = (quiets | attacks) & targets;

            for to in moves {
                if to.rank() == Rank::_1 || to.rank() == Rank::_8 {
                    for pc in &[Piece::QUEEN, Piece::ROOK, Piece::BISHOP, Piece::KNIGHT] {
                        shortcircuit!(f(Move::new_promotion(from, to, *pc)));
                    }
                } else {
                    shortcircuit!(f(Move::new(from, to)));
                }
            }
        }

        if !self.is_check() {
            for from in pieces & self.pinned {
                let targets = targets & attacks::through(from, self.board.king_of(self.us()));
                let quiets = pawn_pushes(us, from, occ);
                let attacks = attacks::pawn(us, from) & them_pieces;
                let moves = (quiets | attacks) & targets;

                for to in moves {
                    if to.rank() == Rank::_1 || to.rank() == Rank::_8 {
                        for pc in &[Piece::QUEEN, Piece::ROOK, Piece::BISHOP, Piece::KNIGHT] {
                            shortcircuit!(f(Move::new_promotion(from, to, *pc)));
                        }
                    } else {
                        shortcircuit!(f(Move::new(from, to)));
                    }
                }
            }
        }

        if self.board.ep_square().is_some() {
            shortcircuit!(self.gen_en_passant(&mut f));
        }

        Continue(())
    }

    fn gen_en_passant<F, R>(&self, mut f: F) -> ControlFlow<R>
    where
        F: FnMut(Move) -> ControlFlow<R>,
    {
        let ep_sq = self.board.ep_square().unwrap();

        let pieces = attacks::pawn(!self.us(), ep_sq) & self.board.pawns() & self.us_pieces();
        let to = ep_sq;

        for from in pieces {
            let mv = Move::new_en_passant(from, to);

            let mut tmp = self.board.clone();
            tmp.make_move(mv);

            let king_sq = tmp.king_of(self.us());

            if !tmp.is_attacked(king_sq, !self.us(), tmp.occupied()) {
                shortcircuit!(f(mv));
            }
        }

        Continue(())
    }

    fn gen_for_piece<F, R>(&self, piece: Piece, mut f: F) -> ControlFlow<R>
    where
        F: FnMut(Move) -> ControlFlow<R>,
    {
        let pieces = self.board.by_piece(piece) & self.us_pieces();
        let targets = self.target_squares();

        for from in pieces & !self.pinned {
            let attacks =
                attacks::for_piece(piece, self.us(), from, self.board.occupied()) & targets;

            for to in attacks {
                shortcircuit!(f(Move::new(from, to)));
            }
        }

        if !self.is_check() {
            for from in pieces & self.pinned {
                let targets = targets & attacks::through(from, self.board.king_of(self.us()));
                let attacks =
                    attacks::for_piece(piece, self.us(), from, self.board.occupied()) & targets;

                for to in attacks {
                    shortcircuit!(f(Move::new(from, to)));
                }
            }
        }

        Continue(())
    }

    fn gen_castling<F, R>(&self, mut f: F) -> ControlFlow<R>
    where
        F: FnMut(Move) -> ControlFlow<R>,
    {
        let us = self.us();
        let back_rank = us.fold(Rank::_1, Rank::_8);
        let king_from = self.board.king_of(us);

        let (king_side, queen_side) = self.board.castling_rights().by_color(self.us());

        if let Some(rook_from) = king_side {
            let king_to = Square::from_coords(File::G, back_rank);
            let rook_to = Square::from_coords(File::F, back_rank);

            if self.can_castle(rook_from, king_to, rook_to) {
                shortcircuit!(f(Move::new_castle(king_from, rook_from)));
            }
        }

        if let Some(rook_from) = queen_side {
            let king_to = Square::from_coords(File::C, back_rank);
            let rook_to = Square::from_coords(File::D, back_rank);

            if self.can_castle(rook_from, king_to, rook_to) {
                shortcircuit!(f(Move::new_castle(king_from, rook_from)));
            }
        }

        Continue(())
    }

    fn can_castle(&self, rook_from: Square, king_to: Square, rook_to: Square) -> bool {
        if self.pinned.contains(rook_from) {
            return false;
        }

        let us = self.us();
        let king_from = self.board.king_of(us);

        let blockers = self
            .board
            .occupied()
            .xor_square(rook_from)
            .xor_square(king_from);

        let must_be_safe = attacks::between(king_from, king_to).or_square(king_to);
        let must_be_empty =
            must_be_safe | attacks::between(king_from, rook_from).or_square(rook_to);

        let is_empty = (must_be_empty & blockers).is_empty();
        let is_safe = (must_be_safe & self.threats).is_empty();

        is_empty && is_safe
    }

    fn target_squares(&self) -> Bitboard {
        let targets = if self.is_check() {
            let checker = Square::from(self.checkers);
            let king_sq = self.board.king_of(self.us());
            attacks::between(checker, king_sq) | self.checkers
        } else {
            Bitboard::FULL
        };
        targets & !self.us_pieces()
    }
}

fn threats(board: &Board) -> Bitboard {
    let us = board.side_to_move();
    let us_pieces = us.fold(board.white(), board.black());
    let them = !us;
    let them_pieces = them.fold(board.white(), board.black());
    let occ = board.occupied() ^ (board.kings() & us_pieces);

    let mut threats = Bitboard::EMPTY;

    for sq in board.kings() & them_pieces {
        threats |= attacks::king(sq);
    }

    for sq in (board.queens() | board.rooks()) & them_pieces {
        threats |= attacks::rook(sq, occ);
    }

    for sq in (board.queens() | board.bishops()) & them_pieces {
        threats |= attacks::bishop(sq, occ);
    }

    for sq in board.knights() & them_pieces {
        threats |= attacks::knight(sq);
    }

    for sq in board.pawns() & them_pieces {
        threats |= attacks::pawn(them, sq);
    }

    threats
}

fn pinned(board: &Board) -> Bitboard {
    let occ = board.occupied();

    let us = board.side_to_move();
    let us_pieces = us.fold(board.white(), board.black());

    let them = !us;
    let them_pieces = them.fold(board.white(), board.black());

    let king_sq = board.king_of(us);

    let mut pinned = Bitboard::EMPTY;

    let rook_pinners = attacks::xray_rook(king_sq, occ, us_pieces)
        & them_pieces
        & (board.rooks() | board.queens());

    let bishop_pinners = attacks::xray_bishop(king_sq, occ, us_pieces)
        & them_pieces
        & (board.bishops() | board.queens());

    for sq in rook_pinners | bishop_pinners {
        pinned |= attacks::between(sq, king_sq) & us_pieces;
    }

    pinned
}

fn pawn_pushes(color: Color, from: Square, occ: Bitboard) -> Bitboard {
    let mut pushes = Bitboard::EMPTY;

    let single = color.fold(from + 8, from - 8);

    pushes.toggle(single);
    pushes &= !occ;

    if pushes.any() && color.fold(from.rank() == Rank::_2, from.rank() == Rank::_7) {
        // fold is not appropriate here because it may overflow with the square maths
        let double = match color {
            Color::WHITE => from + 16,
            Color::BLACK => from - 16,
        };
        pushes.toggle(double);
        pushes &= !occ;
    }

    pushes
}

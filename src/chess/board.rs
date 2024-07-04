use std::ops::ControlFlow;

use crate::chess::movegen::MoveGen;
use crate::chess::{
    attacks, zobrist, Bitboard, Castling, Color, File, Move, MoveList, Piece, Rank, Square,
};

const STARTPOS_SCHARNAGL: usize = 518;

#[must_use]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Board {
    colors: [Bitboard; Color::COUNT],
    pieces: [Bitboard; Piece::COUNT],
    stm: Color,
    ep: Square,
    castling: Castling,
    hash: u64,
}

impl Board {
    fn empty() -> Self {
        Self {
            colors: [Bitboard::EMPTY; Color::COUNT],
            pieces: [Bitboard::EMPTY; Piece::COUNT],
            stm: Color::WHITE,
            ep: Square::NONE,
            castling: Castling::default(),
            hash: 0,
        }
    }

    pub fn startpos() -> Self {
        Self::frc(STARTPOS_SCHARNAGL)
    }

    pub fn frc(scharnagl: usize) -> Self {
        Self::dfrc(scharnagl, scharnagl)
    }

    #[allow(clippy::similar_names)]
    pub fn dfrc(white_scharnagl: usize, black_scharnagl: usize) -> Self {
        let white_back_rank = get_scharnagl_back_rank(white_scharnagl);
        let black_back_rank = get_scharnagl_back_rank(black_scharnagl);

        let mut board = Self::empty();

        for color in &[Color::WHITE, Color::BLACK] {
            let back_rank = color.fold(Rank::_1, Rank::_8);
            let pawn_rank = color.fold(Rank::_2, Rank::_7);

            for (file, piece) in color
                .fold(white_back_rank, black_back_rank)
                .iter()
                .enumerate()
            {
                board.toggle(
                    *color,
                    *piece,
                    Square::from_coords(File::from(file), back_rank),
                );
                board.toggle(
                    *color,
                    Piece::PAWN,
                    Square::from_coords(File::from(file), pawn_rank),
                );
            }
        }

        let rooks = board.rooks().into_iter().collect::<Vec<_>>();

        let [wqs, wks, bqs, bks] = &rooks[0..4] else {
            unreachable!()
        };

        board.castling = Castling::from_squares(*wks, *wqs, *bks, *bqs);

        board.hash = board.generate_zobrist_hash();

        board
    }

    pub fn from_bitboards(
        colors: [Bitboard; Color::COUNT],
        pieces: [Bitboard; Piece::COUNT],
        stm: Color,
        ep: Square,
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
            "-" => Square::NONE,
            s => Square::from_uci(s),
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

    fn by_color(&self, color: Color) -> Bitboard {
        self.colors[color.index()]
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

    pub fn threats_by(&self, attacker: Color, occ: Bitboard) -> Bitboard {
        let mut threats = Bitboard::EMPTY;

        let color = self.colors[attacker.index()];

        for sq in self.kings() & color {
            threats |= attacks::king(sq);
        }

        for sq in (self.queens() | self.rooks()) & color {
            threats |= attacks::rook(sq, occ);
        }

        for sq in (self.queens() | self.bishops()) & color {
            threats |= attacks::bishop(sq, occ);
        }

        for sq in self.knights() & color {
            threats |= attacks::knight(sq);
        }

        for sq in self.pawns() & color {
            threats |= attacks::pawn(attacker, sq);
        }

        threats
    }

    pub fn castling_rights(&self) -> Castling {
        self.castling
    }

    pub fn color_at(&self, sq: Square) -> Color {
        if self.colors[0].contains(sq) {
            Color::WHITE
        } else if self.colors[1].contains(sq) {
            Color::BLACK
        } else {
            panic!("no color at square {sq}");
        }
    }

    pub fn ep_square(&self) -> Square {
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
        let piece = self.piece_at(mov.from());
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
            let promotion = mov.promotion();

            self.toggle(color, piece, mov.from());
            self.toggle(color, promotion, mov.to());

            if capture != Piece::NONE {
                self.toggle(!color, capture, mov.to());
            }
        } else {
            self.toggle(color, piece, mov.from());
            self.toggle(color, piece, mov.to());

            if capture != Piece::NONE {
                self.toggle(!color, capture, mov.to());
            }
        }
    }

    pub fn piece_at(&self, sq: Square) -> Piece {
        for idx in 0..self.pieces.len() {
            if self.pieces[idx].contains(sq) {
                return Piece::from(idx);
            }
        }
        Piece::NONE
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
                    color.fold(
                        mov.from().with_rank(Rank::_3),
                        mov.from().with_rank(Rank::_6),
                    )
                } else {
                    Square::NONE
                }
            }
            _ => Square::NONE,
        };

        if self.ep != Square::NONE {
            self.hash ^= zobrist::ep(self.ep.file());
        }

        if new_ep != Square::NONE {
            self.hash ^= zobrist::ep(new_ep.file());
        }

        self.ep = new_ep;
    }

    fn update_castling(&mut self, color: Color, piece: Piece, mov: Move, capture: Piece) {
        if piece != Piece::KING && piece != Piece::ROOK && capture != Piece::ROOK {
            return;
        }

        self.hash ^= self.castling.hash();

        match piece {
            Piece::KING => self.castling.discard_color(color),
            Piece::ROOK => self.castling.discard_rook(mov.from()),
            _ => (),
        }

        if capture == Piece::ROOK {
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
            let piece = self.piece_at(sq);
            let color = self.color_at(sq);

            hash ^= zobrist::piece(color, piece, sq);
        }

        if self.ep != Square::NONE {
            hash ^= zobrist::ep(self.ep.file());
        }

        hash ^= self.castling.hash();

        if self.stm == Color::WHITE {
            hash ^= zobrist::white_to_move();
        }

        hash
    }

    fn gain(&self, mv: Move) -> i32 {
        if mv.is_enpassant() {
            return Piece::PAWN.see_value();
        }

        let promotion_value = match mv.promotion() {
            Piece::NONE => 0,
            p => p.see_value() - Piece::PAWN.see_value(),
        };

        self.piece_at(mv.to()).see_value() + promotion_value
    }

    #[must_use]
    pub fn see(&self, mv: Move, threshold: i32) -> bool {
        let sq = mv.to();

        let mut next = match mv.promotion() {
            Piece::NONE => self.piece_at(mv.from()),
            p => p,
        };

        let mut score = self.gain(mv) - threshold - next.see_value();

        if score >= 0 {
            return true;
        }

        let mut occ = self.occupied().xor_square(mv.from()).xor_square(sq);

        if mv.is_enpassant() {
            occ = occ.xor_square(mv.to().with_rank(mv.from().rank()));
        }

        let bishops = self.bishops() | self.queens();
        let rooks = self.rooks() | self.queens();

        let mut us = !self.side_to_move();

        let mut attackers = attacks::knight(sq) & self.knights()
            | attacks::king(sq) & self.kings()
            | attacks::pawn(Color::WHITE, sq) & self.pawns() & self.black()
            | attacks::pawn(Color::BLACK, sq) & self.pawns() & self.white()
            | attacks::bishop(sq, occ) & bishops
            | attacks::rook(sq, occ) & rooks;

        loop {
            let our_attackers = attackers & self.by_color(us);
            if our_attackers.is_empty() {
                break;
            }

            for pc in [
                Piece::PAWN,
                Piece::KNIGHT,
                Piece::BISHOP,
                Piece::ROOK,
                Piece::QUEEN,
                Piece::KING,
            ] {
                let board = our_attackers & self.by_piece(pc);
                if board.any() {
                    occ.toggle(board.into_iter().next().unwrap());
                    next = pc;
                    break;
                }
            }

            if [Piece::PAWN, Piece::BISHOP, Piece::QUEEN].contains(&next) {
                attackers |= attacks::bishop(sq, occ) & bishops;
            }
            if [Piece::ROOK, Piece::QUEEN].contains(&next) {
                attackers |= attacks::rook(sq, occ) & rooks;
            }

            attackers &= occ;
            score = -score - 1 - next.see_value();
            us = !us;

            if score >= 0 {
                if next == Piece::KING && (attackers & self.by_color(us)).any() {
                    us = !us;
                }
                break;
            }
        }

        self.side_to_move() != us
    }
}

fn get_scharnagl_back_rank(scharnagl: usize) -> [Piece; 8] {
    let mut back_rank = [Piece::PAWN; 8];

    let nth_empty = |back_rank: [Piece; 8], n: usize| {
        let mut n = n;
        for (i, sq) in back_rank.iter().enumerate() {
            if *sq == Piece::PAWN {
                if n == 0 {
                    return i;
                }
                n -= 1;
            }
        }
        unreachable!()
    };

    let n = scharnagl;

    let (n, b1) = (n / 4, n % 4);
    let (n, b2) = (n / 4, n % 4);
    let (n, q) = (n / 6, n % 6);

    let b1_file = match b1 {
        0 => File::B,
        1 => File::D,
        2 => File::F,
        3 => File::H,
        _ => unreachable!(),
    };

    let b2_file = match b2 {
        0 => File::A,
        1 => File::C,
        2 => File::E,
        3 => File::G,
        _ => unreachable!(),
    };

    back_rank[b1_file.index()] = Piece::BISHOP;
    back_rank[b2_file.index()] = Piece::BISHOP;

    back_rank[nth_empty(back_rank, q)] = Piece::QUEEN;

    let (n1, n2) = match n {
        0 => (0, 1),
        1 => (0, 2),
        2 => (0, 3),
        3 => (0, 4),
        4 => (1, 2),
        5 => (1, 3),
        6 => (1, 4),
        7 => (2, 3),
        8 => (2, 4),
        9 => (3, 4),
        _ => unreachable!(),
    };

    let n1 = nth_empty(back_rank, n1);
    let n2 = nth_empty(back_rank, n2);

    back_rank[n1] = Piece::KNIGHT;
    back_rank[n2] = Piece::KNIGHT;

    back_rank[nth_empty(back_rank, 0)] = Piece::ROOK;
    back_rank[nth_empty(back_rank, 0)] = Piece::KING;
    back_rank[nth_empty(back_rank, 0)] = Piece::ROOK;

    back_rank
}

#[cfg(test)]
mod test {
    use super::*;

    const STARTPOS_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

    #[test]
    fn test_scharnagl() {
        for i in 0..960 {
            let back_rank = get_scharnagl_back_rank(i);

            assert!(!back_rank.contains(&Piece::PAWN));
        }
    }

    #[test]
    fn test_startpos() {
        let board = Board::startpos();

        assert_eq!(board, Board::from_fen(STARTPOS_FEN));
    }

    #[test]
    fn test_see() {
        assert_see(
            true,
            "6k1/1pp4p/p1pb4/6q1/3P1pRr/2P4P/PP1Br1P1/5RKN w - -",
            "f1",
            "f4",
        );

        assert_see(
            true,
            "5rk1/1pp2q1p/p1pb4/8/3P1NP1/2P5/1P1BQ1P1/5RK1 b - - ",
            "d6",
            "f4",
        );

        assert_see(
            false,
            "6rr/6pk/p1Qp1b1p/2n5/1B3p2/5p2/P1P2P2/4RK1R w - -",
            "e1",
            "e8",
        );

        assert_see_promo(
            true,
            "7R/5P2/8/8/6r1/3K4/5p2/4k3 w - -",
            "f7",
            "f8",
            Piece::QUEEN,
        );

        assert_see(true, "8/8/1k6/8/8/2N1N3/4p1K1/3n4 w - -", "c3", "d1");

        assert_see(
            false,
            "2r1k3/pbr3pp/5p1b/2KB3n/1N2N3/3P1PB1/PPP1P1PP/R2Q3R w - -",
            "d5",
            "c6",
        );
    }

    fn assert_see(expected: bool, fen: &str, from: &str, to: &str) {
        assert_see_mv(
            expected,
            fen,
            Move::new(Square::from_uci(from), Square::from_uci(to)),
        );
    }

    fn assert_see_promo(expected: bool, fen: &str, from: &str, to: &str, promo: Piece) {
        assert_see_mv(
            expected,
            fen,
            Move::new_promotion(Square::from_uci(from), Square::from_uci(to), promo),
        );
    }

    fn assert_see_mv(expected: bool, fen: &str, mv: Move) {
        let board = Board::from_fen(fen);
        let result = Board::from_fen(fen).see(mv, -108);

        assert_eq!(result, expected);
    }
}

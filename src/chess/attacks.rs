use crate::chess::{Bitboard, Color, Square};

const NOT_A_FILE: u64 = 0xfefe_fefe_fefe_fefe;
const NOT_H_FILE: u64 = 0x7f7f_7f7f_7f7f_7f7f;

const fn init_pawn_attacks() -> [[Bitboard; 64]; 2] {
    let mut attacks = [[Bitboard::EMPTY; 64]; 2];
    let mut sq = 0;
    while sq < 64 {
        attacks[0][sq] =
            Bitboard::new((((1 << sq) & NOT_A_FILE) << 7) | (((1 << sq) & NOT_H_FILE) << 9));
        attacks[1][sq] =
            Bitboard::new((((1 << sq) & NOT_A_FILE) >> 9) | (((1 << sq) & NOT_H_FILE) >> 7));

        sq += 1;
    }
    attacks
}

const fn init_knight_attacks() -> [Bitboard; 64] {
    let mut attacks = [Bitboard::EMPTY; 64];
    let mut sq = 0;
    while sq < 64 {
        let n = 1 << sq;
        let h1 = ((n >> 1) & 0x7f7f_7f7f_7f7f_7f7f) | ((n << 1) & 0xfefe_fefe_fefe_fefe);
        let h2 = ((n >> 2) & 0x3f3f_3f3f_3f3f_3f3f) | ((n << 2) & 0xfcfc_fcfc_fcfc_fcfc);
        let bb = (h1 << 16) | (h1 >> 16) | (h2 << 8) | (h2 >> 8);

        attacks[sq] = Bitboard::new(bb);
        sq += 1;
    }
    attacks
}

const fn init_king_attacks() -> [Bitboard; 64] {
    let mut attacks = [Bitboard::EMPTY; 64];
    let mut sq = 0;
    while sq < 64 {
        let mut k = 1 << sq;
        k |= (k << 8) | (k >> 8);
        k |= ((k & NOT_A_FILE) >> 1) | ((k & NOT_H_FILE) << 1);
        k ^= 1 << sq;

        attacks[sq] = Bitboard::new(k);
        sq += 1;
    }
    attacks
}

const PAWN_ATTACKS: [[Bitboard; 64]; 2] = init_pawn_attacks();
const KNIGHT_ATTACKS: [Bitboard; 64] = init_knight_attacks();
const KING_ATTACKS: [Bitboard; 64] = init_king_attacks();

pub fn pawn(color: Color, square: Square) -> Bitboard {
    PAWN_ATTACKS[color.index()][square.index()]
}

pub fn knight(square: Square) -> Bitboard {
    KNIGHT_ATTACKS[square.index()]
}

pub fn king(square: Square) -> Bitboard {
    KING_ATTACKS[square.index()]
}

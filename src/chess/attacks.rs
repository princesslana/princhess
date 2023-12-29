use crate::chess::magic::{BISHOP_MAGICS, ROOK_MAGICS};
use crate::chess::{Bitboard, Color, Square};

const NOT_A_FILE: u64 = 0xfefe_fefe_fefe_fefe;
const NOT_H_FILE: u64 = 0x7f7f_7f7f_7f7f_7f7f;

const fn sliding_attacks(square: i32, occupied: u64, deltas: &[i32]) -> u64 {
    let mut attack = 0;

    let mut i = 0;
    let len = deltas.len();
    while i < len {
        let mut previous = square;
        loop {
            let sq = previous + deltas[i];
            let file_diff = (sq & 0x7) - (previous & 0x7);
            if file_diff > 2 || file_diff < -2 || sq < 0 || sq > 63 {
                break;
            }
            let bb = 1 << sq;
            attack |= bb;
            if occupied & bb != 0 {
                break;
            }
            previous = sq;
        }
        i += 1;
    }

    attack
}

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

#[allow(clippy::cast_sign_loss, clippy::large_stack_arrays)]
const fn init_sliding_attacks() -> [Bitboard; 88772] {
    let mut attacks = [Bitboard::EMPTY; 88772];
    let mut sq = 0;
    while sq < 64 {
        let magic = &BISHOP_MAGICS[sq as usize];
        let range = magic.mask;
        let mut subset = 0;
        loop {
            let attack = sliding_attacks(sq, subset, &[9, 7, -7, -9]);
            let idx = (magic.factor.wrapping_mul(subset) >> (64 - 9)) as usize + magic.offset;
            attacks[idx] = Bitboard::new(attack);
            subset = subset.wrapping_sub(range) & range;
            if subset == 0 {
                break;
            }
        }

        let magic = &ROOK_MAGICS[sq as usize];
        let range = magic.mask;
        let mut subset = 0;
        loop {
            let attack = sliding_attacks(sq, subset, &[8, 1, -1, -8]);
            let idx = (magic.factor.wrapping_mul(subset) >> (64 - 12)) as usize + magic.offset;
            attacks[idx] = Bitboard::new(attack);
            subset = subset.wrapping_sub(range) & range;
            if subset == 0 {
                break;
            }
        }

        sq += 1;
    }
    attacks
}

const PAWN_ATTACKS: [[Bitboard; 64]; 2] = init_pawn_attacks();
const KNIGHT_ATTACKS: [Bitboard; 64] = init_knight_attacks();
const KING_ATTACKS: [Bitboard; 64] = init_king_attacks();

static SLIDING_ATTACKS: [Bitboard; 88772] = init_sliding_attacks();

pub fn pawn(color: Color, square: Square) -> Bitboard {
    PAWN_ATTACKS[color.index()][square.index()]
}

pub fn knight(square: Square) -> Bitboard {
    KNIGHT_ATTACKS[square.index()]
}

pub fn bishop(square: Square, occupied: Bitboard) -> Bitboard {
    let magic = unsafe { BISHOP_MAGICS.get_unchecked(square.index()) };

    let idx =
        (magic.factor.wrapping_mul(occupied.0 & magic.mask) >> (64 - 9)) as usize + magic.offset;

    unsafe { *SLIDING_ATTACKS.get_unchecked(idx) }
}

pub fn rook(square: Square, occupied: Bitboard) -> Bitboard {
    let magic = unsafe { ROOK_MAGICS.get_unchecked(square.index()) };

    let idx =
        (magic.factor.wrapping_mul(occupied.0 & magic.mask) >> (64 - 12)) as usize + magic.offset;

    unsafe { *SLIDING_ATTACKS.get_unchecked(idx) }
}

pub fn king(square: Square) -> Bitboard {
    KING_ATTACKS[square.index()]
}

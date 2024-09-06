use crate::chess::magic::{BISHOP_MAGICS, ROOK_MAGICS};
use crate::chess::{Bitboard, Color, Piece, Square};

const A_FILE: u64 = 0x0101_0101_0101_0101;
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

const fn init_in_between() -> [[Bitboard; 64]; 64] {
    let mut arr = [[Bitboard::EMPTY; 64]; 64];
    let mut from = 0;
    while from < 64 {
        let mut to = 0;
        while to < 64 {
            arr[from][to] = in_between(from, to);
            to += 1;
        }
        from += 1;
    }
    arr
}

const fn in_between(sq1: usize, sq2: usize) -> Bitboard {
    const M1: u64 = 0xFFFF_FFFF_FFFF_FFFF;
    const A2A7: u64 = 0x0001_0101_0101_0100;
    const B2G7: u64 = 0x0040_2010_0804_0200;
    const H1B7: u64 = 0x0002_0408_1020_4080;
    let btwn = (M1 << sq1) ^ (M1 << sq2);
    let file = ((sq2 & 7).wrapping_add((sq1 & 7).wrapping_neg())) as u64;
    let rank = (((sq2 | 7).wrapping_sub(sq1)) >> 3) as u64;
    let mut line = ((file & 7).wrapping_sub(1)) & A2A7;
    line += 2 * ((rank & 7).wrapping_sub(1) >> 58);
    line += ((rank.wrapping_sub(file) & 15).wrapping_sub(1)) & B2G7;
    line += ((rank.wrapping_add(file) & 15).wrapping_sub(1)) & H1B7;
    line = line.wrapping_mul(btwn & btwn.wrapping_neg());
    Bitboard::new(line & btwn)
}

const fn init_line_through() -> [[Bitboard; 64]; 64] {
    let mut arr = [[Bitboard::EMPTY; 64]; 64];
    let mut from = 0;
    while from < 64 {
        let mut to = 0;
        while to < 64 {
            arr[from][to] = line_through(from, to);
            to += 1;
        }
        from += 1;
    }
    arr
}

const fn line_through(from: usize, to: usize) -> Bitboard {
    let sq = 1 << to;

    let rank = from / 8;
    let file = from & 7;

    let files = A_FILE << file;
    if files & sq > 0 {
        return Bitboard::new(files);
    }

    let ranks = 0xFF << (8 * rank);
    if ranks & sq > 0 {
        return Bitboard::new(ranks);
    }

    let diags = DIAGS[7 + file - rank];
    if diags & sq > 0 {
        return Bitboard::new(diags);
    }

    let antis = DIAGS[file + rank].swap_bytes();
    if antis & sq > 0 {
        return Bitboard::new(antis);
    }

    Bitboard::EMPTY
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

const IN_BETWEEN: [[Bitboard; 64]; 64] = init_in_between();
const LINE_THROUGH: [[Bitboard; 64]; 64] = init_line_through();

const DIAGS: [u64; 15] = [
    0x0100_0000_0000_0000,
    0x0201_0000_0000_0000,
    0x0402_0100_0000_0000,
    0x0804_0201_0000_0000,
    0x1008_0402_0100_0000,
    0x2010_0804_0201_0000,
    0x4020_1008_0402_0100,
    0x8040_2010_0804_0201,
    0x0080_4020_1008_0402,
    0x0000_8040_2010_0804,
    0x0000_0080_4020_1008,
    0x0000_0000_8040_2010,
    0x0000_0000_0080_4020,
    0x0000_0000_0000_8040,
    0x0000_0000_0000_0080,
];

pub fn for_piece(piece: Piece, color: Color, square: Square, occupied: Bitboard) -> Bitboard {
    match piece {
        Piece::PAWN => pawn(color, square),
        Piece::KNIGHT => knight(square),
        Piece::BISHOP => bishop(square, occupied),
        Piece::ROOK => rook(square, occupied),
        Piece::QUEEN => queen(square, occupied),
        Piece::KING => king(square),
        _ => unreachable!(),
    }
}

pub fn pawn(color: Color, square: Square) -> Bitboard {
    PAWN_ATTACKS[color][square]
}

pub fn knight(square: Square) -> Bitboard {
    KNIGHT_ATTACKS[square]
}

pub fn bishop(square: Square, occupied: Bitboard) -> Bitboard {
    let magic = BISHOP_MAGICS[square];

    let idx =
        (magic.factor.wrapping_mul(occupied.0 & magic.mask) >> (64 - 9)) as usize + magic.offset;

    unsafe { *SLIDING_ATTACKS.get_unchecked(idx) }
}

pub fn rook(square: Square, occupied: Bitboard) -> Bitboard {
    let magic = ROOK_MAGICS[square];

    let idx =
        (magic.factor.wrapping_mul(occupied.0 & magic.mask) >> (64 - 12)) as usize + magic.offset;

    unsafe { *SLIDING_ATTACKS.get_unchecked(idx) }
}

pub fn xray_bishop(square: Square, occupied: Bitboard, blockers: Bitboard) -> Bitboard {
    let attacks = bishop(square, occupied);
    attacks ^ bishop(square, occupied ^ (attacks & blockers))
}

pub fn xray_rook(square: Square, occupied: Bitboard, blockers: Bitboard) -> Bitboard {
    let attacks = rook(square, occupied);
    attacks ^ rook(square, occupied ^ (attacks & blockers))
}

pub fn queen(square: Square, occupied: Bitboard) -> Bitboard {
    bishop(square, occupied) | rook(square, occupied)
}

pub fn king(square: Square) -> Bitboard {
    KING_ATTACKS[square]
}

pub fn between(from: Square, to: Square) -> Bitboard {
    IN_BETWEEN[from][to]
}

pub fn through(from: Square, to: Square) -> Bitboard {
    LINE_THROUGH[from][to]
}

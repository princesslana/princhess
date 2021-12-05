use chess::*;
use features::FeatureVec;
use features_common::*;
use state::{Move, State};

include!(concat!(env!("OUT_DIR"), "/policy_feature_const.rs"));
const NUM_ADVS: usize = 5;
const NUM_ENCODED: usize = NUM_ROLES * NUM_ROLES * NUM_ADVS;
const ROLES: [Piece; NUM_ROLES] = [
    Piece::Pawn,
    Piece::Knight,
    Piece::Bishop,
    Piece::Rook,
    Piece::Queen,
    Piece::King,
];

fn encode_advantage(adv: i32) -> usize {
    assert_eq!(PAWN_P2 + 1, PAWN_P1);
    assert_eq!(PAWN_P2 + 2, PAWN_P0);
    assert_eq!(PAWN_P2 + 3, PAWN_N1);
    assert_eq!(PAWN_P2 + 4, PAWN_N2);
    match adv {
        x if x >= 2 => 0,
        1 => 1,
        0 => 2,
        -1 => 3,
        _ => 4,
    }
}

fn encode_axba(a: Piece, b: Piece, adv: i32) -> usize {
    let adv_i = encode_advantage(adv);
    let x = encode_pair(a, b);
    assert!(adv_i < NUM_ADVS);
    x * NUM_ADVS + adv_i
}

fn attacks(sq: Square, piece: Piece, color: Color, occ: BitBoard) -> BitBoard {
    match piece {
        Piece::Rook => get_rook_moves(sq, occ),
        Piece::Bishop => get_bishop_moves(sq, occ),
        Piece::Queen => get_rook_moves(sq, occ) ^ get_bishop_moves(sq, occ),
        Piece::King => get_king_moves(sq),
        Piece::Knight => get_knight_moves(sq),
        Piece::Pawn => get_pawn_attacks(sq, color, occ),
    }
}

fn rank_from_side(side: Color, sq: Square) -> usize {
    match side {
        Color::White => sq.get_rank().to_index(),
        Color::Black => 7 - sq.get_rank().to_index(),
    }
}

fn encode_urgency(piece: Piece, adv: i32) -> usize {
    let x = 0;
    let x = x * NUM_ROLES + piece as usize;
    let x = x * NUM_ADVS + encode_advantage(adv);
    x + PAWN_P2
}

fn foreach_feature<F>(state: &State, mov: &Move, mut f: F)
where
    F: FnMut(usize, i8),
{
    let mut f = |x| f(x, 1);
    let board = state.board();
    let our_role = board.piece_on(mov.get_source()).unwrap();
    let src_bb = BitBoard::from_square(mov.get_source());
    let dst_bb = BitBoard::from_square(mov.get_dest());
    let without_src = board.combined() & !src_bb;
    let occ = without_src & !dst_bb;
    let src_adv = get_advantage(state, without_src, mov.get_source());
    let dst_adv = get_advantage(state, occ, mov.get_dest());
    let taken = board.piece_on(mov.get_dest());
    f(encode_axba(our_role, taken.unwrap_or(Piece::King), dst_adv));
    f(encode_urgency(our_role, src_adv));
    let follow_ups = attacks(mov.get_dest(), our_role, board.side_to_move(), occ);
    let enemies = board.color_combined(!board.side_to_move());
    for &hit_role in &ROLES {
        if (follow_ups & enemies & board.pieces(hit_role)).0 != 0 {
            f(PAWN_HITS_PAWN + encode_pair(our_role, hit_role));
        }
    }
    if let Some(prev_move) = state.prev_move().as_ref() {
        if mov.get_dest() == prev_move.get_dest() {
            if let Some(captured) = state.prev_capture() {
                f(PAWN_RECAPTURES_PAWN + encode_pair(our_role, captured));
            } else {
                f(CAPTURES_LAST_MOVE);
            }
        }
        if mov.get_dest() == prev_move.get_source() {
            f(TAKES_OLD_PLACE);
        }
    }
    {
        let betw = between(mov.get_source(), mov.get_dest());
        let former = state.formerly_occupied();
        if (betw & former[0]).0 != 0 {
            f(CROSSES_FORMERLY_OCCUPIED_0);
        }
        if (betw & former[1]).0 != 0 {
            f(CROSSES_FORMERLY_OCCUPIED_1);
        }
        if (betw & former[2]).0 != 0 {
            f(CROSSES_FORMERLY_OCCUPIED_2);
        }
        if (betw & former[3]).0 != 0 {
            f(CROSSES_FORMERLY_OCCUPIED_3);
        }
        if (dst_bb & former[0]).0 != 0 {
            f(TAKES_FORMERLY_OCCUPIED_0);
        }
        if (dst_bb & former[1]).0 != 0 {
            f(TAKES_FORMERLY_OCCUPIED_1);
        }
        if (dst_bb & former[2]).0 != 0 {
            f(TAKES_FORMERLY_OCCUPIED_2);
        }
        if (dst_bb & former[3]).0 != 0 {
            f(TAKES_FORMERLY_OCCUPIED_3);
        }
        if (src_bb & former[0]).0 != 0 {
            f(FROM_FORMERLY_OCCUPIED_0);
        }
        if (src_bb & former[1]).0 != 0 {
            f(FROM_FORMERLY_OCCUPIED_1);
        }
        if (src_bb & former[2]).0 != 0 {
            f(FROM_FORMERLY_OCCUPIED_2);
        }
        if (src_bb & former[3]).0 != 0 {
            f(FROM_FORMERLY_OCCUPIED_3);
        }
    }
    let f_src = mov.get_source().get_file();
    let f_dst = mov.get_dest().get_file();
    if our_role == Piece::King && f_src != f_dst.left() && f_src != f_dst && f_src != f_dst.right()
    {
        if f_src.to_index() < f_dst.to_index() {
            f(CASTLE_LONG);
        } else {
            f(CASTLE_SHORT);
        }
    }
    if our_role == Piece::Pawn {
        let src_rank = rank_from_side(board.side_to_move(), mov.get_source());
        let dst_rank = rank_from_side(board.side_to_move(), mov.get_dest());
        f(PAWN_TO_RANK_1 + dst_rank);
        if dst_rank == src_rank + 2 {
            f(PAWN_DOUBLE_MOVE);
        }
        if mov.get_source().get_file() != mov.get_dest().get_file() && taken == None {
            f(EN_PASSANT);
        }
    }
    match mov.get_promotion() {
        Some(Piece::Queen) => f(PROMOTE_QUEEN),
        Some(Piece::Rook) => f(PROMOTE_ROOK),
        Some(Piece::Knight) => f(PROMOTE_KNIGHT),
        Some(Piece::Bishop) => f(PROMOTE_BISHOP),
        _ => (),
    };
    if (dst_bb & board.checkers()).0 != 0 {
        f(TAKES_CHECKER);
    }
    if (src_bb & board.pinned()).0 != 0 {
        f(MOVES_PINNED);
    }
}

pub fn featurize(state: &State, mov: &Move) -> FeatureVec {
    let mut arr = [0i8; NUM_POLICY_FEATURES];
    foreach_feature(state, mov, |i, v| {
        assert!(v == 1);
        arr[i] += v;
    });
    for &x in arr.iter() {
        assert!(x == 0 || x == 1);
    }
    FeatureVec { arr: arr.to_vec() }
}

pub fn evaluate_single(state: &State, mov: &Move) -> f32 {
    let mut result = 0f32;
    foreach_feature(state, mov, |i, _| {
        result += COEF[i];
    });
    result
}

pub fn evaluate_moves(state: &State, moves: &[Move]) -> Vec<f32> {
    let mut evalns: Vec<_> = moves.iter().map(|x| evaluate_single(state, x)).collect();
    softmax(&mut evalns);
    evalns
}

pub fn softmax(arr: &mut [f32]) {
    for x in arr.iter_mut() {
        *x = x.exp();
    }
    let s = 1.0 / arr.iter().sum::<f32>();
    for x in arr.iter_mut() {
        *x *= s;
    }
}

fn name_feature_uc(idx: usize) -> String {
    if idx >= NUM_ENCODED {
        INDEX_NAMES[idx - NUM_ENCODED].into()
    } else {
        for &r1 in &ROLES {
            for &r2 in &ROLES {
                for adv in (-2)..3 {
                    if encode_axba(r1, r2, adv) == idx {
                        return format!("{}_x_{}_{}", r1, r2, adv);
                    }
                }
            }
        }
        unreachable!()
    }
}

pub fn name_feature(idx: usize) -> String {
    name_feature_uc(idx).to_lowercase()
}

#[allow(clippy::many_single_char_names)]
fn get_advantage(state: &State, occ: BitBoard, to: Square) -> i32 {
    let board = state.board();

    let b = get_bishop_moves(to, occ);
    let r = get_rook_moves(to, occ);
    let n = get_knight_moves(to);
    let k = get_king_moves(to);
    let q = b ^ r;

    let b = b & board.pieces(Piece::Bishop);
    let r = r & board.pieces(Piece::Rook);
    let n = n & board.pieces(Piece::Knight);
    let k = k & board.pieces(Piece::King);
    let q = q & board.pieces(Piece::Queen);

    let atk = b ^ r ^ n ^ k ^ q;

    let turn = board.side_to_move();
    let fre = board.color_combined(turn);
    let ene = board.color_combined(!turn);

    let fp = get_pawn_attacks(to, !turn, occ) & board.pieces(Piece::Pawn) & fre;
    let ep = get_pawn_attacks(to, turn, occ) & board.pieces(Piece::Pawn) & ene;

    let fre_atk = (atk & fre) ^ fp;
    let ene_atk = (atk & ene) ^ ep;

    fre_atk.popcnt() as i32 - ene_atk.popcnt() as i32
}

#[cfg(test)]
#[test]
fn test_advantage() {
    let state = State::default();
    let occ = *state.board().combined();
    {
        let sq = Square::make_square(Rank::Sixth, File::C);
        let adv = get_advantage(&state, occ, sq);
        assert_eq!(adv, -3);
    }
    {
        let sq = Square::make_square(Rank::Third, File::C);
        let adv = get_advantage(&state, occ, sq);
        assert_eq!(adv, 3);
    }
    {
        let sq = Square::make_square(Rank::Second, File::E);
        let adv = get_advantage(&state, occ, sq);
        assert_eq!(adv, 4);
    }
}

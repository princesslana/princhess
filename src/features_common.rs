use chess::*;

pub const NUM_ROLES: usize = 6;

pub fn encode_pair(a: Piece, b: Piece) -> usize {
    let x = 0;
    let x = x * NUM_ROLES + a as usize;
    let x = x * NUM_ROLES + b as usize;
    x
}

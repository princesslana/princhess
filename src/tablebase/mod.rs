#![allow(
    dead_code,
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    clippy::all,
    clippy::pedantic
)]

#[cfg(feature = "fathom")]
mod bindings;

use crate::chess::{Board, Move};
#[cfg(feature = "fathom")]
use crate::chess::{Color, Piece, Square};

#[cfg(feature = "fathom")]
use std::ffi::CString;

pub enum Wdl {
    Win,
    Draw,
    Loss,
}

#[cfg(feature = "fathom")]
fn max_pieces() -> usize {
    unsafe { bindings::TB_LARGEST as usize }
}

#[cfg(not(feature = "fathom"))]
fn max_pieces() -> usize {
    0
}

#[cfg(feature = "fathom")]
pub fn set_tablebase_directory(paths: &str) -> Result<(), ()> {
    let c_paths = CString::new(paths).unwrap();

    unsafe {
        if bindings::tb_init(c_paths.as_ptr()) {
            Ok(())
        } else {
            Err(())
        }
    }
}

#[cfg(not(feature = "fathom"))]
pub fn set_tablebase_directory(_paths: &str) -> Result<(), ()> {
    Err(())
}

// Fathom uses 0 for no en passant square, which is safe as a1 can never be one
#[cfg(feature = "fathom")]
fn ep_square(b: &Board) -> u32 {
    match b.ep_square() {
        Square::NONE => 0,
        sq => sq.index() as u32,
    }
}

#[cfg(feature = "fathom")]
pub fn probe_wdl(b: &Board) -> Option<Wdl> {
    if b.occupied().count() > max_pieces() {
        return None;
    }

    if b.is_castling_rights() {
        return None;
    }

    unsafe {
        let wdl = bindings::tb_probe_wdl(
            b.white().0,
            b.black().0,
            b.kings().0,
            b.queens().0,
            b.rooks().0,
            b.bishops().0,
            b.knights().0,
            b.pawns().0,
            0,
            0,
            ep_square(b),
            b.side_to_move() == Color::WHITE,
        );

        match wdl {
            bindings::TB_WIN => Some(Wdl::Win),
            bindings::TB_LOSS => Some(Wdl::Loss),
            bindings::TB_DRAW | bindings::TB_CURSED_WIN | bindings::TB_BLESSED_LOSS => {
                Some(Wdl::Draw)
            }
            _ => None,
        }
    }
}

#[cfg(not(feature = "fathom"))]
pub fn probe_wdl(_b: &Board) -> Option<Wdl> {
    None
}

#[cfg(feature = "fathom")]
fn tb_move(b: &Board, mv: bindings::TbMove) -> Option<Move> {
    let from = Square::from((mv >> 6) & 0x3F);
    let to = Square::from(mv & 0x3F);
    let promotion = u32::from((mv >> 12) & 0x7);

    let promotion_role = match promotion {
        bindings::TB_PROMOTES_QUEEN => Piece::QUEEN,
        bindings::TB_PROMOTES_ROOK => Piece::ROOK,
        bindings::TB_PROMOTES_BISHOP => Piece::BISHOP,
        bindings::TB_PROMOTES_KNIGHT => Piece::KNIGHT,
        _ => Piece::NONE,
    };

    b.legal_moves()
        .into_iter()
        .find(|m| m.from() == from && m.to() == to && m.promotion() == promotion_role)
}

#[cfg(feature = "fathom")]
pub fn probe_root(b: &Board, has_repeated: bool) -> Option<Vec<(Move, i32)>> {
    if b.occupied().count() > max_pieces() {
        return None;
    }

    if b.is_castling_rights() {
        return None;
    }

    // SAFETY: TbRootMoves is plain integers, for which all-zero bytes are valid.
    let mut results = unsafe { Box::<bindings::TbRootMoves>::new_zeroed().assume_init() };

    let success = unsafe {
        bindings::tb_probe_root_dtz(
            b.white().0,
            b.black().0,
            b.kings().0,
            b.queens().0,
            b.rooks().0,
            b.bishops().0,
            b.knights().0,
            b.pawns().0,
            u32::from(b.halfmove_clock()),
            0,
            ep_square(b),
            b.side_to_move() == Color::WHITE,
            has_repeated,
            true,
            &mut *results,
        )
    };

    if success == 0 {
        return None;
    }

    let moves = results.moves[..results.size as usize]
        .iter()
        .filter_map(|m| Some((tb_move(b, m.move_)?, m.tbRank)))
        .collect();

    Some(moves)
}

#[cfg(not(feature = "fathom"))]
pub fn probe_root(_b: &Board, _has_repeated: bool) -> Option<Vec<(Move, i32)>> {
    None
}

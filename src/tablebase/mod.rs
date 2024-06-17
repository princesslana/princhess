#![allow(
    dead_code,
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    clippy::all,
    clippy::pedantic
)]
mod bindings;

use std::ffi::CString;
use std::ptr;

use crate::chess::{Board, Color, Move, Piece, Square};

pub enum Wdl {
    Win,
    Draw,
    Loss,
}

fn max_pieces() -> usize {
    unsafe { bindings::TB_LARGEST as usize }
}

pub fn set_tablebase_directory(paths: &str) {
    let c_paths = CString::new(paths).unwrap();

    unsafe {
        if bindings::tb_init(c_paths.as_ptr()) {
            println!("info string Success initializing tablebase at {}", paths);
        } else {
            println!("info string Error initializing tablebase at {}", paths);
        }
    }
}

pub fn probe_wdl(b: &Board) -> Option<Wdl> {
    if b.occupied().count() > max_pieces() {
        return None;
    }

    if b.is_castling_rights() || b.ep_square() != Square::NONE {
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
            0,
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

pub fn probe_best_move(b: &Board) -> Option<(Move, Wdl)> {
    if b.occupied().count() > max_pieces() {
        return None;
    }

    if b.is_castling_rights() || b.ep_square() != Square::NONE {
        return None;
    }

    unsafe {
        let root = bindings::tb_probe_root(
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
            0,
            b.side_to_move() == Color::WHITE,
            ptr::null_mut(),
        );

        if root == bindings::TB_RESULT_FAILED {
            return None;
        }

        let wdl = match (root & bindings::TB_RESULT_WDL_MASK) >> bindings::TB_RESULT_WDL_SHIFT {
            bindings::TB_WIN => Wdl::Win,
            bindings::TB_LOSS => Wdl::Loss,
            _ => Wdl::Draw,
        };

        let from =
            Square::from((root & bindings::TB_RESULT_FROM_MASK) >> bindings::TB_RESULT_FROM_SHIFT);
        let to = Square::from((root & bindings::TB_RESULT_TO_MASK) >> bindings::TB_RESULT_TO_SHIFT);
        let promotion =
            (root & bindings::TB_RESULT_PROMOTES_MASK) >> bindings::TB_RESULT_PROMOTES_SHIFT;

        let promotion_role = match promotion {
            bindings::TB_PROMOTES_QUEEN => Piece::QUEEN,
            bindings::TB_PROMOTES_ROOK => Piece::ROOK,
            bindings::TB_PROMOTES_BISHOP => Piece::BISHOP,
            bindings::TB_PROMOTES_KNIGHT => Piece::KNIGHT,
            _ => Piece::NONE,
        };

        for m in b.legal_moves() {
            if m.from() == from && m.to() == to && m.promotion() == promotion_role {
                return Some((m.into(), wdl));
            }
        }
    }
    None
}

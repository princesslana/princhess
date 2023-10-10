#![allow(
    dead_code,
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    clippy::all,
    clippy::pedantic
)]
mod bindings;

use shakmaty::{Chess, Color, Move, Position, Role, Setup, Square};
use std::ffi::CString;
use std::ptr;

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

pub fn probe_wdl(pos: &Chess) -> Option<Wdl> {
    let b = pos.board();

    if b.occupied().count() > max_pieces() {
        return None;
    }

    if pos.castling_rights().any() || pos.ep_square().is_some() {
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
            pos.turn() == Color::White,
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

pub fn probe_best_move(pos: &Chess) -> Option<Move> {
    let b = pos.board();

    if b.occupied().count() > max_pieces() {
        return None;
    }

    if pos.castling_rights().any() || pos.ep_square().is_some() {
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
            pos.turn() == Color::White,
            ptr::null_mut(),
        );

        if root == bindings::TB_RESULT_FAILED {
            return None;
        }

        let from =
            Square::new((root & bindings::TB_RESULT_FROM_MASK) >> bindings::TB_RESULT_FROM_SHIFT);
        let to = Square::new((root & bindings::TB_RESULT_TO_MASK) >> bindings::TB_RESULT_TO_SHIFT);
        let promotion =
            (root & bindings::TB_RESULT_PROMOTES_MASK) >> bindings::TB_RESULT_PROMOTES_SHIFT;

        let promotion_role = match promotion {
            bindings::TB_PROMOTES_QUEEN => Some(Role::Queen),
            bindings::TB_PROMOTES_ROOK => Some(Role::Rook),
            bindings::TB_PROMOTES_BISHOP => Some(Role::Bishop),
            bindings::TB_PROMOTES_KNIGHT => Some(Role::Knight),
            _ => None,
        };

        for m in pos.legal_moves() {
            if m.from() == Some(from) && m.to() == to && m.promotion() == promotion_role {
                return Some(m);
            }
        }
    }
    None
}

use std::ffi::{c_char, c_int, c_uint, CString};

use crate::chess::{Board, Color};
use crate::tablebase::Outcome;

/// The table entry for a DTZ probe.
pub enum DtzEntry {
    Value(i32),
    /// DTZ tables store one side to move; this position has the other.
    OtherSide,
}

/// Fathom's `Pos` from `tbchess.c`.
#[repr(C)]
struct Pos {
    white: u64,
    black: u64,
    kings: u64,
    queens: u64,
    rooks: u64,
    bishops: u64,
    knights: u64,
    pawns: u64,
    rule50: u8,
    ep: u8,
    turn: bool,
}

// Table types, Fathom's `enum { WDL, DTM, DTZ }`.
const WDL: c_int = 0;
const DTZ: c_int = 2;

unsafe extern "C" {
    /// Loads the tables found in a path list, replacing any loaded before.
    fn tb_init(path: *const c_char) -> bool;

    /// Pieces in the largest table loaded, 0 when none.
    static TB_LARGEST: c_uint;

    /// Reads one raw table entry (defined in `tbprobe.c`, no header declaration). `s` is the
    /// position's outcome for DTZ reads. Leaves `success` unchanged on a hit, sets it to 0 for a
    /// missing table and to -1 when a DTZ table stores the other side to move.
    fn probe_table(pos: *const Pos, s: c_int, success: *mut c_int, kind: c_int) -> c_int;
}

pub fn init(paths: &str) -> bool {
    let Ok(c_paths) = CString::new(paths) else {
        return false;
    };

    // SAFETY: `c_paths` is NUL-terminated and outlives the call.
    unsafe { tb_init(c_paths.as_ptr()) }
}

pub fn max_pieces() -> usize {
    // SAFETY: a plain read of an integer that only `tb_init` writes.
    unsafe { TB_LARGEST as usize }
}

pub fn probe_wdl_table(b: &Board) -> Option<Outcome> {
    match read(b, 0, WDL) {
        (_, 0) => None,
        (v, _) => outcome(v),
    }
}

pub fn probe_dtz_table(b: &Board, outcome: Outcome) -> Option<DtzEntry> {
    match read(b, table_value(outcome), DTZ) {
        (_, 0) => None,
        (_, s) if s < 0 => Some(DtzEntry::OtherSide),
        (v, _) => Some(DtzEntry::Value(v)),
    }
}

fn outcome(v: c_int) -> Option<Outcome> {
    match v {
        2 => Some(Outcome::Win),
        1 => Some(Outcome::CursedWin),
        0 => Some(Outcome::Draw),
        -1 => Some(Outcome::BlessedLoss),
        -2 => Some(Outcome::Loss),
        _ => None,
    }
}

fn table_value(outcome: Outcome) -> c_int {
    match outcome {
        Outcome::Win => 2,
        Outcome::CursedWin => 1,
        Outcome::Draw => 0,
        Outcome::BlessedLoss => -1,
        Outcome::Loss => -2,
    }
}

/// Returns the raw value and Fathom's success code.
fn read(b: &Board, s: c_int, kind: c_int) -> (c_int, c_int) {
    // A table entry depends only on the pieces and the side to move.
    let pos = Pos {
        white: b.white().0,
        black: b.black().0,
        kings: b.kings().0,
        queens: b.queens().0,
        rooks: b.rooks().0,
        bishops: b.bishops().0,
        knights: b.knights().0,
        pawns: b.pawns().0,
        rule50: 0,
        ep: 0,
        turn: b.side_to_move() == Color::WHITE,
    };

    let mut success = 1;

    // SAFETY: `pos` and `success` are valid for the call; `probe_table` only reads `pos` and
    // writes `success`. Table loading inside it is guarded by Fathom's own lock.
    let v = unsafe { probe_table(&raw const pos, s, &raw mut success, kind) };

    (v, success)
}

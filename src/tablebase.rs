use arc_swap::ArcSwap;
use log::debug;
use once_cell::sync::Lazy;
use shakmaty::{Chess, Move, Position};
use shakmaty_syzygy::{Tablebase, Wdl};
use std::path::Path;
use std::sync::Arc;

static TABLEBASE: Lazy<ArcSwap<Tablebase<Chess>>> =
    Lazy::new(|| ArcSwap::from_pointee(Tablebase::new()));

pub fn set_tablebase_directory<P: AsRef<Path>>(path: P) {
    let mut tb = Tablebase::new();
    let cnt = tb.add_directory(path).unwrap();
    debug!("Added {} files to tablebase.", cnt);
    TABLEBASE.store(Arc::new(tb));
}

pub fn probe_tablebase_wdl(pos: &Chess) -> Option<Wdl> {
    let tb = TABLEBASE.load();
    if pos.board().occupied().count() > tb.max_pieces() {
        None
    } else {
        tb.probe_wdl_after_zeroing(pos).ok()
    }
}

pub fn probe_tablebase_best_move(pos: &Chess) -> Option<Move> {
    let tb = TABLEBASE.load();
    if pos.board().occupied().count() > tb.max_pieces() {
        None
    } else {
        match tb.best_move(pos) {
            Ok(Some((m, _))) => Some(m),
            _ => None,
        }
    }
}

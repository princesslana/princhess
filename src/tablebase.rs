use arc_swap::ArcSwap;
use log::debug;
use once_cell::sync::Lazy;
use shakmaty::{Chess, Move, Setup};
use shakmaty_syzygy::{Syzygy, Tablebase, Wdl};
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
    if pos.board().occupied().count() < Chess::MAX_PIECES {
        return None;
    }

    TABLEBASE.load().get_wdl(pos).ok()
}

pub fn probe_tablebase_best_move(pos: &Chess) -> Option<Move> {
    if pos.board().occupied().count() < Chess::MAX_PIECES {
        return None;
    }

    match TABLEBASE.load().best_move(pos) {
        Ok(Some((m, _))) => Some(m),
        _ => None,
    }
}

use arc_swap::ArcSwap;
use atomics::*;
use log::debug;
use once_cell::sync::Lazy;
use shakmaty::{Chess, Move};
use shakmaty_syzygy::{Tablebase, Wdl};
use std::path::Path;
use std::sync::Arc;

static TABLEBASE: Lazy<ArcSwap<Tablebase<Chess>>> =
    Lazy::new(|| ArcSwap::from_pointee(Tablebase::new()));

static IS_SET: AtomicBool = AtomicBool::new(false);

pub fn set_tablebase_directory<P: AsRef<Path>>(path: P) {
    let mut tb = Tablebase::new();
    let cnt = tb.add_directory(path).unwrap();
    debug!("Added {} files to tablebase.", cnt);
    IS_SET.store(true, Ordering::Relaxed);
    TABLEBASE.store(Arc::new(tb));
}

pub fn probe_tablebase_wdl(pos: &Chess) -> Option<Wdl> {
    if !IS_SET.load(Ordering::Relaxed) {
        return None;
    }
    TABLEBASE.load().probe_wdl(pos).ok()
}

pub fn probe_tablebase_best_move(pos: &Chess) -> Option<Move> {
    match TABLEBASE.load().best_move(pos) {
        Ok(Some((m, _))) => Some(m),
        _ => None,
    }
}

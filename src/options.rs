use once_cell::sync::Lazy;
use std::cmp::max;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::RwLock;

static NUM_THREADS: AtomicUsize = AtomicUsize::new(1);
static HASH_SIZE_MB: AtomicUsize = AtomicUsize::new(16);

static CPUCT: Lazy<RwLock<f32>> = Lazy::new(|| RwLock::new(1.85));
static NEGAMAX_WEIGHT: Lazy<RwLock<f32>> = Lazy::new(|| RwLock::new(0.003));

static CHESS960: AtomicBool = AtomicBool::new(false);

pub fn set_num_threads(threads: usize) {
    NUM_THREADS.store(threads, Ordering::Relaxed);
}

pub fn get_num_threads() -> usize {
    max(1, NUM_THREADS.load(Ordering::Relaxed))
}

pub fn set_hash_size_mb(hs: usize) {
    HASH_SIZE_MB.store(hs, Ordering::Relaxed);
}

pub fn get_hash_size_mb() -> usize {
    max(1, HASH_SIZE_MB.load(Ordering::Relaxed))
}

pub fn set_cpuct(c: f32) {
    let mut cp = CPUCT.write().unwrap();
    *cp = c;
}

pub fn get_cpuct() -> f32 {
    let cp = CPUCT.read().unwrap();
    *cp
}

pub fn set_negamax_weight(w: f32) {
    let mut nw = NEGAMAX_WEIGHT.write().unwrap();
    *nw = w;
}

pub fn get_negamax_weight() -> f32 {
    let nw = NEGAMAX_WEIGHT.read().unwrap();
    *nw
}

pub fn set_chess960(c: bool) {
    CHESS960.store(c, Ordering::Relaxed);
}

pub fn is_chess960() -> bool {
    CHESS960.load(Ordering::Relaxed)
}

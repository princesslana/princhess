use atomics::{AtomicU64, AtomicUsize, Ordering};
use std::cmp::max;

static NUM_THREADS: AtomicUsize = AtomicUsize::new(1);
static HASH_SIZE_MB: AtomicUsize = AtomicUsize::new(16);

static EXPLORATION_CONSTANT: AtomicU64 = AtomicU64::new(200);

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

pub fn set_exploration_constant(c: usize) {
    EXPLORATION_CONSTANT.store(c, Ordering::Relaxed);
}

pub fn get_exploration_constant() -> usize {
    EXPLORATION_CONSTANT.load(Ordering::Relaxed)
}

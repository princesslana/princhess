use atomics::{AtomicUsize, Ordering};
use std::cmp::max;

static NUM_THREADS: AtomicUsize = AtomicUsize::new(1);

pub fn set_num_threads(threads: usize) {
    NUM_THREADS.store(threads, Ordering::Relaxed);
}

pub fn get_num_threads() -> usize {
    max(1, NUM_THREADS.load(Ordering::Relaxed))
}

use once_cell::sync::Lazy;
use std::cmp::max;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::RwLock;

static NUM_THREADS: AtomicUsize = AtomicUsize::new(1);
static HASH_SIZE_MB: AtomicUsize = AtomicUsize::new(16);

static CPUCT: Lazy<RwLock<f32>> = Lazy::new(|| RwLock::new(2.15));
static CPUCT_BASE: AtomicU64 = AtomicU64::new(18368);
static CPUCT_FACTOR: Lazy<RwLock<f32>> = Lazy::new(|| RwLock::new(2.82));

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
    *cp = c
}

pub fn get_cpuct() -> f32 {
    let cp = CPUCT.read().unwrap();
    *cp
}

pub fn set_cpuct_base(c: u64) {
    CPUCT_BASE.store(c, Ordering::Relaxed);
}

pub fn get_cpuct_base() -> u64 {
    CPUCT_BASE.load(Ordering::Relaxed)
}

pub fn set_cpuct_factor(c: f32) {
    let mut cf = CPUCT_FACTOR.write().unwrap();
    *cf = c;
}

pub fn get_cpuct_factor() -> f32 {
    let cf = CPUCT_FACTOR.read().unwrap();
    *cf
}

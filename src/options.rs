use once_cell::sync::Lazy;
use std::cmp::max;
use std::sync::atomic::{AtomicI32, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::RwLock;

static NUM_THREADS: AtomicUsize = AtomicUsize::new(1);
static HASH_SIZE_MB: AtomicUsize = AtomicUsize::new(16);

static CPUCT: Lazy<RwLock<f32>> = Lazy::new(|| RwLock::new(2.15));
static CPUCT_BASE: AtomicU64 = AtomicU64::new(18368);
static CPUCT_FACTOR: Lazy<RwLock<f32>> = Lazy::new(|| RwLock::new(2.82));

static MATE_SCORE: Lazy<RwLock<f32>> = Lazy::new(|| RwLock::new(1.1));

static POLICY_UPDATE_FREQUENCY: AtomicU32 = AtomicU32::new(100);
static POLICY_UPDATE_FACTOR: AtomicI32 = AtomicI32::new(1);

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

pub fn set_mate_score(m: f32) {
    let mut ms = MATE_SCORE.write().unwrap();
    *ms = m;
}

pub fn get_mate_score() -> f32 {
    let ms = MATE_SCORE.read().unwrap();
    *ms
}

pub fn set_policy_update_frequency(u: u32) {
    POLICY_UPDATE_FREQUENCY.store(u, Ordering::Relaxed);
}

pub fn get_policy_update_frequency() -> u32 {
    POLICY_UPDATE_FREQUENCY.load(Ordering::Relaxed)
}

pub fn set_policy_update_factor(f: i32) {
    POLICY_UPDATE_FACTOR.store(f, Ordering::Relaxed);
}

pub fn get_policy_update_factor() -> i32 {
    POLICY_UPDATE_FACTOR.load(Ordering::Relaxed)
}

use std::cmp::max;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};

static NUM_THREADS: AtomicUsize = AtomicUsize::new(1);
static HASH_SIZE_MB: AtomicUsize = AtomicUsize::new(16);

static CPUCT: AtomicU64 = AtomicU64::new(215);
static CPUCT_BASE: AtomicU64 = AtomicU64::new(18368);
static CPUCT_FACTOR: AtomicU64 = AtomicU64::new(282);

static POLICY_UPDATE_FREQUENCY: AtomicU32 = AtomicU32::new(100);

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

pub fn set_cpuct(c: u64) {
    CPUCT.store(c, Ordering::Relaxed);
}

pub fn get_cpuct() -> u64 {
    CPUCT.load(Ordering::Relaxed)
}

pub fn set_cpuct_base(c: u64) {
    CPUCT_BASE.store(c, Ordering::Relaxed);
}

pub fn get_cpuct_base() -> u64 {
    CPUCT_BASE.load(Ordering::Relaxed)
}

pub fn set_cpuct_factor(c: u64) {
    CPUCT_FACTOR.store(c, Ordering::Relaxed);
}

pub fn get_cpuct_factor() -> u64 {
    CPUCT_FACTOR.load(Ordering::Relaxed)
}

pub fn set_policy_update_frequency(u: u32) {
    POLICY_UPDATE_FREQUENCY.store(u, Ordering::Relaxed);
}

pub fn get_policy_update_frequency() -> u32 {
    POLICY_UPDATE_FREQUENCY.load(Ordering::Relaxed)
}

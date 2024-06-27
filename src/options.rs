use once_cell::sync::Lazy;
use std::cmp::max;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::RwLock;

static NUM_THREADS: AtomicUsize = AtomicUsize::new(1);
static HASH_SIZE_MB: AtomicUsize = AtomicUsize::new(16);
static MULTI_PV: AtomicUsize = AtomicUsize::new(1);

static CPUCT: Lazy<RwLock<f32>> = Lazy::new(|| RwLock::new(0.16));
static CPUCT_TAU: Lazy<RwLock<f32>> = Lazy::new(|| RwLock::new(0.84));
static CVISITS_SELECTION: Lazy<RwLock<f32>> = Lazy::new(|| RwLock::new(0.01));
static POLICY_TEMPERATURE: Lazy<RwLock<f32>> = Lazy::new(|| RwLock::new(1.23));
static POLICY_TEMPERATURE_ROOT: Lazy<RwLock<f32>> = Lazy::new(|| RwLock::new(14.5));

static CHESS960: AtomicBool = AtomicBool::new(false);
static POLICY_ONLY: AtomicBool = AtomicBool::new(false);

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

pub fn set_multi_pv(pv: usize) {
    MULTI_PV.store(pv, Ordering::Relaxed);
}

pub fn get_multi_pv() -> usize {
    max(1, MULTI_PV.load(Ordering::Relaxed))
}

pub fn set_cpuct(c: f32) {
    let mut cp = CPUCT.write().unwrap();
    *cp = c;
}

pub fn get_cpuct() -> f32 {
    let cp = CPUCT.read().unwrap();
    *cp
}

pub fn set_cpuct_tau(c: f32) {
    let mut cp = CPUCT_TAU.write().unwrap();
    *cp = c;
}

pub fn get_cpuct_tau() -> f32 {
    let cp = CPUCT_TAU.read().unwrap();
    *cp
}

pub fn set_cvisits_selection(c: f32) {
    let mut cv = CVISITS_SELECTION.write().unwrap();
    *cv = c;
}

pub fn get_cvisits_selection() -> f32 {
    let cv = CVISITS_SELECTION.read().unwrap();
    *cv
}

pub fn set_policy_temperature(t: f32) {
    let mut pt = POLICY_TEMPERATURE.write().unwrap();
    *pt = t;
}

pub fn get_policy_temperature() -> f32 {
    let pt = POLICY_TEMPERATURE.read().unwrap();
    *pt
}

pub fn set_policy_temperature_root(t: f32) {
    let mut pt = POLICY_TEMPERATURE_ROOT.write().unwrap();
    *pt = t;
}

pub fn get_policy_temperature_root() -> f32 {
    let pt = POLICY_TEMPERATURE_ROOT.read().unwrap();
    *pt
}

pub fn set_chess960(c: bool) {
    CHESS960.store(c, Ordering::Relaxed);
}

pub fn is_chess960() -> bool {
    CHESS960.load(Ordering::Relaxed)
}

pub fn set_policy_only(c: bool) {
    POLICY_ONLY.store(c, Ordering::Relaxed);
}

pub fn is_policy_only() -> bool {
    POLICY_ONLY.load(Ordering::Relaxed)
}

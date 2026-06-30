use std::thread;

#[must_use]
pub fn default_thread_count() -> u16 {
    thread::available_parallelism()
        .map(|n| n.get() as u16)
        .unwrap_or(1)
        .saturating_sub(2)
        .max(1)
}

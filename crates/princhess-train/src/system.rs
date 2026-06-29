use std::thread;

/// Get default thread count: all hardware threads - 2, minimum 1
///
/// Uses `std::thread::available_parallelism` to detect logical threads.
///
/// # Panics
/// Panics if thread count cannot be detected.
#[must_use]
pub fn default_thread_count() -> u16 {
    thread::available_parallelism()
        .map(|n| n.get() as u16)
        .expect("Failed to detect CPU count - specify --threads explicitly")
        .saturating_sub(2)
        .max(1)
}

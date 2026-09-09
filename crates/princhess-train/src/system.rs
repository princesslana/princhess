#[must_use]
pub fn default_thread_count() -> u16 {
    let physical = num_cpus::get_physical() as u16;
    physical.saturating_sub(1).max(1)
}

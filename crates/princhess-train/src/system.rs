#[must_use]
pub fn default_thread_count(include_virtual: bool) -> u16 {
    let count = if include_virtual {
        num_cpus::get()
    } else {
        num_cpus::get_physical().min(num_cpus::get())
    } as u16;
    count.saturating_sub(1).max(1)
}

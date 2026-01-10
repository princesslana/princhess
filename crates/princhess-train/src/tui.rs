use std::io::Result;

use crossterm::terminal;

pub struct RawModeGuard;

impl RawModeGuard {
    pub fn enable() -> Result<Self> {
        terminal::enable_raw_mode()?;
        Ok(Self)
    }
}

impl Drop for RawModeGuard {
    fn drop(&mut self) {
        terminal::disable_raw_mode().ok();
    }
}

#[must_use]
pub fn format_elapsed(seconds: u64) -> String {
    let hours = seconds / 3600;
    let minutes = (seconds % 3600) / 60;
    let secs = seconds % 60;
    format!("{hours:2}h {minutes:02}m {secs:02}s")
}

#[must_use]
pub fn format_eta(seconds: u64) -> String {
    let hours = seconds / 3600;
    let minutes = (seconds % 3600) / 60;
    format!("ETA: {hours:2}h {minutes:02}m")
}

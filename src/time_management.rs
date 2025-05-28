use std::time::{Duration, Instant};

use crate::chess::Color;
use crate::state::State;
use crate::uci::Tokens;

const DEFAULT_MOVE_TIME_SECS: u64 = 10;

const MOVE_OVERHEAD: Duration = Duration::from_millis(50);

#[must_use]
#[derive(Copy, Clone, Debug)]
pub struct TimeManagement {
    start: Instant,
    soft_limit: Option<Duration>,
    hard_limit: Option<Duration>,
    node_limit: usize,
}

impl Default for TimeManagement {
    fn default() -> Self {
        Self::from_duration(Duration::from_secs(DEFAULT_MOVE_TIME_SECS))
    }
}

impl TimeManagement {
    pub fn from_duration(d: Duration) -> Self {
        Self {
            start: Instant::now(),
            soft_limit: None,
            hard_limit: Some(d),
            node_limit: usize::MAX,
        }
    }

    pub fn from_limits(soft: Duration, hard: Duration) -> Self {
        Self {
            start: Instant::now(),
            soft_limit: Some(soft),
            hard_limit: Some(hard),
            node_limit: usize::MAX,
        }
    }

    pub fn infinite() -> Self {
        Self {
            start: Instant::now(),
            soft_limit: None,
            hard_limit: None,
            node_limit: usize::MAX,
        }
    }

    #[must_use]
    pub fn soft_limit(&self) -> Option<Duration> {
        self.soft_limit
    }

    #[must_use]
    pub fn hard_limit(&self) -> Option<Duration> {
        self.hard_limit
    }

    #[must_use]
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    #[must_use]
    pub fn node_limit(&self) -> usize {
        self.node_limit
    }

    pub fn set_node_limit(&mut self, node_limit: usize) {
        self.node_limit = node_limit;
    }

    fn parse_ms(tokens: &mut Tokens) -> Option<Duration> {
        tokens
            .next()
            .unwrap_or("")
            .parse()
            .ok()
            .map(Duration::from_millis)
    }

    pub fn from_tokens(mut tokens: Tokens, state: &State, is_policy_only: bool) -> Self {
        let stm = state.side_to_move();

        let mut infinite = false;
        let mut move_time = None;
        let mut increment = Duration::ZERO;
        let mut remaining = None;
        let mut movestogo: Option<u32> = None;
        let mut node_limit = usize::MAX;

        while let Some(s) = tokens.next() {
            match s {
                "infinite" => infinite = true,
                "movetime" => move_time = Self::parse_ms(&mut tokens),
                "wtime" => {
                    if stm == Color::WHITE {
                        remaining = Self::parse_ms(&mut tokens);
                    }
                }
                "btime" => {
                    if stm == Color::BLACK {
                        remaining = Self::parse_ms(&mut tokens);
                    }
                }
                "winc" => {
                    if stm == Color::WHITE {
                        increment = Self::parse_ms(&mut tokens).unwrap_or(Duration::ZERO);
                    }
                }
                "binc" => {
                    if stm == Color::BLACK {
                        increment = Self::parse_ms(&mut tokens).unwrap_or(Duration::ZERO);
                    }
                }
                "movestogo" => {
                    movestogo = tokens.next().unwrap_or("").parse().ok();
                }
                "nodes" => {
                    node_limit = tokens
                        .next()
                        .unwrap_or("")
                        .parse()
                        .ok()
                        .unwrap_or(usize::MAX);
                }
                _ => (),
            }
        }

        let mut think_time = TimeManagement::default();

        if infinite {
            think_time = TimeManagement::infinite();
        } else if let Some(mt) = move_time {
            think_time = TimeManagement::from_duration(mt);
        } else if let Some(r) = remaining {
            let mut move_time_fraction = u32::from(state.moves_left()) * 20 / 27;

            if let Some(m) = movestogo {
                move_time_fraction = (m + 2).min(move_time_fraction);
            }

            let r = r - MOVE_OVERHEAD;

            let soft_limit = (r + move_time_fraction * increment) / move_time_fraction;
            let hard_limit = r / 3;

            think_time = TimeManagement::from_limits(soft_limit.min(hard_limit), hard_limit);
        }

        if is_policy_only {
            node_limit = 1;
        }

        think_time.set_node_limit(node_limit);
        think_time
    }
}

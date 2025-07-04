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
            // 20/27 is a heuristic to scale moves_left, aiming for an average of ~20 moves
            let mut move_time_fraction = u32::from(state.moves_left()) * 20 / 27;

            if let Some(m) = movestogo {
                // If 'movestogo' is explicitly set, use it rather than moves_left
                move_time_fraction = (m + 2).min(move_time_fraction);
            }

            let r = r.saturating_sub(MOVE_OVERHEAD);

            // Soft limit: ideal/target time per move.
            let soft_limit = (r + move_time_fraction * increment) / move_time_fraction;

            // Hard limit: safety net or maximum time to spend on this move.
            let hard_limit = r / 3;

            // Ensure the soft limit is never more than the hard limit.
            think_time = TimeManagement::from_limits(soft_limit.min(hard_limit), hard_limit);
        }

        if is_policy_only {
            node_limit = 1;
        }

        think_time.set_node_limit(node_limit);
        think_time
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::State;

    // FEN for a position where Black is to move
    const BLACK_TO_MOVE_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1";

    // Helper to create tokens from a string
    fn create_tokens(s: &str) -> Tokens {
        s.split_whitespace()
    }

    // Test for "go infinite"
    #[test]
    fn test_from_tokens_infinite() {
        let state = State::default();
        let tokens = create_tokens("infinite");
        let tm = TimeManagement::from_tokens(tokens, &state, false);

        assert_eq!(tm.soft_limit(), None);
        assert_eq!(tm.hard_limit(), None);
        assert_eq!(tm.node_limit(), usize::MAX);
    }

    // Test for "go movetime <ms>"
    #[test]
    fn test_from_tokens_movetime() {
        let state = State::default();
        let tokens = create_tokens("movetime 10000"); // 10 seconds
        let tm = TimeManagement::from_tokens(tokens, &state, false);

        assert_eq!(tm.soft_limit(), None);
        assert_eq!(tm.hard_limit(), Some(Duration::from_secs(10)));
        assert_eq!(tm.node_limit(), usize::MAX);
    }

    // Test for "go nodes <n>"
    #[test]
    fn test_from_tokens_nodes() {
        let state = State::default();
        let tokens = create_tokens("nodes 100000");
        let tm = TimeManagement::from_tokens(tokens, &state, false);

        // Default time limits apply if not specified
        assert_eq!(tm.soft_limit(), None);
        assert_eq!(
            tm.hard_limit(),
            Some(Duration::from_secs(DEFAULT_MOVE_TIME_SECS))
        );
        assert_eq!(tm.node_limit(), 100_000);
    }

    // Test for "go wtime <ms> btime <ms>" (White to move)
    #[test]
    fn test_from_tokens_time_white_to_move() {
        let state = State::default(); // White to move by default, moves_left() = 43
        let tokens = create_tokens("wtime 60000 btime 50000"); // 60s for white, 50s for black
        let tm = TimeManagement::from_tokens(tokens, &state, false);

        let expected_remaining = Duration::from_millis(60000).saturating_sub(MOVE_OVERHEAD);
        let expected_move_time_fraction = u32::from(state.moves_left()) * 20 / 27; // 43 * 20 / 27 = 860 / 27 = 31
        let expected_soft_limit = (expected_remaining
            + expected_move_time_fraction * Duration::ZERO)
            / expected_move_time_fraction;
        let expected_hard_limit = expected_remaining / 3;

        assert_eq!(
            tm.soft_limit(),
            Some(expected_soft_limit.min(expected_hard_limit))
        );
        assert_eq!(tm.hard_limit(), Some(expected_hard_limit));
        assert_eq!(tm.node_limit(), usize::MAX);
    }

    // Test for "go wtime <ms> btime <ms>" (Black to move)
    #[test]
    fn test_from_tokens_time_black_to_move() {
        let state = State::from_fen(BLACK_TO_MOVE_FEN); // Set black to move
        let tokens = create_tokens("wtime 60000 btime 50000"); // 60s for white, 50s for black
        let tm = TimeManagement::from_tokens(tokens, &state, false);

        let expected_remaining = Duration::from_millis(50000).saturating_sub(MOVE_OVERHEAD);
        let expected_move_time_fraction = u32::from(state.moves_left()) * 20 / 27;
        let expected_soft_limit = (expected_remaining
            + expected_move_time_fraction * Duration::ZERO)
            / expected_move_time_fraction;
        let expected_hard_limit = expected_remaining / 3;

        assert_eq!(
            tm.soft_limit(),
            Some(expected_soft_limit.min(expected_hard_limit))
        );
        assert_eq!(tm.hard_limit(), Some(expected_hard_limit));
        assert_eq!(tm.node_limit(), usize::MAX);
    }

    // Test for "go wtime <ms> btime <ms> winc <inc> binc <inc>" (White to move)
    #[test]
    fn test_from_tokens_time_inc_white_to_move() {
        let state = State::default();
        let tokens = create_tokens("wtime 60000 btime 50000 winc 1000 binc 500"); // 1s inc for white
        let tm = TimeManagement::from_tokens(tokens, &state, false);

        let expected_remaining = Duration::from_millis(60000).saturating_sub(MOVE_OVERHEAD);
        let expected_increment = Duration::from_millis(1000);
        let expected_move_time_fraction = u32::from(state.moves_left()) * 20 / 27;
        let expected_soft_limit = (expected_remaining
            + expected_move_time_fraction * expected_increment)
            / expected_move_time_fraction;
        let expected_hard_limit = expected_remaining / 3;

        assert_eq!(
            tm.soft_limit(),
            Some(expected_soft_limit.min(expected_hard_limit))
        );
        assert_eq!(tm.hard_limit(), Some(expected_hard_limit));
        assert_eq!(tm.node_limit(), usize::MAX);
    }

    // Test for "go wtime <ms> btime <ms> movestogo <n>"
    #[test]
    fn test_from_tokens_movestogo() {
        let state = State::default(); // moves_left() = 43
        let tokens = create_tokens("wtime 60000 btime 50000 movestogo 40");
        let tm = TimeManagement::from_tokens(tokens, &state, false);

        let expected_remaining = Duration::from_millis(60000).saturating_sub(MOVE_OVERHEAD);
        // moves_left() = 43. 43 * 20 / 27 = 31.
        // movestogo (40) + 2 = 42.
        // min(42, 31) = 31.
        let expected_move_time_fraction = (40 + 2).min(u32::from(state.moves_left()) * 20 / 27);
        assert_eq!(expected_move_time_fraction, 31);

        let expected_soft_limit = (expected_remaining
            + expected_move_time_fraction * Duration::ZERO)
            / expected_move_time_fraction;
        let expected_hard_limit = expected_remaining / 3;

        assert_eq!(
            tm.soft_limit(),
            Some(expected_soft_limit.min(expected_hard_limit))
        );
        assert_eq!(tm.hard_limit(), Some(expected_hard_limit));
        assert_eq!(tm.node_limit(), usize::MAX);
    }

    // Test for is_policy_only
    #[test]
    fn test_from_tokens_is_policy_only() {
        let state = State::default();
        let tokens = create_tokens("infinite"); // Time control doesn't matter much here
        let tm = TimeManagement::from_tokens(tokens, &state, true); // is_policy_only = true

        assert_eq!(tm.node_limit(), 1);
        // Time limits should still be infinite as per "infinite" token
        assert_eq!(tm.soft_limit(), None);
        assert_eq!(tm.hard_limit(), None);
    }

    // Edge case: remaining time less than MOVE_OVERHEAD
    #[test]
    fn test_from_tokens_low_remaining_time() {
        let state = State::default();
        let tokens = create_tokens("wtime 10 btime 10"); // 10ms remaining
        let tm = TimeManagement::from_tokens(tokens, &state, false);

        let expected_remaining = Duration::from_millis(10).saturating_sub(MOVE_OVERHEAD); // Should be 0ms
        assert_eq!(expected_remaining, Duration::ZERO);

        // If remaining is 0, soft_limit and hard_limit calculations will result in 0
        let expected_soft_limit = Duration::ZERO;
        let expected_hard_limit = Duration::ZERO;

        assert_eq!(
            tm.soft_limit(),
            Some(expected_soft_limit.min(expected_hard_limit))
        );
        assert_eq!(tm.hard_limit(), Some(expected_hard_limit));
    }

    // Edge case: movestogo is 0
    #[test]
    fn test_from_tokens_movestogo_zero() {
        let state = State::default();
        let tokens = create_tokens("wtime 60000 btime 50000 movestogo 0");
        let tm = TimeManagement::from_tokens(tokens, &state, false);

        let expected_remaining = Duration::from_millis(60000).saturating_sub(MOVE_OVERHEAD);
        // (0 + 2) = 2. min(2, 31) = 2.
        let expected_move_time_fraction = 2.min(u32::from(state.moves_left()) * 20 / 27);
        assert_eq!(expected_move_time_fraction, 2);

        let expected_soft_limit = (expected_remaining
            + expected_move_time_fraction * Duration::ZERO)
            / expected_move_time_fraction;
        let expected_hard_limit = expected_remaining / 3;

        assert_eq!(
            tm.soft_limit(),
            Some(expected_soft_limit.min(expected_hard_limit))
        );
        assert_eq!(tm.hard_limit(), Some(expected_hard_limit));
    }

    // Test for missing time value (should fall back to default if parsing fails)
    #[test]
    fn test_from_tokens_missing_time_value() {
        let state = State::default();
        let tokens = create_tokens("wtime btime 50000"); // wtime has no value
        let tm = TimeManagement::from_tokens(tokens, &state, false);

        // Should fall back to default if wtime/btime parsing fails
        assert_eq!(tm.soft_limit(), None);
        assert_eq!(
            tm.hard_limit(),
            Some(Duration::from_secs(DEFAULT_MOVE_TIME_SECS))
        );
        assert_eq!(tm.node_limit(), usize::MAX);
    }

    // Test for missing nodes value (should default to usize::MAX)
    #[test]
    fn test_from_tokens_missing_nodes_value() {
        let state = State::default();
        let tokens = create_tokens("nodes"); // nodes has no value
        let tm = TimeManagement::from_tokens(tokens, &state, false);

        assert_eq!(tm.node_limit(), usize::MAX);
        // Default time limits apply
        assert_eq!(tm.soft_limit(), None);
        assert_eq!(
            tm.hard_limit(),
            Some(Duration::from_secs(DEFAULT_MOVE_TIME_SECS))
        );
    }

    // Test for combined time and nodes
    #[test]
    fn test_from_tokens_time_and_nodes() {
        let state = State::default();
        let tokens = create_tokens("wtime 60000 btime 50000 nodes 50000");
        let tm = TimeManagement::from_tokens(tokens, &state, false);

        let expected_remaining = Duration::from_millis(60000).saturating_sub(MOVE_OVERHEAD);
        let expected_move_time_fraction = u32::from(state.moves_left()) * 20 / 27;
        let expected_soft_limit = (expected_remaining
            + expected_move_time_fraction * Duration::ZERO)
            / expected_move_time_fraction;
        let expected_hard_limit = expected_remaining / 3;

        assert_eq!(
            tm.soft_limit(),
            Some(expected_soft_limit.min(expected_hard_limit))
        );
        assert_eq!(tm.hard_limit(), Some(expected_hard_limit));
        assert_eq!(tm.node_limit(), 50000);
    }

    // Test for very large time values (should not overflow)
    #[test]
    fn test_from_tokens_large_time_values() {
        let state = State::default();
        let tokens = create_tokens("wtime 3600000000 btime 3600000000 winc 100000 binc 100000"); // 1 hour for each, 100s inc
        let tm = TimeManagement::from_tokens(tokens, &state, false);

        let expected_remaining = Duration::from_millis(3_600_000_000).saturating_sub(MOVE_OVERHEAD);
        let expected_increment = Duration::from_millis(100_000);
        let expected_move_time_fraction = u32::from(state.moves_left()) * 20 / 27;

        let expected_soft_limit = (expected_remaining
            + expected_move_time_fraction * expected_increment)
            / expected_move_time_fraction;
        let expected_hard_limit = expected_remaining / 3;

        assert_eq!(
            tm.soft_limit(),
            Some(expected_soft_limit.min(expected_hard_limit))
        );
        assert_eq!(tm.hard_limit(), Some(expected_hard_limit));
    }

    // Test for negative btime when it's white to move
    #[test]
    fn test_from_tokens_negative_btime_white_to_move() {
        let state = State::default(); // White to move
        let tokens = create_tokens("wtime 60000 btime -1000"); // Negative btime
        let tm = TimeManagement::from_tokens(tokens, &state, false);

        // Since btime is negative and it's white to move, 'remaining' for white should still be parsed.
        // The negative btime token will be ignored for white's time.
        let expected_remaining = Duration::from_millis(60000).saturating_sub(MOVE_OVERHEAD);
        let expected_move_time_fraction = u32::from(state.moves_left()) * 20 / 27;
        let expected_soft_limit = (expected_remaining
            + expected_move_time_fraction * Duration::ZERO)
            / expected_move_time_fraction;
        let expected_hard_limit = expected_remaining / 3;

        assert_eq!(
            tm.soft_limit(),
            Some(expected_soft_limit.min(expected_hard_limit))
        );
        assert_eq!(tm.hard_limit(), Some(expected_hard_limit));
        assert_eq!(tm.node_limit(), usize::MAX);
    }

    // Test for negative wtime when it's black to move
    #[test]
    fn test_from_tokens_negative_wtime_black_to_move() {
        let state = State::from_fen(BLACK_TO_MOVE_FEN); // Black to move
        let tokens = create_tokens("wtime -1000 btime 60000"); // Negative wtime
        let tm = TimeManagement::from_tokens(tokens, &state, false);

        // Since wtime is negative and it's black to move, 'remaining' for black should still be parsed.
        // The negative wtime token will be ignored for black's time.
        let expected_remaining = Duration::from_millis(60000).saturating_sub(MOVE_OVERHEAD);
        let expected_move_time_fraction = u32::from(state.moves_left()) * 20 / 27;
        let expected_soft_limit = (expected_remaining
            + expected_move_time_fraction * Duration::ZERO)
            / expected_move_time_fraction;
        let expected_hard_limit = expected_remaining / 3;

        assert_eq!(
            tm.soft_limit(),
            Some(expected_soft_limit.min(expected_hard_limit))
        );
        assert_eq!(tm.hard_limit(), Some(expected_hard_limit));
        assert_eq!(tm.node_limit(), usize::MAX);
    }
}

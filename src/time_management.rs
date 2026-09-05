use std::time::{Duration, Instant};

use crate::chess::Color;
use crate::state::State;
use crate::uci::GoParams;

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

    pub fn from_go(params: &GoParams, state: &State, is_policy_only: bool) -> Self {
        let stm = state.side_to_move();

        let remaining = match stm {
            Color::WHITE => params.wtime,
            Color::BLACK => params.btime,
        };

        let increment = match stm {
            Color::WHITE => params.winc.unwrap_or(Duration::ZERO),
            Color::BLACK => params.binc.unwrap_or(Duration::ZERO),
        };

        let mut think_time = TimeManagement::default();

        if params.infinite {
            think_time = TimeManagement::infinite();
        } else if let Some(mt) = params.movetime {
            think_time = TimeManagement::from_duration(mt);
        } else if let Some(r) = remaining {
            // scale moves_left with 20/27 heuristic, clamp to 1 so late endgame doesn't div by 0 :)
            let mut move_time_fraction = (u32::from(state.moves_left()) * 20 / 27).max(1);

            if let Some(m) = params.movestogo {
                // movestogo is set, use that over moves_left
                move_time_fraction = m.saturating_add(2).min(move_time_fraction).max(1);
            }

            let r = r.saturating_sub(MOVE_OVERHEAD);

            // soft limit: ideal target time for this move
            let total_increment = increment
                .checked_mul(move_time_fraction)
                .unwrap_or(Duration::ZERO);
            let soft_limit = r.checked_add(total_increment).unwrap_or(r) / move_time_fraction;
            // hard limit: safety cap at 1/3 remaining time
            let hard_limit = r / 3;

            // soft limit shouldn't exceed hard limit
            think_time = TimeManagement::from_limits(soft_limit.min(hard_limit), hard_limit);
        }

        let node_limit = if is_policy_only {
            1
        } else {
            params.nodes.unwrap_or(usize::MAX)
        };

        think_time.set_node_limit(node_limit);
        think_time
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::State;
    use crate::uci::parser::{parse_go, Tokenizer};

    // fen for black to move position
    const BLACK_TO_MOVE_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1";

    fn parse_test_go(s: &str) -> GoParams {
        let mut tokens = Tokenizer::new(s);
        parse_go(&mut tokens)
    }

    // Test for "go infinite"
    #[test]
    fn test_from_go_infinite() {
        let state = State::default();
        let params = parse_test_go("infinite");
        let tm = TimeManagement::from_go(&params, &state, false);

        assert_eq!(tm.soft_limit(), None);
        assert_eq!(tm.hard_limit(), None);
        assert_eq!(tm.node_limit(), usize::MAX);
    }

    // Test for "go movetime <ms>"
    #[test]
    fn test_from_go_movetime() {
        let state = State::default();
        let params = parse_test_go("movetime 10000"); // 10 seconds
        let tm = TimeManagement::from_go(&params, &state, false);

        assert_eq!(tm.soft_limit(), None);
        assert_eq!(tm.hard_limit(), Some(Duration::from_secs(10)));
        assert_eq!(tm.node_limit(), usize::MAX);
    }

    // Test for "go nodes <n>"
    #[test]
    fn test_from_go_nodes() {
        let state = State::default();
        let params = parse_test_go("nodes 100000");
        let tm = TimeManagement::from_go(&params, &state, false);

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
    fn test_from_go_time_white_to_move() {
        let state = State::default(); // White to move by default, moves_left() = 43
        let params = parse_test_go("wtime 60000 btime 50000"); // 60s for white, 50s for black
        let tm = TimeManagement::from_go(&params, &state, false);

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
    fn test_from_go_time_black_to_move() {
        let state = State::from_fen(BLACK_TO_MOVE_FEN); // Set black to move
        let params = parse_test_go("wtime 60000 btime 50000"); // 60s for white, 50s for black
        let tm = TimeManagement::from_go(&params, &state, false);

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
    fn test_from_go_time_inc_white_to_move() {
        let state = State::default();
        let params = parse_test_go("wtime 60000 btime 50000 winc 1000 binc 500"); // 1s inc for white
        let tm = TimeManagement::from_go(&params, &state, false);

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
    fn test_from_go_movestogo() {
        let state = State::default(); // moves_left() = 43
        let params = parse_test_go("wtime 60000 btime 50000 movestogo 40");
        let tm = TimeManagement::from_go(&params, &state, false);

        let expected_remaining = Duration::from_millis(60000).saturating_sub(MOVE_OVERHEAD);
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
    fn test_from_go_is_policy_only() {
        let state = State::default();
        let params = parse_test_go("infinite"); // Time control doesn't matter much here
        let tm = TimeManagement::from_go(&params, &state, true); // is_policy_only = true

        assert_eq!(tm.node_limit(), 1);
        // Time limits should still be infinite as per "infinite"
        assert_eq!(tm.soft_limit(), None);
        assert_eq!(tm.hard_limit(), None);
    }

    // Edge case: remaining time less than MOVE_OVERHEAD
    #[test]
    fn test_from_go_low_remaining_time() {
        let state = State::default();
        let params = parse_test_go("wtime 10 btime 10"); // 10ms remaining
        let tm = TimeManagement::from_go(&params, &state, false);

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
    fn test_from_go_movestogo_zero() {
        let state = State::default();
        let params = parse_test_go("wtime 60000 btime 50000 movestogo 0");
        let tm = TimeManagement::from_go(&params, &state, false);

        let expected_remaining = Duration::from_millis(60000).saturating_sub(MOVE_OVERHEAD);
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
    fn test_from_go_missing_time_value() {
        let state = State::default();
        let params = parse_test_go("wtime btime 50000"); // wtime has no value
        let tm = TimeManagement::from_go(&params, &state, false);

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
    fn test_from_go_missing_nodes_value() {
        let state = State::default();
        let params = parse_test_go("nodes"); // nodes has no value
        let tm = TimeManagement::from_go(&params, &state, false);

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
    fn test_from_go_time_and_nodes() {
        let state = State::default();
        let params = parse_test_go("wtime 60000 btime 50000 nodes 50000");
        let tm = TimeManagement::from_go(&params, &state, false);

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
    fn test_from_go_large_time_values() {
        let state = State::default();
        let params = parse_test_go("wtime 3600000000 btime 3600000000 winc 100000 binc 100000"); // 1 hour for each, 100s inc
        let tm = TimeManagement::from_go(&params, &state, false);

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
    fn test_from_go_negative_btime_white_to_move() {
        let state = State::default(); // White to move
        let params = parse_test_go("wtime 60000 btime -1000"); // Negative btime
        let tm = TimeManagement::from_go(&params, &state, false);

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
    fn test_from_go_negative_wtime_black_to_move() {
        let state = State::from_fen(BLACK_TO_MOVE_FEN); // Black to move
        let params = parse_test_go("wtime -1000 btime 60000"); // Negative wtime
        let tm = TimeManagement::from_go(&params, &state, false);

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

    // late endgame positions with moves_left() <= 1 should never divide by 0
    #[test]
    fn test_from_go_late_endgame_moves_left_no_panic() {
        // high fullmove counter causes moves_left() to drop to 0 or 1
        let late_endgame_fen = "8/8/8/8/8/8/4k3/4K3 w - - 99 200";
        let state = State::from_fen(late_endgame_fen);
        assert!(state.moves_left() <= 1);

        let params = parse_test_go("wtime 5000");
        let tm = TimeManagement::from_go(&params, &state, false);

        assert!(tm.soft_limit().is_some());
        assert!(tm.hard_limit().is_some());
    }
}

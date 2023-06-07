use shakmaty::{CastlingMode, Color, Move};
use std::sync::mpsc::Sender;
use std::time::{Duration, Instant};

use crate::mcts::{AsyncSearchOwned, MctsManager};
use crate::options::{get_num_threads, is_chess960};
use crate::state::State;
use crate::tablebase::probe_tablebase_best_move;
use crate::transposition_table::TranspositionTable;
use crate::uci::Tokens;

const DEFAULT_MOVE_TIME_SECS: u64 = 10;
const DEFAULT_MOVE_TIME_FRACTION: u32 = 20;

const MOVE_OVERHEAD: Duration = Duration::from_millis(50);

pub const SCALE: f32 = 1e9;

#[derive(Copy, Clone, Debug)]
pub struct TimeManagement {
    start: Instant,
    end: Option<Instant>,
}

impl Default for TimeManagement {
    fn default() -> Self {
        Self::from_duration(Duration::from_secs(DEFAULT_MOVE_TIME_SECS))
    }
}

impl TimeManagement {
    pub fn from_duration(d: Duration) -> Self {
        let start = Instant::now();
        let end = Some(start + d);

        Self { start, end }
    }

    pub fn infinite() -> Self {
        Self {
            start: Instant::now(),
            end: None,
        }
    }

    pub fn is_after_end(&self) -> bool {
        if let Some(end) = self.end {
            Instant::now() > end
        } else {
            false
        }
    }

    pub fn elapsed(&self) -> Duration {
        Instant::now() - self.start
    }
}

pub struct Search {
    search: AsyncSearchOwned,
}

impl Search {
    pub fn create_manager(state: State, prev_table: TranspositionTable) -> MctsManager {
        MctsManager::new(state, TranspositionTable::empty(), prev_table)
    }

    pub fn new(state: State, prev_table: TranspositionTable) -> Self {
        let search = Self::create_manager(state, prev_table).into();
        Self { search }
    }

    pub fn table(self) -> TranspositionTable {
        let manager = self.stop_and_print_m();
        manager.table()
    }
    fn stop_and_print_m(self) -> MctsManager {
        if self.search.num_threads() == 0 {
            return self.search.halt();
        }
        let manager = self.search.halt();
        if let Some(mov) = manager.best_move() {
            println!("bestmove {}", to_uci(&mov));
        }
        manager
    }

    pub fn stop_and_print(self) -> Self {
        Self {
            search: self.stop_and_print_m().into(),
        }
    }

    fn parse_ms(tokens: &mut Tokens) -> Option<Duration> {
        tokens
            .next()
            .unwrap_or("")
            .parse()
            .ok()
            .map(Duration::from_millis)
    }

    pub fn go(self, mut tokens: Tokens, sender: &Sender<String>) -> Self {
        let manager = self.stop_and_print_m();

        let state = manager.tree().root_state();
        let stm = state.side_to_move();

        let mvs = state.available_moves();

        if mvs.len() == 1 {
            let uci_mv = to_uci(&mvs[0]);
            println!("info depth 1 seldepth 1 nodes 1 nps 1 tbhits 0 time 1 pv {uci_mv}");
            println!("bestmove {uci_mv}");
            return Self {
                search: manager.into(),
            };
        } else if let Some(mv) = probe_tablebase_best_move(state.board()) {
            let uci_mv = to_uci(&mv);
            println!("info depth 1 seldepth 1 nodes 1 nps 1 tbhits 1 time 1 pv {uci_mv}");
            println!("bestmove {uci_mv}");
            return Self {
                search: manager.into(),
            };
        }

        let mut move_time = None;
        let mut increment = Duration::ZERO;
        let mut infinite = false;
        let mut remaining = None;
        let mut movestogo: Option<u32> = None;

        while let Some(s) = tokens.next() {
            match s {
                "movetime" => move_time = Self::parse_ms(&mut tokens),
                "wtime" => {
                    if stm == Color::White {
                        remaining = Self::parse_ms(&mut tokens)
                    }
                }
                "btime" => {
                    if stm == Color::Black {
                        remaining = Self::parse_ms(&mut tokens)
                    }
                }
                "winc" => {
                    if stm == Color::White {
                        increment = Self::parse_ms(&mut tokens).unwrap_or(Duration::ZERO);
                    }
                }
                "binc" => {
                    if stm == Color::Black {
                        increment = Self::parse_ms(&mut tokens).unwrap_or(Duration::ZERO);
                    }
                }
                "infinite" => infinite = true,
                "movestogo" => {
                    movestogo = tokens.next().unwrap_or("").parse().ok();
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
            think_time =
                if movestogo.is_none() && increment.is_zero() && r < Duration::from_millis(60000) {
                    TimeManagement::from_duration(r / 60)
                } else {
                    let move_time_fraction = match movestogo {
                        // plus 2 because we want / 3 to be the max_think_time
                        Some(m) => (m + 2).min(DEFAULT_MOVE_TIME_FRACTION),
                        None => DEFAULT_MOVE_TIME_FRACTION,
                    };

                    let ideal_think_time =
                        (r + 20 * increment - MOVE_OVERHEAD) / move_time_fraction;
                    let max_think_time = r / 3;

                    TimeManagement::from_duration(ideal_think_time.min(max_think_time))
                }
        }

        Self {
            search: manager.into_playout_parallel_async(get_num_threads(), think_time, sender),
        }
    }

    pub fn print_move_list(&self) {
        self.search.get_manager().print_move_list();
    }
}

pub fn to_uci(mov: &Move) -> String {
    mov.to_uci(CastlingMode::from_chess960(is_chess960()))
        .to_string()
}

use mcts::{AsyncSearchOwned, Mcts, MctsManager, MoveInfoHandle};
use options::{get_cpuct, get_num_threads, is_chess960};
use shakmaty::{CastlingMode, Color, Move};
use state::State;
use std::sync::atomic::Ordering;
use std::sync::mpsc::Sender;
use std::thread;
use std::time::Duration;
use tablebase::probe_tablebase_best_move;
use transposition_table::TranspositionTable;
use tree_policy::AlphaGoPolicy;
use uci::Tokens;

const DEFAULT_MOVE_TIME_SECS: u64 = 10;
const DEFAULT_MOVE_TIME_FRACTION: u32 = 20;

const MOVE_OVERHEAD: Duration = Duration::from_millis(50);

pub const SCALE: f32 = 1e9;

fn policy() -> AlphaGoPolicy {
    let cpuct = get_cpuct();

    AlphaGoPolicy::new(cpuct * SCALE)
}

pub struct GooseMcts;

impl Mcts for GooseMcts {
    type TreePolicy = AlphaGoPolicy;

    fn virtual_loss(&self) -> i64 {
        SCALE as i64
    }
    fn select_child_after_search<'a>(&self, children: &[MoveInfoHandle<'a>]) -> MoveInfoHandle<'a> {
        *children
            .iter()
            .max_by_key(|child| {
                child
                    .average_reward()
                    .map(|r| r.round() as i64)
                    .unwrap_or(-SCALE as i64)
            })
            .unwrap()
    }
}

pub struct Search {
    search: AsyncSearchOwned<GooseMcts>,
}

impl Search {
    pub fn create_manager(state: State, prev_table: TranspositionTable) -> MctsManager<GooseMcts> {
        MctsManager::new(
            state,
            GooseMcts,
            policy(),
            TranspositionTable::empty(),
            prev_table,
        )
    }

    pub fn new(state: State, prev_table: TranspositionTable) -> Self {
        let search = Self::create_manager(state, prev_table).into();
        Self { search }
    }

    pub fn table(self) -> TranspositionTable {
        let manager = self.stop_and_print_m();
        manager.table()
    }
    fn stop_and_print_m(self) -> MctsManager<GooseMcts> {
        if self.search.num_threads() == 0 {
            return self.search.halt();
        }
        let manager = self.search.halt();
        if let Some(mov) = manager.best_move() {
            manager.print_info();
            println!("bestmove {}", to_uci(mov));
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
            let uci_mv = to_uci(mvs[0].clone());
            println!(
                "info depth 1 seldepth 1 nodes 1 nps 1 tbhits 0 time 1 pv {}",
                uci_mv
            );
            println!("bestmove {}", uci_mv);
            return Self {
                search: manager.into(),
            };
        } else if let Some(mv) = probe_tablebase_best_move(state.board()) {
            let uci_mv = mv.to_uci(CastlingMode::from_chess960(is_chess960()));
            println!(
                "info depth 1 seldepth 1 nodes 1 nps 1 tbhits 1 time 1 pv {}",
                uci_mv
            );
            println!("bestmove {}", uci_mv);
            return Self {
                search: manager.into(),
            };
        }

        let mut move_time = None;
        let mut increment = Duration::ZERO;
        let mut infinite = false;
        let mut remaining = None;

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
                _ => (),
            }
        }

        let mut think_time = Some(Duration::from_secs(DEFAULT_MOVE_TIME_SECS));

        if infinite {
            think_time = None
        } else if let Some(mt) = move_time {
            think_time = Some(mt)
        } else if let Some(r) = remaining {
            think_time = if increment.is_zero() && r < Duration::from_millis(60000) {
                Some(r / 60)
            } else {
                let ideal_think_time =
                    (r + 20 * increment - MOVE_OVERHEAD) / DEFAULT_MOVE_TIME_FRACTION;
                let max_think_time = r / 3;

                Some(ideal_think_time.min(max_think_time))
            }
        }

        let new_self = Self {
            search: manager.into_playout_parallel_async(get_num_threads(), sender),
        };

        if let Some(t) = think_time {
            let sender = sender.clone();
            let stop_signal = new_self.search.get_stop_signal().clone();
            thread::spawn(move || {
                thread::sleep(t);
                if !stop_signal.load(Ordering::Relaxed) {
                    let _ = sender.send("stop".to_string());
                }
            });
        }

        {
            let sender = sender.clone();
            let stop_signal = new_self.search.get_stop_signal().clone();
            thread::spawn(move || {
                thread::sleep(Duration::from_secs(1));
                while !stop_signal.load(Ordering::Relaxed) {
                    let _ = sender.send("info".to_string());
                    thread::sleep(Duration::from_secs(1));
                }
            });
        }

        new_self
    }

    pub fn print_info(&self) {
        self.search.get_manager().print_info();
    }

    pub fn print_move_list(&self) {
        self.search.get_manager().print_move_list();
    }

    pub fn print_features(&self) {
        let fs = self.search.get_manager().tree().root_state().features();

        let mut idx = 0;

        for _ in 0..12 {
            for _ in 0..8 {
                for _ in 0..8 {
                    print!("{}", i32::from(fs[idx] > 0.5));
                    idx += 1;
                }
                print!(" ");
            }
            println!()
        }
    }
}

pub fn to_uci(mov: Move) -> String {
    mov.to_uci(CastlingMode::from_chess960(is_chess960()))
        .to_string()
}

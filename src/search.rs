use bench::BENCHMARKING_POSITIONS;
use mcts::{AsyncSearchOwned, Mcts, MctsManager, MoveInfoHandle};
use options::{get_cpuct, get_num_threads};
use search_tree::{empty_previous_table, PreviousTable};
use shakmaty::{Color, Move};
use state::{State, StateBuilder};
use std::sync::atomic::Ordering;
use std::sync::mpsc::Sender;
use std::thread;
use std::time::{Duration, Instant};
use tablebase::probe_tablebase_best_move;
use transposition_table::ApproxTable;
use tree_policy::AlphaGoPolicy;
use uci::Tokens;

const DEFAULT_MOVE_TIME_SECS: u64 = 10;
const DEFAULT_MOVE_TIME_FRACTION: u32 = 15;

pub const SCALE: f32 = 1e9;

fn policy() -> AlphaGoPolicy {
    let cpuct = get_cpuct();

    AlphaGoPolicy::new(cpuct * SCALE)
}

pub struct GooseMcts;

impl Mcts for GooseMcts {
    type TreePolicy = AlphaGoPolicy;
    type TranspositionTable = ApproxTable;

    fn node_limit(&self) -> usize {
        4_000_000
    }
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
    pub fn create_manager(
        state: State,
        prev_table: PreviousTable<GooseMcts>,
    ) -> MctsManager<GooseMcts> {
        MctsManager::new(
            state,
            GooseMcts,
            policy(),
            ApproxTable::enough_to_hold(GooseMcts.node_limit()),
            prev_table,
        )
    }

    pub fn new(state: State, prev_table: PreviousTable<GooseMcts>) -> Self {
        let search = Self::create_manager(state, prev_table).into();
        Self { search }
    }

    pub fn table(self) -> PreviousTable<GooseMcts> {
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
            let uci_mv = mv.to_uci(shakmaty::CastlingMode::Standard);
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
        let mut infinite = false;
        let mut remaining = None;
        let mut sudden_death = true;

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
                        if let Some(inc) = Self::parse_ms(&mut tokens) {
                            sudden_death = inc.is_zero();
                        }
                    }
                }
                "binc" => {
                    if stm == Color::Black {
                        if let Some(inc) = Self::parse_ms(&mut tokens) {
                            sudden_death = inc.is_zero();
                        }
                    }
                }
                "movestogo" => {
                    sudden_death = false;
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
            let mut t = r / DEFAULT_MOVE_TIME_FRACTION;

            if sudden_death && r < Duration::from_millis(60000) {
                t = r / 60;
            }

            think_time = Some(t)
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

    pub fn bench(&self) {
        let bench_start = Instant::now();
        let mut nodes = 0;
        let mut search_time_ms = 0;
        for fen in BENCHMARKING_POSITIONS {
            let state: State = StateBuilder::from_fen(fen).unwrap().into();

            let manager = MctsManager::new(
                state,
                GooseMcts,
                policy(),
                ApproxTable::enough_to_hold(GooseMcts.node_limit()),
                empty_previous_table(),
            );

            manager.playout_sync();

            nodes += manager.nodes();
            search_time_ms = Instant::now().duration_since(bench_start).as_millis();

            println!(
                "info string {}/{}",
                nodes,
                nodes * 1000 / search_time_ms as usize
            );
        }

        println!("info nodes {}", nodes,);
        println!("info nps {}", nodes * 1000 / search_time_ms as usize)
    }
}

pub fn to_uci(mov: Move) -> String {
    mov.to_uci(shakmaty::CastlingMode::Standard).to_string()
}

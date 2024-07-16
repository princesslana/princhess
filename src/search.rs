use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crate::chess::{Color, Move};
use crate::evaluation;
use crate::options::{get_num_threads, get_policy_temperature, is_policy_only};
use crate::search_tree::{MoveEdge, PositionNode, SearchTree};
use crate::state::State;
use crate::tablebase;
use crate::transposition_table::{LRAllocator, LRTable};
use crate::uci::{read_stdin, Tokens};

const DEFAULT_MOVE_TIME_SECS: u64 = 10;
const DEFAULT_MOVE_TIME_FRACTION: u32 = 20;

const MOVE_OVERHEAD: Duration = Duration::from_millis(50);

pub const SCALE: f32 = 256. * 256.;

#[derive(Copy, Clone, Debug)]
pub struct TimeManagement {
    start: Instant,
    end: Option<Instant>,
    node_limit: usize,
}

impl Default for TimeManagement {
    fn default() -> Self {
        Self::from_duration(Duration::from_secs(DEFAULT_MOVE_TIME_SECS))
    }
}

impl TimeManagement {
    #[must_use]
    pub fn from_duration(d: Duration) -> Self {
        let start = Instant::now();
        let end = Some(start + d);
        let node_limit = usize::MAX;

        Self {
            start,
            end,
            node_limit,
        }
    }

    #[must_use]
    pub fn infinite() -> Self {
        let start = Instant::now();
        let end = None;
        let node_limit = usize::MAX;

        Self {
            start,
            end,
            node_limit,
        }
    }

    #[must_use]
    pub fn is_after_end(&self) -> bool {
        if let Some(end) = self.end {
            Instant::now() > end
        } else {
            false
        }
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
}

pub struct ThreadData<'a> {
    pub allocator: LRAllocator<'a>,
}

impl<'a> ThreadData<'a> {
    fn create(tree: &'a SearchTree) -> Self {
        Self {
            allocator: tree.allocator(),
        }
    }
}

pub struct Search {
    search_tree: SearchTree,
}

impl Search {
    pub fn new(state: State, table: LRTable) -> Self {
        let search_tree = SearchTree::new(state, table);
        Self { search_tree }
    }

    pub fn reset_table(&mut self) {
        self.search_tree.reset_table();
    }

    pub fn table(self) -> LRTable {
        self.search_tree.table()
    }

    pub fn root_node(&self) -> &PositionNode {
        self.search_tree.root_node()
    }

    pub fn tree(&self) -> &SearchTree {
        &self.search_tree
    }

    fn parse_ms(tokens: &mut Tokens) -> Option<Duration> {
        tokens
            .next()
            .unwrap_or("")
            .parse()
            .ok()
            .map(Duration::from_millis)
    }

    pub fn go(&self, mut tokens: Tokens, next_line: &mut Option<String>) {
        let state = self.search_tree.root_state();
        let stm = state.side_to_move();

        let mvs = state.available_moves();

        if mvs.len() == 1 {
            let uci_mv = mvs[0].to_uci();
            println!("info depth 1 seldepth 1 nodes 1 nps 1 tbhits 0 score cp 0 time 1 pv {uci_mv}");
            println!("bestmove {uci_mv}");
            return;
        } else if let Some((mv, wdl)) = tablebase::probe_best_move(state.board()) {
            let uci_mv = mv.to_uci();

            let score = match wdl {
                tablebase::Wdl::Win => 1000,
                tablebase::Wdl::Loss => -1000,
                tablebase::Wdl::Draw => 0,
            };
            println!(
                "info depth 1 seldepth 1 nodes 1 nps 1 tbhits 1 score cp {score} time 1 pv {uci_mv}"
            );
            println!("bestmove {uci_mv}");
            return;
        }

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
            let move_time_fraction = match movestogo {
                // plus 2 because we want / 3 to be the max_think_time
                Some(m) => (m + 2).min(DEFAULT_MOVE_TIME_FRACTION),
                None => DEFAULT_MOVE_TIME_FRACTION,
            };

            let ideal_think_time = (r + 20 * increment - MOVE_OVERHEAD) / move_time_fraction;
            let max_think_time = r / 3;

            think_time = TimeManagement::from_duration(ideal_think_time.min(max_think_time));
        }

        if is_policy_only() {
            node_limit = 1;
        }

        think_time.set_node_limit(node_limit);

        self.playout_parallel(get_num_threads(), think_time, next_line);
    }

    fn playout_parallel(
        &self,
        num_threads: usize,
        time_management: TimeManagement,
        next_line: &mut Option<String>,
    ) {
        let stop_signal = AtomicBool::new(false);

        let run_search_thread = |tm: &TimeManagement| {
            let mut tld = ThreadData::create(&self.search_tree);
            while self.search_tree.playout(&mut tld, tm, &stop_signal) {}
        };

        std::thread::scope(|s| {
            s.spawn(|| {
                run_search_thread(&time_management);
                self.search_tree.print_info(&time_management);
                stop_signal.store(true, Ordering::Relaxed);
                println!("bestmove {}", self.best_move().to_uci());
            });

            for _ in 0..(num_threads - 1) {
                s.spawn(|| {
                    run_search_thread(&TimeManagement::infinite());
                });
            }

            while !stop_signal.load(Ordering::Relaxed) {
                let line = read_stdin();

                *next_line = match line.trim() {
                    "stop" => {
                        stop_signal.store(true, Ordering::Relaxed);
                        None
                    }
                    "quit" => {
                        stop_signal.store(true, Ordering::Relaxed);
                        std::process::exit(0);
                    }
                    "isready" => {
                        println!("readyok");
                        None
                    }
                    _ => Some(line),
                }
            }
        });
    }

    pub fn playout_sync(&self, playouts: usize) {
        let mut tld = ThreadData::create(&self.search_tree);
        let tm = TimeManagement::infinite();
        let stop_signal = AtomicBool::new(false);

        for _ in 0..playouts {
            if !self.search_tree.playout(&mut tld, &tm, &stop_signal) {
                break;
            }
        }
    }

    pub fn print_move_list(&self) {
        let root_node = self.search_tree.root_node();
        let root_state = self.search_tree.root_state();

        let root_moves = root_node.hots();

        let state_moves = root_state.available_moves();
        let state_moves_eval =
            evaluation::policy(root_state, &state_moves, get_policy_temperature());

        let mut moves: Vec<(&MoveEdge, f32)> = root_moves.iter().zip(state_moves_eval).collect();
        moves.sort_by_key(|(h, e)| (h.average_reward().unwrap_or(*e) * SCALE) as i64);
        for (mov, e) in moves {
            println!(
                "info string {:7} M: {:5} P: {:>6} V: {:7} E: {:>6} ({:>8})",
                format!("{}", mov.get_move().to_uci()),
                format!("{:3.2}", e * 100.),
                format!("{:3.2}", f32::from(mov.policy()) / SCALE * 100.),
                mov.visits(),
                mov.average_reward()
                    .map_or("n/a".to_string(), |r| format!("{:3.2}", r / (SCALE / 100.))),
                mov.average_reward()
                    .map_or("n/a".to_string(), |r| eval_in_cp(r / SCALE))
            );
        }
    }

    pub fn print_eval(&self) {
        let eval = evaluation::value(self.search_tree.root_state());
        let scaled = eval as f32 / SCALE;
        println!(
            "info string eval {} scaled {} cp {}",
            eval,
            scaled,
            eval_in_cp(scaled)
        );
    }

    pub fn best_move(&self) -> Move {
        self.search_tree.best_move()
    }
}

// eval here is [-1.0, 1.0]
#[must_use]
pub fn eval_in_cp(eval: f32) -> String {
    let cps = if eval > 0.5 {
        18. * (eval - 0.5) + 1.
    } else if eval < -0.5 {
        18. * (eval + 0.5) - 1.
    } else {
        2. * eval
    };

    format!("cp {}", (cps * 100.).round().clamp(-1000., 1000.) as i64)
}

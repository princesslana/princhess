use scoped_threadpool::Pool;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crate::chess::{Color, Move};
use crate::evaluation;
use crate::math::Rng;
use crate::options::SearchOptions;
use crate::search_tree::{MoveEdge, PositionNode, SearchTree};
use crate::state::State;
use crate::tablebase;
use crate::transposition_table::{LRAllocator, LRTable};
use crate::uci::{read_stdin, Tokens};

const DEFAULT_MOVE_TIME_SECS: u64 = 10;

const MOVE_OVERHEAD: Duration = Duration::from_millis(50);

pub const SCALE: f32 = 256. * 256.;

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
}

pub struct ThreadData<'a> {
    pub ttable: &'a LRTable,
    pub allocator: LRAllocator<'a>,
    pub playouts: usize,
}

impl<'a> ThreadData<'a> {
    fn create(ttable: &'a LRTable) -> Self {
        Self {
            ttable,
            allocator: ttable.allocator(),
            playouts: 0,
        }
    }
}

#[allow(clippy::struct_field_names)]
#[must_use]
pub struct Search {
    search_tree: SearchTree,
    search_options: SearchOptions,
    ttable: LRTable,
}

impl Search {
    pub fn new(state: State, search_options: SearchOptions) -> Self {
        let ttable = LRTable::empty(search_options.hash_size_mb);
        let search_tree = SearchTree::new(state, &ttable, search_options);
        Self {
            search_tree,
            search_options,
            ttable,
        }
    }

    pub fn set_root_state(&mut self, state: State) {
        self.search_tree = SearchTree::new(state, &self.ttable, self.search_options);
    }

    pub fn root_node(&self) -> &PositionNode {
        self.search_tree.root_node()
    }

    pub fn root_state(&self) -> &State {
        self.search_tree.root_state()
    }

    pub fn tree(&self) -> &SearchTree {
        &self.search_tree
    }

    pub fn table_capacity_remaining(&self) -> usize {
        self.ttable.capacity_remaining()
    }

    pub fn flip_table(&self) {
        self.ttable.flip(|| self.root_node().clear_children_links());
    }

    fn parse_ms(tokens: &mut Tokens) -> Option<Duration> {
        tokens
            .next()
            .unwrap_or("")
            .parse()
            .ok()
            .map(Duration::from_millis)
    }

    pub fn go(&self, threads: &mut Pool, mut tokens: Tokens, next_line: &mut Option<String>) {
        let state = self.search_tree.root_state();
        let stm = state.side_to_move();

        let mvs = state.available_moves();

        if mvs.len() == 1 {
            let uci_mv = self.to_uci(mvs[0]);
            println!(
                "info depth 1 seldepth 1 nodes 1 nps 1 tbhits 0 score cp 0 time 1 pv {uci_mv}"
            );
            println!("bestmove {uci_mv}");
            return;
        } else if let Some((mv, wdl)) = tablebase::probe_best_move(state.board()) {
            let uci_mv = self.to_uci(mv);

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
            let mut move_time_fraction = u32::from(state.moves_left()) * 20 / 27;

            if let Some(m) = movestogo {
                move_time_fraction = (m + 2).min(move_time_fraction);
            }

            let r = r - MOVE_OVERHEAD;

            let soft_limit = (r + move_time_fraction * increment) / move_time_fraction;
            let hard_limit = r / 3;

            think_time = TimeManagement::from_limits(soft_limit.min(hard_limit), hard_limit);
        }

        if self.search_options.is_policy_only {
            node_limit = 1;
        }

        think_time.set_node_limit(node_limit);

        self.playout_parallel(threads, think_time, next_line);
    }

    fn playout_parallel(
        &self,
        threads: &mut Pool,
        time_management: TimeManagement,
        next_line: &mut Option<String>,
    ) {
        let cpuct = self.search_options.mcts_options.cpuct;
        let stop_signal = AtomicBool::new(false);
        let thread_count = threads.thread_count();

        if self.table_capacity_remaining() < threads.thread_count() as usize {
            println!(
                "info string not enough table capacity for {} threads, flipping table",
                threads.thread_count()
            );
            self.flip_table();
        }

        let mut rng = Rng::default();

        let run_search_thread = |cpuct: f32, tm: &TimeManagement| {
            let mut tld = ThreadData::create(&self.ttable);
            while self.search_tree.playout(&mut tld, cpuct, tm, &stop_signal) {}
        };

        threads.scoped(|s| {
            s.execute(|| {
                run_search_thread(cpuct, &time_management);
                self.search_tree
                    .print_info(&time_management, self.ttable.full());
                stop_signal.store(true, Ordering::Relaxed);
                println!("bestmove {}", self.to_uci(self.best_move()),);
            });

            for _ in 0..(thread_count - 1) {
                let jitter = 1. + rng.next_f32_range(-0.03, 0.03);

                s.execute(move || {
                    run_search_thread(cpuct * jitter, &TimeManagement::infinite());
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

    pub fn playout_sync(&self, playouts: u64) {
        let mut tld = ThreadData::create(&self.ttable);
        let cpuct = self.search_options.mcts_options.cpuct;
        let tm = TimeManagement::infinite();
        let stop_signal = AtomicBool::new(false);

        for _ in 0..playouts {
            if !self.search_tree.playout(&mut tld, cpuct, &tm, &stop_signal) {
                break;
            }
        }
    }

    pub fn print_move_list(&self) {
        let root_node = self.search_tree.root_node();
        let root_state = self.search_tree.root_state();

        let root_moves = root_node.hots();

        let state_moves = root_state.available_moves();
        let state_moves_eval = evaluation::policy(
            root_state,
            &state_moves,
            self.search_options.mcts_options.policy_temperature,
        );

        let mut moves: Vec<(&MoveEdge, f32)> = root_moves.iter().zip(state_moves_eval).collect();
        moves.sort_by_key(|(h, e)| (h.reward().average, (e * SCALE) as i64));
        for (mov, e) in moves {
            let reward = mov.reward();

            println!(
                "info string {:7} M: {:>5.2} P: {:>5.2} V: {:7} ({:>5.2}%) E: {:>7.2} ({:>8})",
                self.to_uci(*mov.get_move()),
                e * 100.,
                f32::from(mov.policy()) / SCALE * 100.,
                mov.visits(),
                mov.visits() as f32 / root_node.visits() as f32 * 100.,
                reward.average as f32 / (SCALE / 100.),
                eval_in_cp(reward.average as f32 / SCALE)
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

    pub fn most_visited_move(&self) -> Move {
        *self
            .search_tree
            .root_node()
            .select_child_by_visits()
            .get_move()
    }

    fn to_uci(&self, mov: Move) -> String {
        mov.to_uci(self.search_options.is_chess960)
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

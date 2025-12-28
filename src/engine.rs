use std::sync::atomic::{AtomicBool, Ordering};

use arrayvec::ArrayVec;

use crate::chess::Move;
use crate::evaluation;
use crate::graph::{select_edge_by_visits, MoveEdge, PositionNode, Reward};
use crate::math::{self, Rng};
use crate::mcts::{self, Mcts};
use crate::options::{EngineOptions, MctsOptions};
use crate::state::State;
use crate::tablebase;
use crate::threadpool::{Scope, ThreadPool};
use crate::time_management::TimeManagement;
use crate::transposition_table::{LRAllocator, LRTable};
use crate::uci::{read_stdin, Tokens};

pub const SCALE: f32 = 256. * 256.;

/// Thread-local buffer for root edge statistics to reduce atomic contention.
#[derive(Clone, Copy)]
pub struct RootEdge {
    synced_visits: u32,
    synced_sum_evaluations: i64,
    delta_visits: u32,
    delta_sum_evaluations: i64,
}

impl Default for RootEdge {
    fn default() -> Self {
        Self::new()
    }
}

impl RootEdge {
    pub const fn new() -> Self {
        Self {
            synced_visits: 0,
            synced_sum_evaluations: 0,
            delta_visits: 0,
            delta_sum_evaluations: 0,
        }
    }

    pub fn from_edge(edge: &MoveEdge) -> Self {
        Self {
            synced_visits: edge.visits(),
            synced_sum_evaluations: edge.sum_evaluations(),
            delta_visits: 0,
            delta_sum_evaluations: 0,
        }
    }

    pub fn from_edges(edges: &[MoveEdge]) -> ArrayVec<RootEdge, 256> {
        let mut root_edges = ArrayVec::new();
        for edge in edges {
            root_edges.push(RootEdge::from_edge(edge));
        }
        root_edges
    }

    pub fn clear(&mut self) {
        self.delta_visits = 0;
        self.delta_sum_evaluations = 0;
    }

    pub fn visits(&self) -> u32 {
        self.synced_visits + self.delta_visits
    }

    pub fn reward(&self) -> Reward {
        let visits = self.visits();

        if visits == 0 {
            return Reward::ZERO;
        }

        let sum = self.synced_sum_evaluations + self.delta_sum_evaluations;

        Reward {
            average: sum / i64::from(visits),
            visits,
        }
    }

    pub fn down(&mut self) {
        self.delta_visits += 1;
    }

    pub fn up(&mut self, evaln: i64) {
        self.delta_sum_evaluations += evaln;
    }

    pub fn flush(&mut self, edge: &MoveEdge) {
        self.synced_visits = edge.add_visits(self.delta_visits);
        self.synced_sum_evaluations = edge.add_sum_evaluations(self.delta_sum_evaluations);
        self.clear();
    }
}

pub struct ThreadData<'a> {
    pub ttable: &'a LRTable,
    pub allocator: LRAllocator<'a>,
    pub playouts: usize,
    pub thread_id: usize,
    pub num_nodes: usize,
    pub max_depth: usize,
    pub tb_hits: usize,
    pub root_edges: ArrayVec<RootEdge, 256>,
    pub root_gini: f32,
}

impl<'a> ThreadData<'a> {
    fn create(
        ttable: &'a LRTable,
        thread_id: usize,
        root_edges: ArrayVec<RootEdge, 256>,
        root_gini: f32,
    ) -> Self {
        Self {
            ttable,
            allocator: ttable.allocator(),
            playouts: 0,
            thread_id,
            num_nodes: 0,
            max_depth: 0,
            tb_hits: 0,
            root_edges,
            root_gini,
        }
    }

    pub fn is_main_thread(&self) -> bool {
        self.thread_id == 0
    }
}

#[allow(clippy::struct_field_names)]
#[must_use]
pub struct Engine {
    mcts: Mcts,
    engine_options: EngineOptions,
    ttable: LRTable,
    threads: ThreadPool,
}

impl Engine {
    pub fn new(state: State, engine_options: EngineOptions) -> Self {
        let ttable = LRTable::empty(engine_options.hash_size_mb);
        let mcts = Mcts::new(state, &ttable, engine_options);
        let threads = ThreadPool::new(engine_options.threads as usize);
        Self {
            mcts,
            engine_options,
            ttable,
            threads,
        }
    }

    pub fn set_root_state(&mut self, state: State) {
        self.mcts = Mcts::new(state, &self.ttable, self.engine_options);
    }

    pub fn root_edges(&self) -> &[MoveEdge] {
        self.mcts.root_edges()
    }

    pub fn root_state(&self) -> &State {
        self.mcts.root_state()
    }

    pub fn mcts(&self) -> &Mcts {
        &self.mcts
    }

    pub fn table_full(&self) -> usize {
        self.ttable.full()
    }

    pub fn table_capacity_remaining(&self) -> usize {
        self.ttable.capacity_remaining()
    }

    pub fn flip_table(&self) {
        self.ttable.flip(|| self.mcts.clear_root_children_links());
    }

    pub fn go(&self, tokens: Tokens, is_interactive: bool) -> Option<String> {
        let state = self.mcts.root_state();
        let mvs = state.available_moves();

        if mvs.len() == 1 {
            let uci_mv = self.to_uci(mvs[0]);
            println!(
                "info depth 1 seldepth 1 nodes 1 nps 1 tbhits 0 score cp 0 time 1 pv {uci_mv}"
            );
            println!("bestmove {uci_mv}");
            return None;
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
            return None;
        }

        let think_time =
            TimeManagement::from_tokens(tokens, state, self.engine_options.is_policy_only);

        self.playout_parallel(think_time, is_interactive)
    }

    fn playout_parallel(
        &self,
        time_management: TimeManagement,
        is_interactive: bool,
    ) -> Option<String> {
        let stop_signal = AtomicBool::new(false);
        let thread_count = self.threads.current_num_threads();

        if self.table_capacity_remaining() < thread_count {
            println!(
                "info string not enough table capacity for {thread_count} threads, flipping table"
            );
            self.flip_table();
        }

        let mut rng = Rng::default();
        let mut returned_line: Option<String> = None;
        let root_edges = RootEdge::from_edges(self.mcts.root_edges());

        let run_search_thread = |options: &MctsOptions, tm: &TimeManagement, thread_id: usize| {
            let mut tld = ThreadData::create(&self.ttable, thread_id, root_edges.clone(), 0.0);
            while self.mcts.playout(&mut tld, options, tm, &stop_signal) {}
            self.mcts.flush_root_edges(&mut tld);
            self.mcts.flush_thread_stats(&mut tld);
        };

        self.threads.scope(|s: &Scope| {
            let base_options = &self.engine_options.mcts_options;

            s.spawn(|| {
                run_search_thread(base_options, &time_management, 0);
                self.mcts.print_info(&time_management, self.ttable.full());
                stop_signal.store(true, Ordering::Relaxed);
                println!("bestmove {}", self.to_uci(self.best_move()));
            });

            for thread_id in 1..thread_count {
                let jitter_range = self.engine_options.mcts_options.cpuct_jitter;
                let jitter = 1. + rng.next_f32_range(-jitter_range, jitter_range);
                let jittered_options = MctsOptions {
                    cpuct: base_options.cpuct * jitter,
                    ..*base_options
                };

                s.spawn(move || {
                    run_search_thread(&jittered_options, &TimeManagement::infinite(), thread_id);
                });
            }

            if is_interactive {
                while !stop_signal.load(Ordering::Relaxed) {
                    let line = read_stdin();

                    returned_line = match line.trim() {
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
                    };

                    if returned_line.is_some() {
                        break;
                    }
                }
            }
        });

        returned_line
    }

    pub fn playout_sync(&self, playouts: u64) {
        let root_edges = RootEdge::from_edges(self.mcts.root_edges());
        let mut tld = ThreadData::create(&self.ttable, 0, root_edges, 0.0);
        let options = &self.engine_options.mcts_options;
        let tm = TimeManagement::infinite();
        let stop_signal = AtomicBool::new(false);

        for _ in 0..playouts {
            if !self.mcts.playout(&mut tld, options, &tm, &stop_signal) {
                break;
            }
        }

        self.mcts.flush_root_edges(&mut tld);
        self.mcts.flush_thread_stats(&mut tld);
    }

    pub fn print_move_list(&self, tokens: Tokens) {
        let mut current_node: Option<&PositionNode> = None;
        let mut current_edges = self.mcts.root_edges();
        let mut current_state = self.mcts.root_state().clone();

        // Navigate down the tree following the move sequence
        for move_str in tokens {
            let edge = current_edges
                .iter()
                .find(|e| self.to_uci(*e.get_move()) == move_str);

            match edge {
                Some(e) if e.visits() > 0 => {
                    if let Some(child) = e.child() {
                        current_node = Some(child);
                        current_edges = child.edges();
                        current_state.make_move(*e.get_move());
                    } else {
                        println!("info string error: move {move_str} has no child node");
                        return;
                    }
                }
                Some(_) => {
                    println!("info string error: move {move_str} not explored in search tree");
                    return;
                }
                None => {
                    println!("info string error: move {move_str} not found");
                    return;
                }
            }
        }

        let node_moves = current_edges;

        let state_moves = current_state.available_moves();
        let state_moves_eval = evaluation::policy(
            &current_state,
            &state_moves,
            self.engine_options.mcts_options.policy_temperature,
        );

        // Calculate exploration coefficient for Q+U display
        let total_visits: u64 = node_moves
            .iter()
            .map(|e| u64::from(e.visits()))
            .sum::<u64>();
        let gini = if let Some(node) = current_node {
            f32::from(node.gini()) / SCALE
        } else {
            // At root, calculate gini from visit distribution
            math::gini(node_moves.iter().map(MoveEdge::visits), total_visits)
        };
        let explore_coef = self.mcts.exploration_coefficient(
            &self.engine_options.mcts_options,
            total_visits,
            false,
            gini,
        );

        let mut moves: Vec<(&MoveEdge, f32)> = node_moves.iter().zip(state_moves_eval).collect();
        moves.sort_by_key(|(h, e)| (h.reward().average, (e * SCALE) as i64));
        for (mov, e) in moves {
            let reward = mov.reward();
            let u = mcts::exploration_bonus(explore_coef, mov.policy(), reward.visits);

            println!(
                "info string {:7} M: {:>5.2} P: {:>5.2} V: {:7} ({:>5.2}%) Q: {:>7.2} ({:>8}) U: {:>7.2}",
                self.to_uci(*mov.get_move()),
                e * 100.,
                f32::from(mov.policy()) / SCALE * 100.,
                mov.visits(),
                mov.visits() as f32 / total_visits as f32 * 100.,
                reward.average as f32 / (SCALE / 100.),
                eval_in_cp(reward.average as f32 / SCALE),
                u as f32 / (SCALE / 100.),
            );
        }
    }

    pub fn print_eval(&self) {
        let eval = evaluation::value(self.mcts.root_state());
        let scaled = eval as f32 / SCALE;
        println!(
            "info string eval {} scaled {} {}",
            eval,
            scaled,
            eval_in_cp(scaled)
        );
    }

    pub fn best_move(&self) -> Move {
        self.mcts.best_move()
    }

    /// Returns the move with the highest visit count from the root position.
    ///
    /// # Panics
    ///
    /// Panics if the root node has no child moves (e.g., checkmate or stalemate positions).
    pub fn most_visited_move(&self) -> Move {
        *select_edge_by_visits(self.mcts.root_edges())
            .expect("Root node must have moves to determine most visited move")
            .get_move()
    }

    fn to_uci(&self, mov: Move) -> String {
        mov.to_uci(self.engine_options.is_chess960)
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

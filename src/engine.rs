use std::sync::atomic::{AtomicBool, Ordering};

use crate::chess::Move;
use crate::evaluation;
use crate::graph::{MoveEdge, PositionNode};
use crate::math::Rng;
use crate::mcts::Mcts;
use crate::options::EngineOptions;
use crate::state::State;
use crate::tablebase;
use crate::threadpool::{Scope, ThreadPool};
use crate::time_management::TimeManagement;
use crate::transposition_table::{LRAllocator, LRTable};
use crate::uci::{read_stdin, Tokens};

pub const SCALE: f32 = 256. * 256.;

pub struct ThreadData<'a> {
    pub ttable: &'a LRTable,
    pub allocator: LRAllocator<'a>,
    pub playouts: usize,
    pub thread_id: usize,
}

impl<'a> ThreadData<'a> {
    fn create(ttable: &'a LRTable, thread_id: usize) -> Self {
        Self {
            ttable,
            allocator: ttable.allocator(),
            playouts: 0,
            thread_id,
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

    pub fn root_node(&self) -> &PositionNode {
        self.mcts.root_node()
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
        self.ttable.flip(|| self.root_node().clear_children_links());
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
        let cpuct = self.engine_options.mcts_options.cpuct;
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

        let run_search_thread = |cpuct: f32, tm: &TimeManagement, thread_id: usize| {
            let mut tld = ThreadData::create(&self.ttable, thread_id);
            while self.mcts.playout(&mut tld, cpuct, tm, &stop_signal) {}
        };

        self.threads.scope(|s: &Scope| {
            s.spawn(|| {
                run_search_thread(cpuct, &time_management, 0);
                self.mcts.print_info(&time_management, self.ttable.full());
                stop_signal.store(true, Ordering::Relaxed);
                println!("bestmove {}", self.to_uci(self.best_move()));
            });

            for thread_id in 1..thread_count {
                let jitter = 1. + rng.next_f32_range(-0.03, 0.03);

                s.spawn(move || {
                    run_search_thread(cpuct * jitter, &TimeManagement::infinite(), thread_id);
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
        let mut tld = ThreadData::create(&self.ttable, 0);
        let cpuct = self.engine_options.mcts_options.cpuct;
        let tm = TimeManagement::infinite();
        let stop_signal = AtomicBool::new(false);

        for _ in 0..playouts {
            if !self.mcts.playout(&mut tld, cpuct, &tm, &stop_signal) {
                break;
            }
        }
    }

    pub fn print_move_list(&self) {
        let root_node = self.mcts.root_node();
        let root_state = self.mcts.root_state();

        let root_moves = root_node.edges();

        let state_moves = root_state.available_moves();
        let state_moves_eval = evaluation::policy(
            root_state,
            &state_moves,
            self.engine_options.mcts_options.policy_temperature,
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
        *self
            .mcts
            .root_node()
            .select_child_by_visits()
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

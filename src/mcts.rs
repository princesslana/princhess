use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::Arc;
use std::thread::JoinHandle;

use crate::evaluation;
use crate::search::{TimeManagement, SCALE};
pub use crate::search_tree::*;
use crate::state::State;
use crate::transposition_table::{LRAllocator, TranspositionTable};

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

pub struct Mcts {
    search_tree: SearchTree,
}

impl Mcts {
    pub fn new(state: State, table: TranspositionTable, prev_table: TranspositionTable) -> Self {
        let search_tree = SearchTree::new(state, table, prev_table);
        Self { search_tree }
    }

    unsafe fn spawn_worker_thread(
        &self,
        stop_signal: Arc<AtomicBool>,
        time_managment: TimeManagement,
        sender: &Sender<String>,
    ) -> JoinHandle<()> {
        let search_tree = &self.search_tree;
        let sender_clone = sender.clone();
        crossbeam::spawn_unsafe(move || {
            let mut tld = ThreadData::create(search_tree);
            loop {
                if stop_signal.load(Ordering::SeqCst) {
                    break;
                }
                if !search_tree.playout(&mut tld, time_managment) {
                    if !stop_signal.swap(true, Ordering::SeqCst) {
                        sender_clone.send("stop".to_string()).unwrap_or(());
                    }
                    break;
                }
            }
        })
    }

    #[allow(unused)]
    pub fn playout_sync(&self) {
        let search_tree = &self.search_tree;
        let mut tld = ThreadData::create(search_tree);
        while search_tree.playout(&mut tld, TimeManagement::infinite()) {}
    }

    pub fn playout_sync_n(&self, n: usize) {
        let search_tree = &self.search_tree;
        let mut tld = ThreadData::create(search_tree);
        for _ in 0..n {
            if !search_tree.playout(&mut tld, TimeManagement::infinite()) {
                break;
            }
        }
    }

    pub fn into_playout_parallel_async(
        self,
        num_threads: usize,
        time_management: TimeManagement,
        sender: &Sender<String>,
    ) -> AsyncSearchOwned {
        assert!(num_threads != 0);
        let self_box = Box::new(self);
        let stop_signal = Arc::new(AtomicBool::new(false));
        let threads = (0..num_threads)
            .map(|_| {
                let stop_signal = stop_signal.clone();
                unsafe { self_box.spawn_worker_thread(stop_signal, time_management, sender) }
            })
            .collect();
        AsyncSearchOwned {
            manager: Some(self_box),
            stop_signal,
            threads,
        }
    }

    pub fn principal_variation(&self, num_moves: usize) -> Vec<shakmaty::Move> {
        self.search_tree
            .principal_variation(num_moves)
            .into_iter()
            .map(MoveInfoHandle::get_move)
            .cloned()
            .collect()
    }

    pub fn tree(&self) -> &SearchTree {
        &self.search_tree
    }

    pub fn table(self) -> TranspositionTable {
        self.search_tree.table()
    }

    pub fn best_move(&self) -> Option<shakmaty::Move> {
        self.principal_variation(1).get(0).cloned()
    }

    pub fn eval(&self) -> f32 {
        self.search_tree.eval()
    }

    pub fn print_move_list(&self) {
        let root_node = self.tree().root_node();
        let root_state = self.tree().root_state();

        let root_moves = root_node.moves();

        let state_moves = root_state.available_moves();
        let state_moves_eval = evaluation::evaluate_policy(root_state, &state_moves);

        let mut moves: Vec<(MoveInfoHandle, f32)> = root_moves.zip(state_moves_eval).collect();
        moves.sort_by_key(|(h, e)| (h.average_reward().unwrap_or(*e) * SCALE) as i64);
        for (mov, e) in moves {
            println!(
                "info string {:>6} M: {:>6} P: {:>6} V: {:7} E: {:>6} ({:>8})",
                format!("{}", mov.get_move()),
                format!("{:3.2}", e * 100.),
                format!("{:3.2}", mov.policy() * 100.),
                mov.visits(),
                mov.average_reward()
                    .map_or("n/a".to_string(), |r| format!("{:3.2}", r / (SCALE / 100.))),
                mov.average_reward()
                    .map_or("n/a".to_string(), |r| eval_in_cp(r / SCALE))
            );
        }
    }
}

pub struct AsyncSearchOwned {
    manager: Option<Box<Mcts>>,
    stop_signal: Arc<AtomicBool>,
    threads: Vec<JoinHandle<()>>,
}

impl AsyncSearchOwned {
    fn stop_threads(&mut self) {
        self.stop_signal.store(true, Ordering::SeqCst);
        drain_join_unwrap(&mut self.threads);
    }
    pub fn halt(mut self) -> Mcts {
        self.stop_threads();
        *self.manager.take().unwrap()
    }
    pub fn get_manager(&self) -> &Mcts {
        self.manager.as_ref().unwrap()
    }
    pub fn num_threads(&self) -> usize {
        self.threads.len()
    }
}

impl Drop for AsyncSearchOwned {
    fn drop(&mut self) {
        self.stop_threads();
    }
}

impl From<Mcts> for AsyncSearchOwned {
    /// An `Mcts` is an `AsyncSearchOwned` with zero threads searching.
    fn from(m: Mcts) -> Self {
        Self {
            manager: Some(Box::new(m)),
            stop_signal: Arc::new(AtomicBool::new(false)),
            threads: Vec::new(),
        }
    }
}

fn drain_join_unwrap(threads: &mut Vec<JoinHandle<()>>) {
    let join_results: Vec<_> = threads.drain(..).map(JoinHandle::join).collect();
    for x in join_results {
        x.unwrap();
    }
}

// eval here is [-1.0, 1.0]
pub fn eval_in_cp(eval: f32) -> String {
    let cps = if eval > 0.5 {
        18. * (eval - 0.5) + 1.
    } else if eval < -0.5 {
        18. * (eval + 0.5) - 1.
    } else {
        2. * eval
    };

    format!("cp {}", (cps * 100.).round().max(-1000.).min(1000.) as i64)
}

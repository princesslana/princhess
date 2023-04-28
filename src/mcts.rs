use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Arc, RwLock};
use std::thread::JoinHandle;
use std::time::Instant;

use crate::evaluation;
use crate::search::{to_uci, SCALE};
pub use crate::search_tree::*;
use crate::state::State;
use crate::transposition_table::*;
use crate::tree_policy::*;

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

pub type MoveEvaluation = f32;

pub struct MctsManager {
    search_tree: SearchTree,
    search_start: RwLock<Option<Instant>>,
}

impl MctsManager {
    pub fn new(
        state: State,
        tree_policy: Puct,
        table: TranspositionTable,
        prev_table: TranspositionTable,
    ) -> Self {
        let search_tree = SearchTree::new(state, tree_policy, table, prev_table);
        Self {
            search_tree,
            search_start: RwLock::new(None),
        }
    }

    unsafe fn spawn_worker_thread(
        &self,
        stop_signal: Arc<AtomicBool>,
        sender: &Sender<String>,
    ) -> JoinHandle<()> {
        let search_tree = &self.search_tree;
        {
            let mut lock = self.search_start.write().unwrap();
            let _ = lock.get_or_insert(Instant::now());
        }
        let sender_clone = sender.clone();
        crossbeam::spawn_unsafe(move || {
            let mut tld = ThreadData::create(search_tree);
            loop {
                if stop_signal.load(Ordering::SeqCst) {
                    break;
                }
                if !search_tree.playout(&mut tld) {
                    if !stop_signal.swap(true, Ordering::SeqCst) {
                        let _ = sender_clone.send("stop".to_string());
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
        while search_tree.playout(&mut tld) {}
    }

    pub fn into_playout_parallel_async(
        self,
        num_threads: usize,
        sender: &Sender<String>,
    ) -> AsyncSearchOwned {
        assert!(num_threads != 0);
        let self_box = Box::new(self);
        let stop_signal = Arc::new(AtomicBool::new(false));
        let threads = (0..num_threads)
            .map(|_| {
                let stop_signal = stop_signal.clone();
                unsafe { self_box.spawn_worker_thread(stop_signal, sender) }
            })
            .collect();
        AsyncSearchOwned {
            manager: Some(self_box),
            stop_signal,
            threads,
        }
    }

    pub fn principal_variation_info(&self, num_moves: usize) -> Vec<MoveInfoHandle> {
        self.search_tree.principal_variation(num_moves)
    }
    pub fn principal_variation(&self, num_moves: usize) -> Vec<shakmaty::Move> {
        self.search_tree
            .principal_variation(num_moves)
            .into_iter()
            .map(|x| x.get_move())
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

    pub fn eval_in_cp(&self) -> String {
        eval_in_cp(
            self.principal_variation_info(1)
                .get(0)
                .map(|x| (x.sum_rewards() / x.visits() as i64) as f32 / SCALE)
                .unwrap_or(0.0),
        )
    }

    pub fn print_info(&self) {
        let search_start = self.search_start.read().unwrap();

        let search_time_ms = Instant::now()
            .duration_since(search_start.unwrap())
            .as_millis();

        let nodes = self.tree().num_nodes();
        let depth = fastapprox::faster::ln(nodes as f32).round();
        let pv = self.principal_variation(64);
        let sel_depth = pv.len();
        let pv_string: String = pv.into_iter().map(|x| format!(" {}", to_uci(x))).collect();

        let nps = nodes * 1000 / search_time_ms as usize;

        let info_str = format!(
            "info depth {} seldepth {} nodes {} nps {} tbhits {} score {} time {} pv{}",
            depth,
            sel_depth,
            nodes,
            nps,
            self.tree().tb_hits(),
            self.eval_in_cp(),
            search_time_ms,
            pv_string,
        );
        println!("{info_str}");
    }

    pub fn print_move_list(&self) {
        let root_node = self.tree().root_node();
        let root_state = self.tree().root_state();

        let root_moves = root_node.moves();

        let state_moves = root_state.available_moves();
        let (state_moves_eval, _) = evaluation::evaluate_new_state(root_state, &state_moves);

        let mut moves: Vec<(MoveInfoHandle, f32)> = root_moves.zip(state_moves_eval).collect();
        moves.sort_by_key(|(h, e)| (h.average_reward().unwrap_or(*e) * SCALE) as i64);
        for (mov, e) in moves {
            println!(
                "info string {:>6} M: {:>6} P: {:>6} V: {:7} E: {:>6} ({:>8})",
                format!("{}", mov.get_move()),
                format!("{:3.2}", e * 100.),
                format!("{:3.2}", mov.move_evaluation() * 100.),
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
    manager: Option<Box<MctsManager>>,
    stop_signal: Arc<AtomicBool>,
    threads: Vec<JoinHandle<()>>,
}

impl AsyncSearchOwned {
    fn stop_threads(&mut self) {
        self.stop_signal.store(true, Ordering::SeqCst);
        drain_join_unwrap(&mut self.threads);
    }
    pub fn halt(mut self) -> MctsManager {
        self.stop_threads();
        *self.manager.take().unwrap()
    }
    pub fn get_manager(&self) -> &MctsManager {
        self.manager.as_ref().unwrap()
    }
    pub fn get_stop_signal(&self) -> &Arc<AtomicBool> {
        &self.stop_signal
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

impl From<MctsManager> for AsyncSearchOwned {
    /// An `MctsManager` is an `AsyncSearchOwned` with zero threads searching.
    fn from(m: MctsManager) -> Self {
        Self {
            manager: Some(Box::new(m)),
            stop_signal: Arc::new(AtomicBool::new(false)),
            threads: Vec::new(),
        }
    }
}

fn drain_join_unwrap(threads: &mut Vec<JoinHandle<()>>) {
    let join_results: Vec<_> = threads.drain(..).map(|x| x.join()).collect();
    for x in join_results {
        x.unwrap();
    }
}

// eval here is [-1.0, 1.0]
fn eval_in_cp(eval: f32) -> String {
    let cps = if eval > 0.5 {
        20. * (eval - 0.5) + 0.5
    } else if eval < -0.5 {
        20. * (eval + 0.5) - 0.5
    } else {
        2. * eval
    };

    format!("cp {}", (cps * 100.).round().max(-1000.).min(1000.) as i64)
}

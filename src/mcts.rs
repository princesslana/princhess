use arena::ArenaAllocator;
pub use search_tree::*;
use transposition_table::*;
use tree_policy::*;

use chess;
use fastapprox;
use float_ord::FloatOrd;
use policy_features::evaluate_moves;
use search::{to_uci, SCALE};
use state::State;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Arc, RwLock};
use std::thread::JoinHandle;
use std::time::Instant;

pub trait MCTS: Sized + Sync {
    type Eval: Evaluator<Self>;
    type TreePolicy: TreePolicy<Self>;
    type TranspositionTable: TranspositionTable;
    type ExtraThreadData;

    /// Virtual loss subtracted from a node's evaluation when a search thread chooses it in a playout,
    /// then added back when the playout is complete.
    /// Used to reduce contention between threads. Defaults to 0.
    fn virtual_loss(&self) -> i64 {
        0
    }

    /// Maximum number of nodes beyond which calling `playout` will do nothing. Defaults to `std::usize::MAX`.
    fn node_limit(&self) -> usize {
        std::usize::MAX
    }
    /// Rule for selecting the best move once the search is over. Defaults to choosing the child with the most visits.
    fn select_child_after_search<'a>(&self, children: &[MoveInfoHandle<'a>]) -> MoveInfoHandle<'a> {
        *children
            .into_iter()
            .max_by_key(|child| child.visits())
            .unwrap()
    }
    /// `playout` panics when this length is exceeded. Defaults to one million.
    fn max_playout_length(&self) -> usize {
        1_000_000
    }

    fn cycle_behaviour(&self) -> CycleBehaviour {
        if std::mem::size_of::<Self::TranspositionTable>() == 0 {
            CycleBehaviour::Ignore
        } else {
            CycleBehaviour::PanicWhenCycleDetected
        }
    }
}

pub struct ThreadData<'a, Spec: MCTS> {
    pub policy_data: TreePolicyThreadData<Spec>,
    pub extra_data: Spec::ExtraThreadData,
    pub allocator: ArenaAllocator<'a>,
}

impl<'a, Spec: MCTS> ThreadData<'a, Spec>
where
    TreePolicyThreadData<Spec>: Default,
    Spec::ExtraThreadData: Default,
{
    fn create(tree: &'a SearchTree<Spec>) -> Self {
        Self {
            policy_data: Default::default(),
            extra_data: Default::default(),
            allocator: tree.arena().allocator(),
        }
    }
}

pub type MoveEvaluation = f32;
pub type StateEvaluation = i64;
pub type MoveList = <State as GameState>::MoveList;
pub type Player = <State as GameState>::Player;
pub type TreePolicyThreadData<Spec> =
    <<Spec as MCTS>::TreePolicy as TreePolicy<Spec>>::ThreadLocalData;

pub trait GameState: Clone {
    type Player: Sync;
    type MoveList: std::iter::IntoIterator<Item = chess::ChessMove>;

    fn current_player(&self) -> Self::Player;
    fn available_moves(&self) -> Self::MoveList;
    fn make_move(&mut self, mov: &chess::ChessMove);
}

pub trait Evaluator<Spec: MCTS>: Sync {
    fn evaluate_new_state(
        &self,
        state: &State,
        moves: &MoveList,
    ) -> (Vec<MoveEvaluation>, StateEvaluation, bool);

    fn interpret_evaluation_for_player(&self, evaluation: &StateEvaluation, player: &Player)
        -> i64;
}

pub struct MCTSManager<Spec: MCTS> {
    search_tree: SearchTree<Spec>,
    search_start: RwLock<Option<Instant>>,
}

impl<Spec: MCTS> MCTSManager<Spec>
where
    TreePolicyThreadData<Spec>: Default,
    Spec::ExtraThreadData: Default,
{
    pub fn new(
        state: State,
        manager: Spec,
        eval: Spec::Eval,
        tree_policy: Spec::TreePolicy,
        table: Spec::TranspositionTable,
        prev_table: PreviousTable<Spec>,
    ) -> Self {
        let search_tree = SearchTree::new(state, manager, tree_policy, eval, table, prev_table);
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

    pub fn playout_sync(&self) {
        let search_tree = &self.search_tree;
        let mut tld = ThreadData::create(search_tree);
        while search_tree.playout(&mut tld) {}
    }

    pub fn into_playout_parallel_async(
        self,
        num_threads: usize,
        sender: &Sender<String>,
    ) -> AsyncSearchOwned<Spec> {
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
    pub fn principal_variation(&self, num_moves: usize) -> Vec<chess::ChessMove> {
        self.search_tree
            .principal_variation(num_moves)
            .into_iter()
            .map(|x| x.get_move())
            .map(|x| x.clone())
            .collect()
    }

    pub fn tree(&self) -> &SearchTree<Spec> {
        &self.search_tree
    }

    pub fn table(self) -> PreviousTable<Spec> {
        self.search_tree.table()
    }

    pub fn best_move(&self) -> Option<chess::ChessMove> {
        self.principal_variation(1).get(0).map(|x| x.clone())
    }

    pub fn eval_in_cp(&self) -> String {
        eval_in_cp(
            self.principal_variation_info(1)
                .get(0)
                .map(|x| (x.sum_rewards() / x.visits() as i64) as f32 / SCALE)
                .unwrap_or(0.0),
        )
    }

    pub fn nodes(&self) -> usize {
        self.tree().num_nodes()
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
        println!("{}", info_str);
    }

    pub fn print_move_list(&self) {
        let root_node = self.tree().root_node();
        let root_state = self.tree().root_state();
        let root_features = root_state.features();

        let root_moves = root_node.moves();

        let state_moves = root_state.available_moves();
        let state_moves_eval = evaluate_moves(&root_state, &root_features, state_moves.as_slice());

        let mut moves: Vec<(MoveInfoHandle, f32)> = root_moves.zip(state_moves_eval).collect();
        moves.sort_by_key(|(h, e)| FloatOrd(h.average_reward().unwrap_or(*e)));
        for (mov, e) in moves {
            println!(
                "info string {:5} M: {:>6} P: {:>6} V: {:7} E: {:>6} ({:>8})",
                mov.get_move(),
                format!("{:3.2}", e * 100.),
                format!("{:3.2}", mov.move_evaluation() * 100.),
                mov.visits(),
                mov.average_reward()
                    .map_or("n/a".to_string(), |r| format!("{:3.2}", r / (SCALE / 100.))),
                mov.average_reward()
                    .map_or("n/a".to_string(), |r| format!("{}", eval_in_cp(r / SCALE)))
            );
        }
    }
}

pub struct AsyncSearchOwned<Spec: MCTS> {
    manager: Option<Box<MCTSManager<Spec>>>,
    stop_signal: Arc<AtomicBool>,
    threads: Vec<JoinHandle<()>>,
}

impl<Spec: MCTS> AsyncSearchOwned<Spec> {
    fn stop_threads(&mut self) {
        self.stop_signal.store(true, Ordering::SeqCst);
        drain_join_unwrap(&mut self.threads);
    }
    pub fn halt(mut self) -> MCTSManager<Spec> {
        self.stop_threads();
        *self.manager.take().unwrap()
    }
    pub fn get_manager(&self) -> &MCTSManager<Spec> {
        self.manager.as_ref().unwrap()
    }
    pub fn get_stop_signal(&self) -> &Arc<AtomicBool> {
        &self.stop_signal
    }
    pub fn num_threads(&self) -> usize {
        self.threads.len()
    }
}

impl<Spec: MCTS> Drop for AsyncSearchOwned<Spec> {
    fn drop(&mut self) {
        self.stop_threads();
    }
}

impl<Spec: MCTS> From<MCTSManager<Spec>> for AsyncSearchOwned<Spec> {
    /// An `MCTSManager` is an `AsyncSearchOwned` with zero threads searching.
    fn from(m: MCTSManager<Spec>) -> Self {
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

pub enum CycleBehaviour {
    Ignore,
    UseCurrentEvalWhenCycleDetected,
    PanicWhenCycleDetected,
    UseThisEvalWhenCycleDetected(StateEvaluation),
}

// eval here is [0.0, 1.0]
fn eval_in_cp(eval: f32) -> String {
    if eval.abs() > 1.0 {
        let plies = (1.1 - eval.abs()) / 0.001;
        let mvs = plies / 2.;
        let mate_score = (eval.signum() * mvs).round();
        format!("mate {:+}", mate_score)
    } else {
        // Based upon leela's formula.
        // Tweaked to appear in the range (-1000, 1000)
        format!("cp {}", (101.139 * (1.47 * eval).tan()).round())
    }
}

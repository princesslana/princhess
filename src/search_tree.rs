use mcts::*;
use options::get_hash_size_mb;
use policy_features::softmax;
use search::{GooseMCTS, SCALE};
use smallvec::SmallVec;
use state::State;
use std::mem;
use std::ptr::null_mut;
use std::sync::atomic::{AtomicI64, AtomicPtr, AtomicU32, AtomicUsize, Ordering};
use transposition_table::{ApproxTable, TranspositionTable};

use log::debug;
use pod::Pod;

use tree_policy::TreePolicy;

use arena::{Arena, ArenaAllocator, ArenaError};

/// You're not intended to use this class (use an `MCTSManager` instead),
/// but you can use it if you want to manage the threads yourself.
pub struct SearchTree<Spec: MCTS> {
    root_node: SearchNode,
    root_state: State,
    tree_policy: Spec::TreePolicy,
    table: Spec::TranspositionTable,
    prev_table: PreviousTable<Spec>,
    eval: Spec::Eval,
    manager: Spec,
    arena: Box<Arena>,

    num_nodes: AtomicUsize,
    tb_hits: AtomicUsize,
    transposition_table_hits: AtomicUsize,
    delayed_transposition_table_hits: AtomicUsize,
    expansion_contention_events: AtomicUsize,
}

pub struct PreviousTable<Spec: MCTS> {
    table: Spec::TranspositionTable,
    #[allow(dead_code)]
    arena: Box<Arena>,
}

pub fn empty_previous_table() -> PreviousTable<GooseMCTS> {
    PreviousTable {
        table: ApproxTable::enough_to_hold(0),
        arena: Box::new(Arena::new(2)),
    }
}

impl<Spec: MCTS> PreviousTable<Spec> {
    pub fn lookup_into(&self, state: &State, dest: &mut SearchNode) {
        if let Some(src) = self.table.lookup(state) {
            dest.replace(src);
            dest.evaln = src.evaln;

            let lhs = dest.hots();
            let rhs = src.hots();

            for i in 0..lhs.len().min(rhs.len()) {
                lhs[i].replace(&rhs[i]);
            }
        }
    }
}

trait NodeStats {
    fn get_visits(&self) -> &AtomicU32;
    fn get_sum_evaluations(&self) -> &AtomicI64;

    fn down<Spec: MCTS>(&self, manager: &Spec) {
        self.get_sum_evaluations()
            .fetch_sub(manager.virtual_loss() as i64, Ordering::Relaxed);
        self.get_visits().fetch_add(1, Ordering::Relaxed);
    }
    fn up<Spec: MCTS>(&self, manager: &Spec, evaln: i64) {
        let delta = evaln + manager.virtual_loss();
        self.get_sum_evaluations()
            .fetch_add(delta as i64, Ordering::Relaxed);
    }
    fn replace<T: NodeStats>(&self, other: &T) {
        self.get_visits().store(
            other.get_visits().load(Ordering::Relaxed),
            Ordering::Relaxed,
        );
        self.get_sum_evaluations().store(
            other.get_sum_evaluations().load(Ordering::Relaxed),
            Ordering::Relaxed,
        );
    }
}

impl NodeStats for HotMoveInfo {
    fn get_visits(&self) -> &AtomicU32 {
        &self.visits
    }
    fn get_sum_evaluations(&self) -> &AtomicI64 {
        &self.sum_evaluations
    }
}
impl NodeStats for SearchNode {
    fn get_visits(&self) -> &AtomicU32 {
        &self.visits
    }
    fn get_sum_evaluations(&self) -> &AtomicI64 {
        &self.sum_evaluations
    }
}

struct HotMoveInfo {
    sum_evaluations: AtomicI64,
    visits: AtomicU32,
    move_evaluation: MoveEvaluation,
    mov: chess::ChessMove,
    child: AtomicPtr<SearchNode>,
}
pub struct MoveInfoHandle<'a> {
    hot: &'a HotMoveInfo,
}

unsafe impl Pod for HotMoveInfo {}
unsafe impl Pod for SearchNode {}

impl<'a> Clone for MoveInfoHandle<'a> {
    fn clone(&self) -> Self {
        Self { hot: self.hot }
    }
}
impl<'a> Copy for MoveInfoHandle<'a> {}

pub struct SearchNode {
    hots: *const [()],
    evaln: StateEvaluation,
    sum_evaluations: AtomicI64,
    visits: AtomicU32,
}

unsafe impl Sync for SearchNode {}

impl SearchNode {
    fn new<'a>(hots: &'a [HotMoveInfo], evaln: StateEvaluation) -> Self {
        Self {
            hots: hots as *const _ as *const [()],
            evaln,
            visits: AtomicU32::default(),
            sum_evaluations: AtomicI64::default(),
        }
    }
    fn hots(&self) -> &[HotMoveInfo] {
        unsafe { &*(self.hots as *const [HotMoveInfo]) }
    }
    fn update_policy(&self) {
        let mut evals: Vec<f32> = self
            .moves()
            .into_iter()
            .map(|m| m.average_reward().unwrap_or(-SCALE) / SCALE)
            .collect();

        softmax(&mut evals);

        let mut hots = unsafe { &mut *(self.hots as *mut [HotMoveInfo]) };

        for i in 0..hots.len().min(evals.len()) {
            hots[i].move_evaluation = evals[i];
        }
    }
    pub fn moves(&self) -> Moves {
        Moves {
            hots: self.hots(),
            index: 0,
        }
    }
}

impl HotMoveInfo {
    fn new(move_evaluation: MoveEvaluation, mov: chess::ChessMove) -> Self {
        Self {
            move_evaluation,
            sum_evaluations: AtomicI64::default(),
            visits: AtomicU32::default(),
            mov,
            child: AtomicPtr::default(),
        }
    }
}

impl<'a> MoveInfoHandle<'a> {
    pub fn get_move(&self) -> &'a chess::ChessMove {
        &self.hot.mov
    }

    pub fn move_evaluation(&self) -> &'a MoveEvaluation {
        &self.hot.move_evaluation
    }

    pub fn visits(&self) -> u64 {
        self.hot.visits.load(Ordering::Relaxed) as u64
    }

    pub fn sum_rewards(&self) -> i64 {
        self.hot.sum_evaluations.load(Ordering::Relaxed) as i64
    }

    pub fn average_reward(&self) -> Option<f32> {
        match self.visits() {
            0 => None,
            x => Some(self.sum_rewards() as f32 / x as f32),
        }
    }
}

enum CreationHelper<'a: 'b, 'b, Spec: 'a + MCTS> {
    Handle(SearchHandle<'a, 'b, Spec>),
    Allocator(&'b ArenaAllocator<'a>),
}

#[inline(always)]
fn create_node<'a, 'b, Spec: MCTS>(
    eval: &Spec::Eval,
    policy: &Spec::TreePolicy,
    state: &State,
    tb_hits: &AtomicUsize,
    ch: CreationHelper<'a, 'b, Spec>,
) -> Result<SearchNode, ArenaError> {
    let (allocator, _handle) = match ch {
        CreationHelper::Allocator(x) => (x, None),
        CreationHelper::Handle(x) => {
            // this is safe because nothing will move into x.tld.allocator
            // since ThreadData.allocator is a private field
            let allocator = unsafe { &*(&x.tld.allocator as *const _) };
            (allocator, Some(x))
        }
    };
    let moves = state.available_moves();
    let (move_eval, state_eval, is_tb_hit) = eval.evaluate_new_state(state, &moves);

    if is_tb_hit {
        tb_hits.fetch_add(1, Ordering::Relaxed);
    }

    policy.validate_evaluations(&move_eval);
    let hots = allocator.alloc_slice(move_eval.len())?;
    let moves_slice = moves.as_slice();
    for (i, x) in hots.iter_mut().enumerate() {
        *x = HotMoveInfo::new(move_eval[i], moves_slice[i]);
    }
    Ok(SearchNode::new(hots, state_eval))
}

fn is_cycle<T>(past: &[&T], current: &T) -> bool {
    past.iter().any(|x| std::ptr::eq(*x, current))
}

impl<Spec: MCTS> SearchTree<Spec> {
    pub fn new(
        state: State,
        manager: Spec,
        tree_policy: Spec::TreePolicy,
        eval: Spec::Eval,
        table: Spec::TranspositionTable,
        prev_table: PreviousTable<Spec>,
    ) -> Self {
        let arena = Box::new(Arena::new(get_hash_size_mb() / 2));
        let tb_hits = 0.into();

        let mut root_node = create_node::<Spec>(
            &eval,
            &tree_policy,
            &state,
            &tb_hits,
            CreationHelper::Allocator(&arena.allocator()),
        )
        .expect("Unable to create root node");

        prev_table.lookup_into(&state, &mut root_node);

        root_node.update_policy();

        Self {
            root_state: state,
            root_node,
            manager,
            tree_policy,
            eval,
            table,
            prev_table,
            num_nodes: 1.into(),
            tb_hits,
            arena,
            transposition_table_hits: 0.into(),
            delayed_transposition_table_hits: 0.into(),
            expansion_contention_events: 0.into(),
        }
    }

    pub fn spec(&self) -> &Spec {
        &self.manager
    }

    pub fn table(self) -> PreviousTable<Spec> {
        PreviousTable {
            table: self.table,
            arena: self.arena,
        }
    }

    pub fn num_nodes(&self) -> usize {
        self.num_nodes.load(Ordering::SeqCst)
    }

    pub fn tb_hits(&self) -> usize {
        self.tb_hits.load(Ordering::SeqCst)
    }

    pub fn arena(&self) -> &Arena {
        &self.arena
    }

    #[inline(never)]
    pub fn playout<'a: 'b, 'b>(&'a self, tld: &'b mut ThreadData<'a, Spec>) -> bool {
        const LARGE_DEPTH: usize = 64;
        let sentinel = IncreaseSentinel::new(&self.num_nodes);
        if sentinel.num_nodes >= self.manager.node_limit() {
            debug!(
                "Node limit of {} reached. Halting search.",
                self.spec().node_limit()
            );
            println!("info hashfull 1000");
            return false;
        }
        let mut state = self.root_state.clone();
        let mut path: SmallVec<[MoveInfoHandle; LARGE_DEPTH]> = SmallVec::new();
        let mut node_path: SmallVec<[&SearchNode; LARGE_DEPTH]> = SmallVec::new();
        let mut players: SmallVec<[Player; LARGE_DEPTH]> = SmallVec::new();
        let mut node = &self.root_node;
        loop {
            if node.hots().is_empty() {
                break;
            }
            if path.len() >= self.manager.max_playout_length() {
                break;
            }

            if path.len() > 0 {
                node.update_policy();
            }

            let choice = self.tree_policy.choose_child(
                &state,
                node.moves(),
                self.make_handle(tld, &node_path),
            );
            choice.hot.down(&self.manager);
            players.push(state.current_player());
            path.push(choice);
            assert!(path.len() <= self.manager.max_playout_length(),
                "playout length exceeded maximum of {} (maybe the transposition table is creating an infinite loop?)",
                self.manager.max_playout_length());
            state.make_move(&choice.hot.mov);

            let new_node = match self.descend(&state, choice.hot, tld, &node_path) {
                Ok(r) => r,
                Err(ArenaError::Full) => {
                    debug!("Hash reached max capacity");
                    println!("info hashfull 1000");
                    return false;
                }
            };

            node = new_node;
            match self.manager.cycle_behaviour() {
                CycleBehaviour::Ignore => (),
                CycleBehaviour::PanicWhenCycleDetected => {
                    if is_cycle(&node_path, node) {
                        panic!("cycle detected! you should do one of the following:\n- make states acyclic\n- remove transposition table\n- change cycle_behaviour()");
                    }
                }
                CycleBehaviour::UseCurrentEvalWhenCycleDetected => {
                    if is_cycle(&node_path, node) {
                        break;
                    }
                }
                CycleBehaviour::UseThisEvalWhenCycleDetected(e) => {
                    if is_cycle(&node_path, node) {
                        self.finish_playout(&path, &node_path, &players, &e);
                        return true;
                    }
                }
            };
            node_path.push(node);
            node.down(&self.manager);
            if node.get_visits().load(Ordering::Relaxed) == 1 {
                break;
            }
        }

        self.finish_playout(&path, &node_path, &players, &node.evaln);
        self.root_node.get_visits().fetch_add(1, Ordering::Relaxed);

        true
    }

    fn descend<'a>(
        &'a self,
        state: &State,
        choice: &HotMoveInfo,
        tld: &mut ThreadData<'a, Spec>,
        path: &[&'a SearchNode],
    ) -> Result<&'a SearchNode, ArenaError> {
        let child = choice.child.load(Ordering::Relaxed) as *const _;
        if !child.is_null() {
            return unsafe { Ok(&*child) };
        }

        if let Some(node) = self.table.lookup(state) {
            let child = choice.child.compare_and_swap(
                null_mut(),
                node as *const _ as *mut _,
                Ordering::Relaxed,
            ) as *const _;
            if child.is_null() {
                self.transposition_table_hits
                    .fetch_add(1, Ordering::Relaxed);
                return Ok(node);
            } else {
                return unsafe { Ok(&*child) };
            }
        }

        let mut created_here = create_node(
            &self.eval,
            &self.tree_policy,
            state,
            &self.tb_hits,
            CreationHelper::Handle(self.make_handle(tld, path)),
        )?;

        self.prev_table.lookup_into(state, &mut created_here);

        let created = tld.allocator.alloc_one()?;
        *created = created_here;
        let other_child =
            choice
                .child
                .compare_and_swap(null_mut(), created as *mut _, Ordering::Relaxed);
        if !other_child.is_null() {
            self.expansion_contention_events
                .fetch_add(1, Ordering::Relaxed);
            unsafe {
                return Ok(&*other_child);
            }
        }
        if let Some(existing) = self.table.insert(state, created) {
            self.delayed_transposition_table_hits
                .fetch_add(1, Ordering::Relaxed);
            let existing_ptr = existing as *const _ as *mut _;
            choice.child.store(existing_ptr, Ordering::Relaxed);
            return Ok(existing);
        }
        self.num_nodes.fetch_add(1, Ordering::Relaxed);
        Ok(created)
    }

    fn finish_playout<'a>(
        &'a self,
        path: &[MoveInfoHandle],
        node_path: &[&'a SearchNode],
        players: &[Player],
        evaln: &StateEvaluation,
    ) {
        for ((move_info, player), node) in
            path.iter().zip(players.iter()).zip(node_path.iter()).rev()
        {
            let evaln_value = self.eval.interpret_evaluation_for_player(evaln, player);
            node.up(&self.manager, evaln_value);
            move_info.hot.replace(*node);
        }
    }

    fn make_handle<'a, 'b>(
        &'a self,
        tld: &'b mut ThreadData<'a, Spec>,
        path: &'b [&'a SearchNode],
    ) -> SearchHandle<'a, 'b, Spec> {
        let shared = SharedSearchHandle { tree: self, path };
        SearchHandle { shared, tld }
    }

    pub fn root_state(&self) -> &State {
        &self.root_state
    }
    pub fn root_node(&self) -> NodeHandle {
        NodeHandle {
            node: &self.root_node,
        }
    }

    pub fn principal_variation(&self, num_moves: usize) -> Vec<MoveInfoHandle> {
        let mut result = Vec::new();
        let mut crnt = &self.root_node;
        while !crnt.hots().is_empty() && result.len() < num_moves {
            let choice = self
                .manager
                .select_child_after_search(&crnt.moves().collect::<Vec<_>>());
            result.push(choice);
            let child = choice.hot.child.load(Ordering::SeqCst) as *const SearchNode;
            if child.is_null() {
                break;
            } else {
                unsafe {
                    crnt = &*child;
                }
            }
        }
        result
    }
}

pub struct NodeHandle<'a> {
    node: &'a SearchNode,
}
impl<'a> Clone for NodeHandle<'a> {
    fn clone(&self) -> Self {
        Self { node: self.node }
    }
}
impl<'a> Copy for NodeHandle<'a> {}

impl<'a> NodeHandle<'a> {
    pub fn moves(&self) -> Moves {
        self.node.moves()
    }
}

pub struct Moves<'a> {
    hots: &'a [HotMoveInfo],
    index: usize,
}

impl<'a> Clone for Moves<'a> {
    fn clone(&self) -> Self {
        Self {
            hots: self.hots,
            index: self.index,
        }
    }
}

impl<'a> Copy for Moves<'a> {}

impl<'a> Iterator for Moves<'a> {
    type Item = MoveInfoHandle<'a>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.index == self.hots.len() {
            None
        } else {
            let handle = unsafe {
                MoveInfoHandle {
                    hot: self.hots.get_unchecked(self.index),
                }
            };
            self.index += 1;
            Some(handle)
        }
    }
}

pub struct SharedSearchHandle<'a: 'b, 'b, Spec: 'a + MCTS> {
    tree: &'a SearchTree<Spec>,
    path: &'b [&'a SearchNode],
}
impl<'a: 'b, 'b, Spec: 'a + MCTS> Clone for SharedSearchHandle<'a, 'b, Spec> {
    fn clone(&self) -> Self {
        let tree = self.tree;
        let path = self.path;
        Self { tree, path }
    }
}
impl<'a: 'b, 'b, Spec: 'a + MCTS> Copy for SharedSearchHandle<'a, 'b, Spec> {}

pub struct SearchHandle<'a: 'b, 'b, Spec: 'a + MCTS> {
    pub tld: &'b mut ThreadData<'a, Spec>,
    pub shared: SharedSearchHandle<'a, 'b, Spec>,
}

impl<'a, 'b, Spec: MCTS> SearchHandle<'a, 'b, Spec> {
    pub fn thread_data(&mut self) -> &mut ThreadData<'a, Spec> {
        self.tld
    }
}

struct IncreaseSentinel<'a> {
    x: &'a AtomicUsize,
    num_nodes: usize,
}

impl<'a> IncreaseSentinel<'a> {
    fn new(x: &'a AtomicUsize) -> Self {
        let num_nodes = x.fetch_add(1, Ordering::Relaxed);
        Self { x, num_nodes }
    }
}

impl<'a> Drop for IncreaseSentinel<'a> {
    fn drop(&mut self) {
        self.x.fetch_sub(1, Ordering::Relaxed);
    }
}

pub fn print_size_list() {
    println!(
        "info string SearchNode {} HotMoveInfo {}",
        mem::size_of::<SearchNode>(),
        mem::size_of::<HotMoveInfo>(),
    );
}

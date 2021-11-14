use mcts::*;
use options::get_hash_size_mb;
use policy_features::softmax;
use search::{GooseMCTS, SCALE};
use smallvec::SmallVec;
use state::State;
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
    root_node: SearchNode<Spec>,
    root_state: State,
    tree_policy: Spec::TreePolicy,
    table: Spec::TranspositionTable,
    prev_table: PreviousTable<Spec>,
    eval: Spec::Eval,
    manager: Spec,
    arena: Box<Arena>,

    num_nodes: AtomicUsize,
    sum_depth: AtomicUsize,
    max_depth: AtomicUsize,
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
    pub fn lookup_into(&self, state: &State, dest: &mut SearchNode<Spec>) -> bool {
        if let Some(src) = self.table.lookup(state) {
            dest.replace(src);
            dest.evaln = src.evaln;

            let lhs = dest.hots();
            let rhs = src.hots();

            for i in 0..lhs.len().min(rhs.len()) {
                lhs[i].replace(&rhs[i]);
            }
            true
        } else {
            false
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
impl<Spec: MCTS> NodeStats for SearchNode<Spec> {
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
}
struct ColdMoveInfo<Spec: MCTS> {
    mov: chess::ChessMove,
    child: AtomicPtr<SearchNode<Spec>>,
}
pub struct MoveInfoHandle<'a, Spec: 'a + MCTS> {
    hot: &'a HotMoveInfo,
    cold: &'a ColdMoveInfo<Spec>,
}

unsafe impl Pod for HotMoveInfo {}
unsafe impl<Spec: MCTS> Pod for ColdMoveInfo<Spec> {}
unsafe impl<Spec: MCTS> Pod for SearchNode<Spec> {}

impl<'a, Spec: MCTS> Clone for MoveInfoHandle<'a, Spec> {
    fn clone(&self) -> Self {
        Self {
            hot: self.hot,
            cold: self.cold,
        }
    }
}
impl<'a, Spec: MCTS> Copy for MoveInfoHandle<'a, Spec> {}

pub struct SearchNode<Spec: MCTS> {
    hots: *const [()],
    colds: *const [()],
    evaln: StateEvaluation<Spec>,
    sum_evaluations: AtomicI64,
    visits: AtomicU32,
}

unsafe impl<Spec: MCTS> Sync for SearchNode<Spec> where
    StateEvaluation<Spec>: Sync
    // NodeStats: Sync,
    // for<'a> &'a[HotMoveInfo]: Sync,
    // for<'a> &'a[ColdMoveInfo<Spec>]: Sync,
{
}

impl<Spec: MCTS> SearchNode<Spec> {
    fn new<'a>(
        hots: &'a [HotMoveInfo],
        colds: &'a [ColdMoveInfo<Spec>],
        evaln: StateEvaluation<Spec>,
    ) -> Self {
        Self {
            hots: hots as *const _ as *const [()],
            colds: colds as *const _ as *const [()],
            evaln,
            visits: AtomicU32::default(),
            sum_evaluations: AtomicI64::default(),
        }
    }
    fn hots(&self) -> &[HotMoveInfo] {
        unsafe { &*(self.hots as *const [HotMoveInfo]) }
    }
    fn update_policy(&mut self, evals: Vec<f32>) {
        let mut hots = unsafe { &mut *(self.hots as *mut [HotMoveInfo]) };

        for i in 0..hots.len().min(evals.len()) {
            hots[i].move_evaluation = evals[i];
        }
    }
    fn colds(&self) -> &[ColdMoveInfo<Spec>] {
        unsafe { &*(self.colds as *const [ColdMoveInfo<Spec>]) }
    }
    pub fn moves(&self) -> Moves<Spec> {
        Moves {
            hots: self.hots(),
            colds: self.colds(),
            index: 0,
        }
    }
}

impl HotMoveInfo {
    fn new(move_evaluation: MoveEvaluation) -> Self {
        Self {
            move_evaluation,
            sum_evaluations: AtomicI64::default(),
            visits: AtomicU32::default(),
        }
    }
}
impl<'a, Spec: MCTS> ColdMoveInfo<Spec> {
    fn new(mov: chess::ChessMove) -> Self {
        Self {
            mov,
            child: AtomicPtr::default(),
        }
    }
}

impl<'a, Spec: MCTS> MoveInfoHandle<'a, Spec> {
    pub fn get_move(&self) -> &'a chess::ChessMove {
        &self.cold.mov
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

    pub fn child(&self) -> Option<NodeHandle<'a, Spec>> {
        let ptr = self.cold.child.load(Ordering::Relaxed);
        if ptr.is_null() {
            None
        } else {
            unsafe { Some(NodeHandle { node: &*ptr }) }
        }
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
    ch: CreationHelper<'a, 'b, Spec>,
) -> Result<SearchNode<Spec>, ArenaError> {
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
    let (move_eval, state_eval) = eval.evaluate_new_state(state, &moves);
    policy.validate_evaluations(&move_eval);
    let hots = allocator.alloc_slice(move_eval.len())?;
    let colds = allocator.alloc_slice(move_eval.len())?;
    for (x, y) in hots.iter_mut().zip(move_eval.into_iter()) {
        *x = HotMoveInfo::new(y);
    }
    for (x, y) in colds.iter_mut().zip(moves.into_iter()) {
        *x = ColdMoveInfo::new(y);
    }
    Ok(SearchNode::new(hots, colds, state_eval))
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
        let mut root_node = create_node(
            &eval,
            &tree_policy,
            &state,
            CreationHelper::Allocator(&arena.allocator()),
        )
        .expect("Unable to create root node");

        let _ = prev_table.lookup_into(&state, &mut root_node);

        let mut avg_rewards: Vec<f32> = root_node
            .moves()
            .into_iter()
            .map(|m| m.average_reward().unwrap_or(-SCALE) / SCALE)
            .collect();

        softmax(&mut avg_rewards);

        root_node.update_policy(avg_rewards);

        Self {
            root_state: state,
            root_node,
            manager,
            tree_policy,
            eval,
            table,
            prev_table,
            num_nodes: 1.into(),
            sum_depth: 1.into(),
            max_depth: 1.into(),
            tb_hits: 0.into(),
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

    pub fn average_depth(&self) -> usize {
        self.sum_depth.load(Ordering::SeqCst) / self.num_nodes()
    }

    pub fn max_depth(&self) -> usize {
        self.max_depth.load(Ordering::SeqCst)
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
        let mut path: SmallVec<[MoveInfoHandle<Spec>; LARGE_DEPTH]> = SmallVec::new();
        let mut node_path: SmallVec<[&SearchNode<Spec>; LARGE_DEPTH]> = SmallVec::new();
        let mut players: SmallVec<[Player; LARGE_DEPTH]> = SmallVec::new();
        let mut did_we_create = false;
        let mut node = &self.root_node;
        loop {
            if node.hots().is_empty() {
                break;
            }
            if path.len() >= self.manager.max_playout_length() {
                break;
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
            state.make_move(&choice.cold.mov);

            let (new_node, new_did_we_create) =
                match self.descend(&state, choice.cold, tld, &node_path) {
                    Ok(r) => r,
                    Err(ArenaError::Full) => {
                        debug!("Hash reached max capacity");
                        println!("info hashfull 1000");
                        return false;
                    }
                };

            node = new_node;
            did_we_create = new_did_we_create;
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
            if node.get_visits().load(Ordering::Relaxed) as u64
                <= self.manager.visits_before_expansion()
            {
                break;
            }
        }
        let new_evaln = if did_we_create {
            None
        } else {
            Some(node.evaln)
        };
        let evaln = new_evaln.as_ref().unwrap_or(&node.evaln);
        self.finish_playout(&path, &node_path, &players, evaln);

        self.sum_depth.fetch_add(node_path.len(), Ordering::Relaxed);

        if state.is_tb_hit() {
            self.tb_hits.fetch_add(1, Ordering::Relaxed);
        }

        true
    }

    fn descend<'a>(
        &'a self,
        state: &State,
        choice: &ColdMoveInfo<Spec>,
        tld: &mut ThreadData<'a, Spec>,
        path: &[&'a SearchNode<Spec>],
    ) -> Result<(&'a SearchNode<Spec>, bool), ArenaError> {
        let child = choice.child.load(Ordering::Relaxed) as *const _;
        if !child.is_null() {
            return unsafe { Ok((&*child, false)) };
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
                return Ok((node, false));
            } else {
                return unsafe { Ok((&*child, false)) };
            }
        }

        let mut created_here = create_node(
            &self.eval,
            &self.tree_policy,
            state,
            CreationHelper::Handle(self.make_handle(tld, path)),
        )?;

        let did_we_create = !self.prev_table.lookup_into(state, &mut created_here);

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
                return Ok((&*other_child, false));
            }
        }
        if let Some(existing) = self.table.insert(state, created) {
            self.delayed_transposition_table_hits
                .fetch_add(1, Ordering::Relaxed);
            let existing_ptr = existing as *const _ as *mut _;
            choice.child.store(existing_ptr, Ordering::Relaxed);
            return Ok((existing, false));
        }
        self.num_nodes.fetch_add(1, Ordering::Relaxed);
        let depth = path.len();
        self.sum_depth.fetch_add(depth, Ordering::Relaxed);
        self.max_depth.fetch_max(depth, Ordering::Relaxed);
        Ok((created, did_we_create))
    }

    fn finish_playout<'a>(
        &'a self,
        path: &[MoveInfoHandle<Spec>],
        node_path: &[&'a SearchNode<Spec>],
        players: &[Player],
        evaln: &StateEvaluation<Spec>,
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
        path: &'b [&'a SearchNode<Spec>],
    ) -> SearchHandle<'a, 'b, Spec> {
        let shared = SharedSearchHandle { tree: self, path };
        SearchHandle { shared, tld }
    }

    pub fn root_state(&self) -> &State {
        &self.root_state
    }
    pub fn root_node(&self) -> NodeHandle<Spec> {
        NodeHandle {
            node: &self.root_node,
        }
    }

    pub fn principal_variation(&self, num_moves: usize) -> Vec<MoveInfoHandle<Spec>> {
        let mut result = Vec::new();
        let mut crnt = &self.root_node;
        while !crnt.hots().is_empty() && result.len() < num_moves {
            let choice = self
                .manager
                .select_child_after_search(&crnt.moves().collect::<Vec<_>>());
            result.push(choice);
            let child = choice.cold.child.load(Ordering::SeqCst) as *const SearchNode<Spec>;
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

    pub fn diagnose(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!(
            "{} nodes\n",
            thousands_separate(self.num_nodes.load(Ordering::Relaxed))
        ));
        s.push_str(&format!(
            "{} transposition table hits\n",
            thousands_separate(self.transposition_table_hits.load(Ordering::Relaxed))
        ));
        s.push_str(&format!(
            "{} delayed transposition table hits\n",
            thousands_separate(
                self.delayed_transposition_table_hits
                    .load(Ordering::Relaxed)
            )
        ));
        s.push_str(&format!(
            "{} expansion contention events\n",
            thousands_separate(self.expansion_contention_events.load(Ordering::Relaxed))
        ));
        s
    }
}

pub struct NodeHandle<'a, Spec: 'a + MCTS> {
    node: &'a SearchNode<Spec>,
}
impl<'a, Spec: MCTS> Clone for NodeHandle<'a, Spec> {
    fn clone(&self) -> Self {
        Self { node: self.node }
    }
}
impl<'a, Spec: MCTS> Copy for NodeHandle<'a, Spec> {}

impl<'a, Spec: MCTS> NodeHandle<'a, Spec> {
    pub fn moves(&self) -> Moves<Spec> {
        self.node.moves()
    }
    pub fn into_raw(self) -> *const () {
        self.node as *const _ as *const ()
    }
    pub unsafe fn from_raw(ptr: *const ()) -> Self {
        NodeHandle {
            node: &*(ptr as *const SearchNode<Spec>),
        }
    }
}

pub struct Moves<'a, Spec: 'a + MCTS> {
    hots: &'a [HotMoveInfo],
    colds: &'a [ColdMoveInfo<Spec>],
    index: usize,
}

impl<'a, Spec: MCTS> Clone for Moves<'a, Spec> {
    fn clone(&self) -> Self {
        Self {
            hots: self.hots,
            colds: self.colds,
            index: self.index,
        }
    }
}

impl<'a, Spec: MCTS> Copy for Moves<'a, Spec> {}

impl<'a, Spec: 'a + MCTS> Iterator for Moves<'a, Spec> {
    type Item = MoveInfoHandle<'a, Spec>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.index == self.hots.len() {
            None
        } else {
            let handle = unsafe {
                MoveInfoHandle {
                    hot: self.hots.get_unchecked(self.index),
                    cold: self.colds.get_unchecked(self.index),
                }
            };
            self.index += 1;
            Some(handle)
        }
    }
}

pub struct SharedSearchHandle<'a: 'b, 'b, Spec: 'a + MCTS> {
    tree: &'a SearchTree<Spec>,
    path: &'b [&'a SearchNode<Spec>],
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

impl<'a, 'b, Spec: MCTS> SharedSearchHandle<'a, 'b, Spec> {
    pub fn node(&self) -> NodeHandle<'a, Spec> {
        self.nth_parent(0).unwrap()
    }
    pub fn parent(&self) -> Option<NodeHandle<'a, Spec>> {
        self.nth_parent(1)
    }
    pub fn grandparent(&self) -> Option<NodeHandle<'a, Spec>> {
        self.nth_parent(2)
    }
    pub fn mcts(&self) -> &'a Spec {
        &self.tree.manager
    }
    pub fn tree_policy(&self) -> &'a Spec::TreePolicy {
        &self.tree.tree_policy
    }
    pub fn evaluator(&self) -> &'a Spec::Eval {
        &self.tree.eval
    }
    /// The depth of the current search. A depth of 0 means we are at the root node.
    pub fn depth(&self) -> usize {
        self.path.len()
    }
    pub fn nth_parent(&self, n: usize) -> Option<NodeHandle<'a, Spec>> {
        if n >= self.path.len() {
            None
        } else {
            Some(NodeHandle {
                node: self.path[self.path.len() - n - 1],
            })
        }
    }
}

impl<'a, 'b, Spec: MCTS> SearchHandle<'a, 'b, Spec> {
    pub fn thread_data(&mut self) -> &mut ThreadData<'a, Spec> {
        self.tld
    }
    pub fn node(&self) -> NodeHandle<'a, Spec> {
        self.shared.node()
    }
    pub fn mcts(&self) -> &'a Spec {
        self.shared.mcts()
    }
    pub fn tree_policy(&self) -> &'a Spec::TreePolicy {
        self.shared.tree_policy()
    }
    pub fn evaluator(&self) -> &'a Spec::Eval {
        self.shared.evaluator()
    }
    pub fn depth(&self) -> usize {
        self.shared.depth()
    }
    pub fn nth_parent(&self, n: usize) -> Option<NodeHandle<'a, Spec>> {
        self.shared.nth_parent(n)
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

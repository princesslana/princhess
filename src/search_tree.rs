#![cfg_attr(feature = "nightly", feature(integer_atomics))]

use atomics::*;
use mcts::*;
use smallvec::SmallVec;
use std::fmt;
use std::fmt::{Debug, Display, Formatter};
use std::ptr::{null, null_mut};
use transposition_table::TranspositionTable;

use pod::Pod;

use tree_policy::TreePolicy;

use arena::{Arena, ArenaAllocator};

/// You're not intended to use this class (use an `MCTSManager` instead),
/// but you can use it if you want to manage the threads yourself.
pub struct SearchTree<Spec: MCTS> {
    root_node: SearchNode<Spec>,
    root_state: Spec::State,
    tree_policy: Spec::TreePolicy,
    table: Spec::TranspositionTable,
    eval: Spec::Eval,
    manager: Spec,
    arena: Box<Arena>,

    num_nodes: AtomicUsize,
    transposition_table_hits: AtomicUsize,
    delayed_transposition_table_hits: AtomicUsize,
    expansion_contention_events: AtomicUsize,
}

trait NodeStats {
    fn get_visits(&self) -> &FakeU32;
    fn get_sum_evaluations(&self) -> &AtomicI64;

    fn down<Spec: MCTS>(&self, manager: &Spec) {
        self.get_sum_evaluations()
            .fetch_sub(manager.virtual_loss() as FakeI64, Ordering::Relaxed);
        self.get_visits().fetch_add(1, Ordering::Relaxed);
    }
    fn up<Spec: MCTS>(&self, manager: &Spec, evaln: i64) {
        let delta = evaln + manager.virtual_loss();
        self.get_sum_evaluations()
            .fetch_add(delta as FakeI64, Ordering::Relaxed);
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

impl<Spec: MCTS> NodeStats for HotMoveInfo<Spec> {
    fn get_visits(&self) -> &FakeU32 {
        &self.visits
    }
    fn get_sum_evaluations(&self) -> &AtomicI64 {
        &self.sum_evaluations
    }
}
impl<Spec: MCTS> NodeStats for SearchNode<Spec> {
    fn get_visits(&self) -> &FakeU32 {
        &self.visits
    }
    fn get_sum_evaluations(&self) -> &AtomicI64 {
        &self.sum_evaluations
    }
}

struct HotMoveInfo<Spec: MCTS> {
    sum_evaluations: AtomicI64,
    visits: FakeU32,
    move_evaluation: MoveEvaluation<Spec>,
}
struct ColdMoveInfo<Spec: MCTS> {
    mov: Move<Spec>,
    child: AtomicPtr<SearchNode<Spec>>,
    owned: AtomicBool,
}
pub struct MoveInfoHandle<'a, Spec: 'a + MCTS> {
    hot: &'a HotMoveInfo<Spec>,
    cold: &'a ColdMoveInfo<Spec>,
}

unsafe impl<Spec: MCTS> Pod for HotMoveInfo<Spec> {}
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
    data: Spec::NodeData,
    evaln: StateEvaluation<Spec>,
    sum_evaluations: AtomicI64,
    visits: FakeU32,
}

unsafe impl<Spec: MCTS> Sync for SearchNode<Spec>
where
    Spec::NodeData: Sync,
    StateEvaluation<Spec>: Sync,
    // NodeStats: Sync,
    // for<'a> &'a[HotMoveInfo<Spec>]: Sync,
    // for<'a> &'a[ColdMoveInfo<Spec>]: Sync,
{
}

impl<Spec: MCTS> SearchNode<Spec> {
    fn new<'a>(
        hots: &'a [HotMoveInfo<Spec>],
        colds: &'a [ColdMoveInfo<Spec>],
        evaln: StateEvaluation<Spec>,
    ) -> Self {
        Self {
            hots: hots as *const _ as *const [()],
            colds: colds as *const _ as *const [()],
            data: Default::default(),
            evaln,
            visits: FakeU32::default(),
            sum_evaluations: AtomicI64::default(),
        }
    }
    fn hots<'a>(&'a self) -> &'a [HotMoveInfo<Spec>] {
        unsafe { &*(self.hots as *const [HotMoveInfo<Spec>]) }
    }
    fn colds<'a>(&'a self) -> &'a [ColdMoveInfo<Spec>] {
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

impl<Spec: MCTS> HotMoveInfo<Spec> {
    fn new(move_evaluation: MoveEvaluation<Spec>) -> Self {
        Self {
            move_evaluation,
            sum_evaluations: AtomicI64::default(),
            visits: FakeU32::default(),
        }
    }
}
impl<'a, Spec: MCTS> ColdMoveInfo<Spec> {
    fn new(mov: Move<Spec>) -> Self {
        Self {
            mov,
            child: AtomicPtr::default(),
            owned: AtomicBool::new(false),
        }
    }
}

impl<'a, Spec: MCTS> MoveInfoHandle<'a, Spec> {
    pub fn get_move(&self) -> &'a Move<Spec> {
        &self.cold.mov
    }

    pub fn move_evaluation(&self) -> &'a MoveEvaluation<Spec> {
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
        if ptr == null_mut() {
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

impl<'a, Spec: MCTS> Display for MoveInfoHandle<'a, Spec>
where
    Move<Spec>: Display,
{
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let own_str = if self.cold.owned.load(Ordering::Relaxed) {
            ""
        } else {
            " [child pointer is alias]"
        };
        if self.visits() == 0 {
            write!(f, "{} [0 visits]{}", self.get_move(), own_str)
        } else {
            write!(
                f,
                "{} [{} visit{}] [{} avg reward]{}",
                self.get_move(),
                self.visits(),
                if self.visits() == 1 { "" } else { "s" },
                self.sum_rewards() as f64 / self.visits() as f64,
                own_str
            )
        }
    }
}

impl<'a, Spec: MCTS> Debug for MoveInfoHandle<'a, Spec>
where
    Move<Spec>: Debug,
{
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let own_str = if self.cold.owned.load(Ordering::Relaxed) {
            ""
        } else {
            " [child pointer is alias]"
        };
        if self.visits() == 0 {
            write!(f, "{:?} [0 visits]{}", self.get_move(), own_str)
        } else {
            write!(
                f,
                "{:?} [{} visit{}] [{} avg reward]{}",
                self.get_move(),
                self.visits(),
                if self.visits() == 1 { "" } else { "s" },
                self.sum_rewards() as f64 / self.visits() as f64,
                own_str
            )
        }
    }
}

enum CreationHelper<'a: 'b, 'b, Spec: 'a + MCTS> {
    Handle(SearchHandle<'a, 'b, Spec>),
    Allocator(&'b ArenaAllocator<'a>),
}

#[inline(always)]
fn create_node<'a, 'b, 'c, Spec: MCTS>(
    eval: &Spec::Eval,
    policy: &Spec::TreePolicy,
    state: &Spec::State,
    ch: CreationHelper<'a, 'b, Spec>,
) -> SearchNode<Spec> {
    let (allocator, handle) = match ch {
        CreationHelper::Allocator(x) => (x, None),
        CreationHelper::Handle(x) => {
            // this is safe because nothing will move into x.tld.allocator
            // since ThreadData.allocator is a private field
            let allocator = unsafe { &*(&x.tld.allocator as *const _) };
            (allocator, Some(x))
        }
    };
    let moves = state.available_moves();
    let (move_eval, state_eval) = eval.evaluate_new_state(&state, &moves, handle);
    policy.validate_evaluations(&move_eval);
    let hots = allocator.alloc_slice(move_eval.len());
    let colds = allocator.alloc_slice(move_eval.len());
    for (x, y) in hots.iter_mut().zip(move_eval.into_iter()) {
        *x = HotMoveInfo::new(y);
    }
    for (x, y) in colds.iter_mut().zip(moves.into_iter()) {
        *x = ColdMoveInfo::new(y);
    }
    SearchNode::new(hots, colds, state_eval)
}

fn is_cycle<T>(past: &[&T], current: &T) -> bool {
    past.iter().any(|x| *x as *const T == current as *const T)
}

impl<Spec: MCTS> SearchTree<Spec> {
    pub fn new(
        state: Spec::State,
        manager: Spec,
        tree_policy: Spec::TreePolicy,
        eval: Spec::Eval,
        table: Spec::TranspositionTable,
    ) -> Self {
        let arena = Box::new(Arena::new());
        let root_node = create_node(
            &eval,
            &tree_policy,
            &state,
            CreationHelper::Allocator(&arena.allocator()),
        );
        Self {
            root_state: state,
            root_node,
            manager,
            tree_policy,
            eval,
            table,
            num_nodes: 1.into(),
            arena,
            transposition_table_hits: 0.into(),
            delayed_transposition_table_hits: 0.into(),
            expansion_contention_events: 0.into(),
        }
    }

    pub fn reset(self) -> Self {
        Self::new(
            self.root_state,
            self.manager,
            self.tree_policy.reset(),
            self.eval,
            self.table,
        )
    }

    pub fn spec(&self) -> &Spec {
        &self.manager
    }

    pub fn num_nodes(&self) -> usize {
        self.num_nodes.load(Ordering::SeqCst)
    }

    pub fn arena(&self) -> &Arena {
        &self.arena
    }

    #[inline(never)]
    pub fn playout<'a: 'b, 'b>(&'a self, tld: &'b mut ThreadData<'a, Spec>) -> bool {
        const LARGE_DEPTH: usize = 64;
        let sentinel = IncreaseSentinel::new(&self.num_nodes);
        if sentinel.num_nodes >= self.manager.node_limit() {
            return false;
        }
        let mut state = self.root_state.clone();
        let mut playout_data = Spec::PlayoutData::default();
        let mut path: SmallVec<[MoveInfoHandle<Spec>; LARGE_DEPTH]> = SmallVec::new();
        let mut node_path: SmallVec<[&SearchNode<Spec>; LARGE_DEPTH]> = SmallVec::new();
        let mut players: SmallVec<[Player<Spec>; LARGE_DEPTH]> = SmallVec::new();
        let mut did_we_create = false;
        let mut node = &self.root_node;
        loop {
            if node.hots().len() == 0 {
                break;
            }
            if path.len() >= self.manager.max_playout_length() {
                break;
            }
            let choice = match self
                .manager
                .override_policy(&playout_data, &state, node.moves())
            {
                Some(choice) => choice,
                None => self.tree_policy.choose_child(
                    &state,
                    node.moves(),
                    self.make_handle(tld, &node_path),
                ),
            };
            self.manager.on_choice_made(
                &mut playout_data,
                &state,
                node.moves(),
                choice,
                self.make_handle(tld, &node_path),
            );
            choice.hot.down(&self.manager);
            players.push(state.current_player());
            path.push(choice);
            assert!(path.len() <= self.manager.max_playout_length(),
                "playout length exceeded maximum of {} (maybe the transposition table is creating an infinite loop?)",
                self.manager.max_playout_length());
            state.make_move(&choice.cold.mov);
            let (new_node, new_did_we_create) = self.descend(&state, choice.cold, tld, &node_path);
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
                        self.finish_playout(&path, &node_path, &players, tld, &e);
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
            Some(self.eval.evaluate_existing_state(
                &state,
                &node.evaln,
                self.make_handle(tld, &node_path),
            ))
        };
        let evaln = new_evaln.as_ref().unwrap_or(&node.evaln);
        self.finish_playout(&path, &node_path, &players, tld, evaln);
        true
    }

    fn descend<'a>(
        &'a self,
        state: &Spec::State,
        choice: &ColdMoveInfo<Spec>,
        tld: &mut ThreadData<'a, Spec>,
        path: &[&'a SearchNode<Spec>],
    ) -> (&'a SearchNode<Spec>, bool) {
        let child = choice.child.load(Ordering::Relaxed) as *const _;
        if child != null() {
            return unsafe { (&*child, false) };
        }
        if let Some(node) = self.table.lookup(state, self.make_handle(tld, path)) {
            let child = choice.child.compare_and_swap(
                null_mut(),
                node as *const _ as *mut _,
                Ordering::Relaxed,
            ) as *const _;
            if child == null() {
                self.transposition_table_hits
                    .fetch_add(1, Ordering::Relaxed);
                return (node, false);
            } else {
                return unsafe { (&*child, false) };
            }
        }
        let created_here = create_node(
            &self.eval,
            &self.tree_policy,
            state,
            CreationHelper::Handle(self.make_handle(tld, path)),
        );
        let created = tld.allocator.alloc_one();
        *created = created_here;
        let other_child =
            choice
                .child
                .compare_and_swap(null_mut(), created as *mut _, Ordering::Relaxed);
        if other_child != null_mut() {
            self.expansion_contention_events
                .fetch_add(1, Ordering::Relaxed);
            unsafe {
                return (&*other_child, false);
            }
        }
        if let Some(existing) = self
            .table
            .insert(state, created, self.make_handle(tld, path))
        {
            self.delayed_transposition_table_hits
                .fetch_add(1, Ordering::Relaxed);
            let existing_ptr = existing as *const _ as *mut _;
            choice.child.store(existing_ptr, Ordering::Relaxed);
            return (existing, false);
        }
        choice.owned.store(true, Ordering::Relaxed);
        self.num_nodes.fetch_add(1, Ordering::Relaxed);
        (created, true)
    }

    fn finish_playout<'a>(
        &'a self,
        path: &[MoveInfoHandle<Spec>],
        node_path: &[&'a SearchNode<Spec>],
        players: &[Player<Spec>],
        tld: &mut ThreadData<'a, Spec>,
        evaln: &StateEvaluation<Spec>,
    ) {
        for ((move_info, player), node) in
            path.iter().zip(players.iter()).zip(node_path.iter()).rev()
        {
            let evaln_value = self.eval.interpret_evaluation_for_player(evaln, player);
            node.up(&self.manager, evaln_value);
            move_info.hot.replace(*node);
            self.manager
                .on_backpropagation(&evaln, self.make_handle(tld, node_path));
        }
        self.manager
            .on_backpropagation(&evaln, self.make_handle(tld, node_path));
    }

    fn make_handle<'a, 'b>(
        &'a self,
        tld: &'b mut ThreadData<'a, Spec>,
        path: &'b [&'a SearchNode<Spec>],
    ) -> SearchHandle<'a, 'b, Spec> {
        let shared = SharedSearchHandle { tree: self, path };
        SearchHandle { shared, tld }
    }

    pub fn root_state(&self) -> &Spec::State {
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
        while crnt.hots().len() != 0 && result.len() < num_moves {
            let choice = self
                .manager
                .select_child_after_search(&crnt.moves().collect::<Vec<_>>());
            result.push(choice);
            let child = choice.cold.child.load(Ordering::SeqCst) as *const SearchNode<Spec>;
            if child == null() {
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

impl<Spec: MCTS> SearchTree<Spec>
where
    Move<Spec>: Debug,
{
    pub fn debug_moves(&self) {
        let mut moves: Vec<MoveInfoHandle<Spec>> = self.root_node.moves().collect();
        moves.sort_by_key(|x| -(x.visits() as i64));
        for mov in moves {
            println!("{:?}", mov);
        }
    }
}

impl<Spec: MCTS> SearchTree<Spec>
where
    Move<Spec>: Display,
{
    pub fn display_moves(&self) {
        let mut moves: Vec<MoveInfoHandle<Spec>> = self.root_node.moves().collect();
        moves.sort_by_key(|x| -(x.visits() as i64));
        for mov in moves {
            println!("{}", mov);
        }
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
    pub fn data(&self) -> &'a Spec::NodeData {
        &self.node.data
    }
    pub fn moves(&self) -> Moves<Spec> {
        self.node.moves()
    }
    pub fn into_raw(&self) -> *const () {
        self.node as *const _ as *const ()
    }
    pub unsafe fn from_raw(ptr: *const ()) -> Self {
        NodeHandle {
            node: &*(ptr as *const SearchNode<Spec>),
        }
    }
}

pub struct Moves<'a, Spec: 'a + MCTS> {
    hots: &'a [HotMoveInfo<Spec>],
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

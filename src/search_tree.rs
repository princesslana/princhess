use arrayvec::ArrayVec;
use pod::Pod;
use shakmaty::{Color, MoveList, Position};
use std::mem;
use std::ptr::null_mut;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicPtr, AtomicU32, AtomicUsize, Ordering};
use std::sync::Mutex;

use crate::arena::{Arena, ArenaAllocator, ArenaError};
use crate::evaluation;
use crate::math;
use crate::mcts::*;
use crate::search::SCALE;
use crate::state::State;
use crate::transposition_table::TranspositionTable;
use crate::tree_policy::Puct;

const MAX_PLAYOUT_LENGTH: usize = 256;

const VIRTUAL_LOSS: i64 = SCALE as i64;

/// You're not intended to use this class (use an `MctsManager` instead),
/// but you can use it if you want to manage the threads yourself.
pub struct SearchTree {
    root_node: SearchNode,
    root_state: State,
    tree_policy: Puct,
    #[allow(dead_code)]
    root_table: TranspositionTable,
    left_table: TranspositionTable,
    right_table: TranspositionTable,
    is_left_current: AtomicBool,

    flip_lock: Mutex<()>,
    num_nodes: AtomicUsize,
    tb_hits: AtomicUsize,
}

pub trait NodeStats {
    fn get_visits(&self) -> &AtomicU32;
    fn get_sum_evaluations(&self) -> &AtomicI64;

    fn down(&self) {
        self.get_sum_evaluations()
            .fetch_sub(VIRTUAL_LOSS, Ordering::Relaxed);
        self.get_visits().fetch_add(1, Ordering::Relaxed);
    }
    fn up(&self, evaln: i64) {
        let delta = evaln + VIRTUAL_LOSS;
        self.get_sum_evaluations()
            .fetch_add(delta, Ordering::Relaxed);
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

pub struct HotMoveInfo {
    sum_evaluations: AtomicI64,
    visits: AtomicU32,
    move_evaluation: MoveEvaluation,
    mov: shakmaty::Move,
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
    tablebase: bool,
    sum_evaluations: AtomicI64,
    visits: AtomicU32,
}

unsafe impl Sync for SearchNode {}

impl SearchNode {
    fn new(hots: &[HotMoveInfo], evaln: StateEvaluation, tablebase: bool) -> Self {
        Self {
            hots: hots as *const _ as *const [()],
            evaln,
            tablebase,
            visits: AtomicU32::default(),
            sum_evaluations: AtomicI64::default(),
        }
    }

    pub fn evaln(&self) -> StateEvaluation {
        self.evaln
    }

    pub fn set_evaln(&mut self, evaln: StateEvaluation) {
        self.evaln = evaln;
    }

    pub fn hots(&self) -> &[HotMoveInfo] {
        unsafe { &*(self.hots as *const [HotMoveInfo]) }
    }

    fn update_policy(&mut self, evals: Vec<f32>) {
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

    pub fn clear_children_links(&self) {
        let hots = unsafe { &*(self.hots as *mut [HotMoveInfo]) };

        for h in hots {
            h.child.store(null_mut(), Ordering::SeqCst);
        }
    }
}

impl HotMoveInfo {
    fn new(move_evaluation: MoveEvaluation, mov: shakmaty::Move) -> Self {
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
    pub fn get_move(&self) -> &'a shakmaty::Move {
        &self.hot.mov
    }

    pub fn move_evaluation(&self) -> &'a MoveEvaluation {
        &self.hot.move_evaluation
    }

    pub fn visits(&self) -> u64 {
        self.hot.visits.load(Ordering::Relaxed) as u64
    }

    pub fn sum_rewards(&self) -> i64 {
        self.hot.sum_evaluations.load(Ordering::Relaxed)
    }

    pub fn average_reward(&self) -> Option<f32> {
        match self.visits() {
            0 => None,
            x => Some(self.sum_rewards() as f32 / x as f32),
        }
    }
}

enum CreationHelper<'a: 'b, 'b> {
    Handle(bool, SearchHandle<'a, 'b>),
    Allocator(&'b ArenaAllocator<'a>),
}

#[inline(always)]
fn create_node(
    policy: &Puct,
    state: &State,
    tb_hits: &AtomicUsize,
    ch: CreationHelper<'_, '_>,
) -> Result<SearchNode, ArenaError> {
    let (allocator, _handle) = match ch {
        CreationHelper::Allocator(x) => (x, None),
        CreationHelper::Handle(is_left, x) => {
            // this is safe because nothing will move into x.tld.allocator
            // since ThreadData.allocator is a private field
            let allocator_ref = if is_left {
                &x.tld.allocator.0
            } else {
                &x.tld.allocator.1
            };
            let allocator = unsafe { &*(allocator_ref as *const _) };
            (allocator, Some(x))
        }
    };
    let mut moves = MoveList::new();

    let (move_eval, state_eval, is_tb_hit) =
        if state.drawn_by_repetition() || state.board().is_insufficient_material() {
            (Vec::with_capacity(0), 0, false)
        } else {
            moves = state.available_moves();
            evaluation::evaluate_new_state(state, &moves)
        };

    if is_tb_hit {
        tb_hits.fetch_add(1, Ordering::Relaxed);
    }

    policy.validate_evaluations(&move_eval);
    let hots = allocator.alloc_slice(move_eval.len())?;
    for (i, x) in hots.iter_mut().enumerate() {
        *x = HotMoveInfo::new(move_eval[i], moves[i].clone());
    }
    Ok(SearchNode::new(hots, state_eval, is_tb_hit))
}

impl SearchTree {
    pub fn new(
        state: State,
        tree_policy: Puct,
        current_table: TranspositionTable,
        previous_table: TranspositionTable,
    ) -> Self {
        let tb_hits = 0.into();

        let root_table = TranspositionTable::for_root();

        let mut root_node = create_node(
            &tree_policy,
            &state,
            &tb_hits,
            CreationHelper::Allocator(&root_table.arena().allocator()),
        )
        .expect("Unable to create root node");

        previous_table.lookup_into(&state, &mut root_node);

        let mut avg_rewards: Vec<f32> = root_node
            .moves()
            .into_iter()
            .map(|m| m.average_reward().unwrap_or(-SCALE) / SCALE)
            .collect();

        math::softmax(&mut avg_rewards);

        root_node.update_policy(avg_rewards);

        Self {
            root_state: state,
            root_node,
            tree_policy,
            root_table,
            left_table: current_table,
            right_table: previous_table,
            is_left_current: AtomicBool::new(true),
            flip_lock: Default::default(),
            num_nodes: 1.into(),
            tb_hits,
        }
    }

    fn is_left_current(&self) -> bool {
        self.is_left_current.load(Ordering::Relaxed)
    }

    fn current_table(&self) -> &TranspositionTable {
        if self.is_left_current() {
            &self.left_table
        } else {
            &self.right_table
        }
    }

    fn previous_table(&self) -> &TranspositionTable {
        if self.is_left_current() {
            &self.right_table
        } else {
            &self.left_table
        }
    }

    fn flip_tables(&self) {
        self.previous_table().clear();
        self.is_left_current.store(
            !self.is_left_current.load(Ordering::SeqCst),
            Ordering::SeqCst,
        );
    }

    pub fn table(self) -> TranspositionTable {
        if self.is_left_current() {
            self.left_table
        } else {
            self.right_table
        }
    }

    pub fn num_nodes(&self) -> usize {
        self.num_nodes.load(Ordering::Relaxed)
    }

    pub fn tb_hits(&self) -> usize {
        self.tb_hits.load(Ordering::Relaxed)
    }

    pub fn arenas(&self) -> (&Arena, &Arena) {
        (self.left_table.arena(), self.right_table.arena())
    }

    #[inline(never)]
    pub fn playout<'a: 'b, 'b>(&'a self, tld: &'b mut ThreadData<'a>) -> bool {
        self.num_nodes.fetch_add(1, Ordering::Relaxed);
        let mut state = self.root_state.clone();
        let mut path: ArrayVec<MoveInfoHandle, MAX_PLAYOUT_LENGTH> = ArrayVec::new();
        let mut node_path: ArrayVec<&SearchNode, MAX_PLAYOUT_LENGTH> = ArrayVec::new();
        let mut node = &self.root_node;
        loop {
            {
                let _lock = self.flip_lock.lock().unwrap();
            }
            if node.hots().is_empty() || state.drawn_by_fifty_move_rule() {
                break;
            }
            // We need path.len() check for when the root node is a tablebase position
            if node.tablebase && path.len() > 2 {
                break;
            }
            if path.len() >= MAX_PLAYOUT_LENGTH {
                break;
            }
            let choice = self.tree_policy.choose_child(
                &state,
                node.moves(),
                self.make_handle(tld, &node_path),
            );
            choice.hot.down();
            path.push(choice);
            state.make_move(&choice.hot.mov);

            let new_node = match self.descend(&state, choice.hot, tld, &node_path) {
                Ok(r) => r,
                Err(ArenaError::Full) => {
                    let _lock = self.flip_lock.lock().unwrap();
                    if self.current_table().arena().full() {
                        self.flip_tables();
                        self.root_node.clear_children_links();
                    }
                    return true;
                }
            };

            node = new_node;

            node_path.push(node);
            node.down();
            if node.get_visits().load(Ordering::Relaxed) == 1 {
                break;
            }
        }

        let last_move_was_black = state.side_to_move() == Color::White;

        let evaln = if state.drawn_by_fifty_move_rule() {
            0
        } else if last_move_was_black {
            -node.evaln
        } else {
            node.evaln
        };

        self.finish_playout(&path, &node_path, evaln);

        true
    }

    fn descend<'a>(
        &'a self,
        state: &State,
        choice: &HotMoveInfo,
        tld: &mut ThreadData<'a>,
        path: &[&'a SearchNode],
    ) -> Result<&'a SearchNode, ArenaError> {
        let child = choice.child.load(Ordering::Relaxed) as *const SearchNode;
        if !child.is_null() {
            return unsafe { Ok(&*child) };
        }

        if let Some(node) = self.current_table().lookup(state) {
            return match choice.child.compare_exchange(
                null_mut(),
                node as *const _ as *mut _,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => Ok(node),
                Err(_) => unsafe { Ok(&*child) },
            };
        }

        let is_left_current = self.is_left_current();

        let mut created_here = create_node(
            &self.tree_policy,
            state,
            &self.tb_hits,
            CreationHelper::Handle(is_left_current, self.make_handle(tld, path)),
        )?;

        self.previous_table().lookup_into(state, &mut created_here);

        let created = if is_left_current {
            tld.allocator.0.alloc_one()?
        } else {
            tld.allocator.1.alloc_one()?
        };

        *created = created_here;
        let other_child = choice.child.compare_exchange(
            null_mut(),
            created as *mut _,
            Ordering::Relaxed,
            Ordering::Relaxed,
        );
        if let Err(v) = other_child {
            unsafe {
                return Ok(&*v);
            }
        }

        if let Some(existing) = self.current_table().insert(state, created) {
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
        evaln: StateEvaluation,
    ) {
        let mut evaln_value = evaln;
        for (move_info, node) in path.iter().zip(node_path.iter()).rev() {
            node.up(evaln_value);
            move_info.hot.replace(*node);
            evaln_value = -evaln_value;
        }
    }

    fn make_handle<'a, 'b>(
        &'a self,
        tld: &'b mut ThreadData<'a>,
        path: &'b [&'a SearchNode],
    ) -> SearchHandle<'a, 'b> {
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
            let choice = select_child_after_search(&crnt.moves().collect::<Vec<_>>());
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

fn select_child_after_search<'a>(children: &[MoveInfoHandle<'a>]) -> MoveInfoHandle<'a> {
    *children
        .iter()
        .max_by_key(|child| {
            child
                .average_reward()
                .map(|r| r.round() as i64)
                .unwrap_or(-SCALE as i64)
        })
        .unwrap()
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

pub struct SharedSearchHandle<'a: 'b, 'b> {
    tree: &'a SearchTree,
    path: &'b [&'a SearchNode],
}
impl<'a: 'b, 'b> Clone for SharedSearchHandle<'a, 'b> {
    fn clone(&self) -> Self {
        let tree = self.tree;
        let path = self.path;
        Self { tree, path }
    }
}
impl<'a: 'b, 'b> Copy for SharedSearchHandle<'a, 'b> {}

pub struct SearchHandle<'a: 'b, 'b> {
    pub tld: &'b mut ThreadData<'a>,
    pub shared: SharedSearchHandle<'a, 'b>,
}

impl<'a, 'b> SearchHandle<'a, 'b> {
    pub fn is_root(&self) -> bool {
        self.shared.path.is_empty()
    }
}

pub fn print_size_list() {
    println!(
        "info string SearchNode {} HotMoveInfo {}",
        mem::size_of::<SearchNode>(),
        mem::size_of::<HotMoveInfo>(),
    );
}

use arrayvec::ArrayVec;
use shakmaty::{Color, Position};
use std::mem;
use std::ptr::null_mut;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicPtr, AtomicU32, AtomicUsize, Ordering};

use crate::arena::Error as ArenaError;
use crate::evaluation::{self, Evaluation};
use crate::math;
use crate::mcts::{eval_in_cp, ThreadData};
use crate::options::get_cpuct;
use crate::search::{to_uci, TimeManagement, SCALE};
use crate::state::State;
use crate::transposition_table::{LRAllocator, LRTable, TranspositionTable};
use crate::tree_policy;

const MAX_PLAYOUT_LENGTH: usize = 256;

const VIRTUAL_LOSS: i64 = SCALE as i64;

/// You're not intended to use this class (use an `MctsManager` instead),
/// but you can use it if you want to manage the threads yourself.
pub struct SearchTree {
    root_node: SearchNode,
    root_state: State,

    cpuct: f32,

    #[allow(dead_code)]
    root_table: TranspositionTable,
    ttable: LRTable,

    num_nodes: AtomicUsize,
    playouts: AtomicUsize,
    max_depth: AtomicUsize,
    tb_hits: AtomicUsize,
}

pub struct HotMoveInfo {
    sum_evaluations: AtomicI64,
    visits: AtomicU32,
    policy: f32,
    mov: shakmaty::Move,
    child: AtomicPtr<SearchNode>,
}

#[derive(Clone, Copy)]
pub struct MoveInfoHandle<'a> {
    hot: &'a HotMoveInfo,
}

pub struct SearchNode {
    hots: *const [HotMoveInfo],
    evaln: Evaluation,
    visited: AtomicBool,
}

unsafe impl Sync for SearchNode {}

static DRAW_NODE: SearchNode = SearchNode::new(&[], Evaluation::draw());

impl SearchNode {
    const fn new(hots: &[HotMoveInfo], evaln: Evaluation) -> Self {
        Self {
            hots,
            evaln,
            visited: AtomicBool::new(false),
        }
    }

    pub fn evaln(&self) -> &Evaluation {
        &self.evaln
    }

    pub fn set_evaln(&mut self, evaln: Evaluation) {
        self.evaln = evaln;
    }

    pub fn visited(&self) {
        self.visited.store(true, Ordering::Relaxed);
    }

    pub fn is_first_visit(&self) -> bool {
        !self.visited.load(Ordering::Relaxed)
    }

    pub fn is_terminal(&self) -> bool {
        self.evaln.is_terminal()
    }

    pub fn is_tablebase(&self) -> bool {
        self.evaln.is_tablebase()
    }

    pub fn hots(&self) -> &[HotMoveInfo] {
        unsafe { &*(self.hots as *const [HotMoveInfo]) }
    }

    fn update_policy(&mut self, evals: &[f32]) {
        let mut hots = unsafe { &mut *(self.hots as *mut [HotMoveInfo]) };

        for i in 0..hots.len().min(evals.len()) {
            hots[i].policy = evals[i];
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
    fn new(policy: f32, mov: shakmaty::Move) -> Self {
        Self {
            policy,
            sum_evaluations: AtomicI64::default(),
            visits: AtomicU32::default(),
            mov,
            child: AtomicPtr::default(),
        }
    }

    pub fn get_visits(&self) -> &AtomicU32 {
        &self.visits
    }

    pub fn get_sum_evaluations(&self) -> &AtomicI64 {
        &self.sum_evaluations
    }

    pub fn down(&self) {
        self.get_sum_evaluations()
            .fetch_sub(VIRTUAL_LOSS, Ordering::Relaxed);
        self.get_visits().fetch_add(1, Ordering::Relaxed);
    }

    pub fn up(&self, evaln: i64) {
        let delta = evaln + VIRTUAL_LOSS;
        self.get_sum_evaluations()
            .fetch_add(delta, Ordering::Relaxed);
    }

    pub fn replace(&self, other: &HotMoveInfo) {
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

impl<'a> MoveInfoHandle<'a> {
    pub fn get_move(self) -> &'a shakmaty::Move {
        &self.hot.mov
    }

    pub fn policy(self) -> f32 {
        self.hot.policy
    }

    pub fn visits(self) -> u32 {
        self.hot.visits.load(Ordering::Relaxed)
    }

    pub fn sum_rewards(self) -> i64 {
        self.hot.sum_evaluations.load(Ordering::Relaxed)
    }

    pub fn average_reward(self) -> Option<f32> {
        match self.visits() {
            0 => None,
            x => Some(self.sum_rewards() as f32 / x as f32),
        }
    }
}

fn create_node<'a, F>(
    state: &State,
    tb_hits: &AtomicUsize,
    alloc_slice: F,
) -> Result<SearchNode, ArenaError>
where
    F: FnOnce(usize) -> Result<&'a mut [HotMoveInfo], ArenaError>,
{
    let moves = state.available_moves();

    let (move_eval, state_eval) = evaluation::evaluate_new_state(state, &moves);

    if state_eval.is_tablebase() {
        tb_hits.fetch_add(1, Ordering::Relaxed);
    }

    let hots = alloc_slice(move_eval.len())?;
    for (i, x) in hots.iter_mut().enumerate() {
        *x = HotMoveInfo::new(move_eval[i], moves[i].clone());
    }
    Ok(SearchNode::new(hots, state_eval))
}

impl SearchTree {
    pub fn new(
        state: State,
        current_table: TranspositionTable,
        previous_table: TranspositionTable,
    ) -> Self {
        let tb_hits = 0.into();

        let root_table = TranspositionTable::for_root();

        let mut root_node = create_node(&state, &tb_hits, |sz| {
            root_table.arena().allocator().alloc_slice(sz)
        })
        .expect("Unable to create root node");

        previous_table.lookup_into(&state, &mut root_node);

        let mut avg_rewards: Vec<f32> = root_node
            .moves()
            .into_iter()
            .map(|m| m.average_reward().unwrap_or(-SCALE) / SCALE)
            .collect();

        math::softmax(&mut avg_rewards);

        root_node.update_policy(&avg_rewards);

        Self {
            root_state: state,
            root_node,
            cpuct: get_cpuct(),
            root_table,
            ttable: LRTable::new(current_table, previous_table),
            num_nodes: 1.into(),
            playouts: 0.into(),
            max_depth: 0.into(),
            tb_hits,
        }
    }

    fn flip_tables(&self) {
        self.ttable.flip_tables();
    }

    pub fn table(self) -> TranspositionTable {
        self.ttable.table()
    }

    pub fn num_nodes(&self) -> usize {
        self.num_nodes.load(Ordering::Relaxed)
    }

    pub fn playouts(&self) -> usize {
        self.playouts.load(Ordering::Relaxed)
    }

    pub fn max_depth(&self) -> usize {
        self.max_depth.load(Ordering::Relaxed)
    }

    pub fn tb_hits(&self) -> usize {
        self.tb_hits.load(Ordering::Relaxed)
    }

    pub fn allocator(&self) -> LRAllocator {
        self.ttable.allocator()
    }

    #[inline(never)]
    pub fn playout<'a: 'b, 'b>(
        &'a self,
        tld: &'b mut ThreadData<'a>,
        time_management: TimeManagement,
    ) -> bool {
        let mut state = self.root_state.clone();
        let mut path: ArrayVec<MoveInfoHandle, MAX_PLAYOUT_LENGTH> = ArrayVec::new();
        let mut node = &self.root_node;
        loop {
            {
                let _lock = self.ttable.flip_lock().lock().unwrap();
            }
            if node.is_terminal() {
                break;
            }
            if node.hots().is_empty() {
                break;
            }
            if node.is_tablebase() && state.halfmove_counter() == 0 {
                break;
            }
            if path.len() >= MAX_PLAYOUT_LENGTH {
                break;
            }
            let choice = tree_policy::choose_child(node.moves(), self.cpuct, path.is_empty());
            choice.hot.down();
            path.push(choice);
            state.make_move(&choice.hot.mov);

            let new_node = match self.descend(&state, choice.hot, tld) {
                Ok(r) => r,
                Err(ArenaError::Full) => {
                    let _lock = self.ttable.flip_lock().lock().unwrap();
                    if self.ttable.is_arena_full() {
                        self.flip_tables();
                        self.root_node.clear_children_links();
                    }
                    return true;
                }
            };

            node = new_node;

            if node.is_first_visit() {
                node.visited();
                break;
            }
        }

        let last_move_was_black = state.side_to_move() == Color::White;

        let evaln = if last_move_was_black {
            node.evaln.flip()
        } else {
            node.evaln
        };

        Self::finish_playout(&path, evaln);

        // -1 because we don't count the root node
        let depth = path.len() - 1;
        self.num_nodes.fetch_add(depth, Ordering::Relaxed);
        self.max_depth.fetch_max(depth, Ordering::Relaxed);
        let playouts = self.playouts.fetch_add(1, Ordering::Relaxed);

        if playouts > 4096 && (playouts & (playouts - 1)) == 0 {
            self.print_info(&time_management);
        }

        if playouts % 128 == 0 && time_management.is_after_end() {
            self.print_info(&time_management);
            return false;
        }

        true
    }

    fn descend<'a>(
        &'a self,
        state: &State,
        choice: &HotMoveInfo,
        tld: &mut ThreadData<'a>,
    ) -> Result<&'a SearchNode, ArenaError> {
        if state.is_repetition()
            || state.drawn_by_fifty_move_rule()
            || state.board().is_insufficient_material()
        {
            return Ok(&DRAW_NODE);
        }

        let child = choice.child.load(Ordering::Relaxed) as *const SearchNode;
        if !child.is_null() {
            return unsafe { Ok(&*child) };
        }

        if let Some(node) = self.ttable.lookup(state) {
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

        let mut created_here =
            create_node(state, &self.tb_hits, |sz| tld.allocator.alloc_move_info(sz))?;

        self.ttable.lookup_into(state, &mut created_here);

        let created = tld.allocator.alloc_node()?;

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

        if let Some(existing) = self.ttable.insert(state, created) {
            let existing_ptr = existing as *const _ as *mut _;
            choice.child.store(existing_ptr, Ordering::Relaxed);
            return Ok(existing);
        }
        Ok(created)
    }

    fn finish_playout(path: &[MoveInfoHandle], evaln: Evaluation) {
        let mut evaln_value = evaln;
        for move_info in path.iter().rev() {
            move_info.hot.up(evaln_value.into());
            evaln_value = evaln_value.flip();
        }
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
            }
            unsafe {
                crnt = &*child;
            }
        }
        result
    }

    fn print_info(&self, time_management: &TimeManagement) {
        let search_time_ms = time_management.elapsed().as_millis();

        if search_time_ms == 0 {
            return;
        }

        let nodes = self.num_nodes();
        let depth = nodes / self.playouts();
        let sel_depth = self.max_depth();
        let pv = self.principal_variation(depth);
        let pv_string: String = pv
            .into_iter()
            .map(|x| format!(" {}", to_uci(x.get_move())))
            .collect();

        let nps = nodes * 1000 / search_time_ms as usize;

        let info_str = format!(
            "info depth {} seldepth {} nodes {} nps {} tbhits {} score {} time {} pv{}",
            depth,
            sel_depth,
            nodes,
            nps,
            self.tb_hits(),
            self.eval_in_cp(),
            search_time_ms,
            pv_string,
        );
        println!("{info_str}");
    }

    pub fn eval(&self) -> f32 {
        self.principal_variation(1)
            .get(0)
            .map_or(0., |x| (x.sum_rewards() / i64::from(x.visits())) as f32 / SCALE)
    }

    fn eval_in_cp(&self) -> String {
        eval_in_cp(self.eval())
    }
}

fn select_child_after_search<'a>(children: &[MoveInfoHandle<'a>]) -> MoveInfoHandle<'a> {
    *children
        .iter()
        .max_by_key(|child| {
            child
                .average_reward()
                .map_or(-SCALE as i64, |r| r.round() as i64)
        })
        .unwrap()
}

#[derive(Clone)]
pub struct NodeHandle<'a> {
    node: &'a SearchNode,
}

impl<'a> Copy for NodeHandle<'a> {}

impl<'a> NodeHandle<'a> {
    pub fn moves(&self) -> Moves {
        self.node.moves()
    }
}

#[derive(Clone)]
pub struct Moves<'a> {
    hots: &'a [HotMoveInfo],
    index: usize,
}

impl<'a> Copy for Moves<'a> {}

#[allow(clippy::copy_iterator)]
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

pub fn print_size_list() {
    println!(
        "info string SearchNode {} HotMoveInfo {}",
        mem::size_of::<SearchNode>(),
        mem::size_of::<HotMoveInfo>(),
    );
}

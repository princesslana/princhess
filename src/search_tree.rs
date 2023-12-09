use arrayvec::ArrayVec;
use shakmaty::Position;
use std::fmt::Write;
use std::mem;
use std::ptr::null_mut;
use std::sync::atomic::{AtomicI64, AtomicPtr, AtomicU32, AtomicU64, AtomicUsize, Ordering};

use crate::arena::Error as ArenaError;
use crate::evaluation::{self, Flag};
use crate::mcts::{eval_in_cp, ThreadData};
use crate::options::{
    get_cpuct, get_cpuct_root, get_cvisits_selection, get_policy_temperature,
    get_policy_temperature_root,
};
use crate::search::{to_uci, TimeManagement, SCALE};
use crate::state::State;
use crate::transposition_table::{LRAllocator, LRTable, TranspositionTable};
use crate::tree_policy;

const MAX_PLAYOUT_LENGTH: usize = 256;

const VIRTUAL_LOSS: i64 = SCALE as i64;

/// You're not intended to use this class (use an `MctsManager` instead),
/// but you can use it if you want to manage the threads yourself.
pub struct SearchTree {
    root_node: PositionNode,
    root_state: State,

    cpuct: f32,
    cpuct_root: f32,
    policy_t: f32,

    #[allow(dead_code)]
    root_table: TranspositionTable,
    ttable: LRTable,

    num_nodes: AtomicUsize,
    playouts: AtomicUsize,
    max_depth: AtomicUsize,
    tb_hits: AtomicUsize,
    next_info: AtomicU64,
}

pub struct MoveEdge {
    sum_evaluations: AtomicI64,
    visits: AtomicU32,
    policy: u16,
    mov: shakmaty::Move,
    child: AtomicPtr<PositionNode>,
}

pub struct PositionNode {
    hots: *const [MoveEdge],
    flag: Flag,
}

unsafe impl Sync for PositionNode {}

static DRAW_NODE: PositionNode = PositionNode::new(&[], Flag::TerminalDraw);
static UNEXPANDED_NODE: PositionNode = PositionNode::new(&[], Flag::Standard);

impl PositionNode {
    const fn new(hots: &[MoveEdge], flag: Flag) -> Self {
        Self { hots, flag }
    }

    pub fn flag(&self) -> Flag {
        self.flag
    }

    pub fn set_flag(&mut self, flag: Flag) {
        self.flag = flag;
    }

    pub fn is_terminal(&self) -> bool {
        self.flag.is_terminal()
    }

    pub fn is_tablebase(&self) -> bool {
        self.flag.is_tablebase()
    }

    pub fn hots(&self) -> &[MoveEdge] {
        unsafe { &*(self.hots) }
    }

    pub fn clear_children_links(&self) {
        let hots = unsafe { &*(self.hots.cast_mut()) };

        for h in hots {
            h.child.store(null_mut(), Ordering::SeqCst);
        }
    }
}

impl MoveEdge {
    fn new(policy: u16, mov: shakmaty::Move) -> Self {
        Self {
            policy,
            sum_evaluations: AtomicI64::default(),
            visits: AtomicU32::default(),
            mov,
            child: AtomicPtr::default(),
        }
    }

    pub fn get_move(&self) -> &shakmaty::Move {
        &self.mov
    }

    pub fn visits(&self) -> u32 {
        self.visits.load(Ordering::Relaxed)
    }

    pub fn sum_rewards(&self) -> i64 {
        self.sum_evaluations.load(Ordering::Relaxed)
    }

    pub fn policy(&self) -> u16 {
        self.policy
    }

    pub fn average_reward(&self) -> Option<f32> {
        match self.visits() {
            0 => None,
            x => Some(self.sum_rewards() as f32 / x as f32),
        }
    }

    pub fn down(&self) {
        self.sum_evaluations
            .fetch_sub(VIRTUAL_LOSS, Ordering::Relaxed);
        self.visits.fetch_add(1, Ordering::Relaxed);
    }

    pub fn up(&self, evaln: i64) {
        let delta = evaln + VIRTUAL_LOSS;
        self.sum_evaluations.fetch_add(delta, Ordering::Relaxed);
    }

    pub fn replace(&self, other: &MoveEdge) {
        self.visits
            .store(other.visits.load(Ordering::Relaxed), Ordering::Relaxed);
        self.sum_evaluations.store(
            other.sum_evaluations.load(Ordering::Relaxed),
            Ordering::Relaxed,
        );
    }

    pub fn child(&self) -> Option<&PositionNode> {
        let child = self.child.load(Ordering::Relaxed).cast_const();
        if child.is_null() {
            None
        } else {
            unsafe { Some(&*child) }
        }
    }

    // Returns None if the child was set, or Some(child) if it was already set.
    pub fn set_or_get_child<'a>(&'a self, new_child: &'a PositionNode) -> Option<&'a PositionNode> {
        match self.child.compare_exchange(
            null_mut(),
            (new_child as *const PositionNode).cast_mut(),
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => None,
            Err(existing) => unsafe { Some(&*existing) },
        }
    }
}

fn create_node<'a, F>(
    state: &State,
    tb_hits: &AtomicUsize,
    alloc_slice: F,
    policy_t: f32,
) -> Result<PositionNode, ArenaError>
where
    F: FnOnce(usize) -> Result<&'a mut [MoveEdge], ArenaError>,
{
    let moves = state.available_moves();

    let state_flag = evaluation::evaluate_state_flag(state, &moves);
    let move_eval = evaluation::evaluate_policy(state, &moves, policy_t);

    if state_flag.is_tablebase() {
        tb_hits.fetch_add(1, Ordering::Relaxed);
    }

    let hots = alloc_slice(move_eval.len())?;

    #[allow(clippy::cast_sign_loss)]
    for (i, x) in hots.iter_mut().enumerate() {
        *x = MoveEdge::new((move_eval[i] * SCALE) as u16, moves[i].clone());
    }

    Ok(PositionNode::new(hots, state_flag))
}

impl SearchTree {
    pub fn new(
        state: State,
        current_table: TranspositionTable,
        previous_table: TranspositionTable,
    ) -> Self {
        let tb_hits = 0.into();

        let root_table = TranspositionTable::for_root();

        let mut root_node = create_node(
            &state,
            &tb_hits,
            |sz| root_table.arena().allocator().alloc_slice(sz),
            get_policy_temperature_root(),
        )
        .expect("Unable to create root node");

        previous_table.lookup_into(&state, &mut root_node);

        Self {
            root_state: state,
            root_node,
            cpuct: get_cpuct(),
            cpuct_root: get_cpuct_root(),
            policy_t: get_policy_temperature(),
            root_table,
            ttable: LRTable::new(current_table, previous_table),
            num_nodes: 1.into(),
            playouts: 0.into(),
            max_depth: 0.into(),
            tb_hits,
            next_info: 0.into(),
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
        let mut node = &self.root_node;
        let mut path: ArrayVec<&MoveEdge, MAX_PLAYOUT_LENGTH> = ArrayVec::new();
        let mut evaln = 0;
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

            let is_root = path.is_empty();

            let cpuct = if is_root { self.cpuct_root } else { self.cpuct };

            let fpu = path
                .last()
                .map_or(0, |x| -x.sum_rewards() / i64::from(x.visits()));

            let choice = tree_policy::choose_child(node.hots(), cpuct, fpu);
            choice.down();
            path.push(choice);
            state.make_move(&choice.mov);

            if choice.visits() == 1 {
                evaln = evaluation::evaluate_state(&state);
                node = &UNEXPANDED_NODE;
                break;
            }

            if state.is_repetition()
                || state.drawn_by_fifty_move_rule()
                || state.board().is_insufficient_material()
            {
                evaln = 0;
                node = &DRAW_NODE;
                break;
            }

            let new_node = match self.descend(&state, choice, tld) {
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
        }

        evaln = match node.flag {
            Flag::TerminalWin | Flag::TablebaseWin => SCALE as i64,
            Flag::TerminalLoss | Flag::TablebaseLoss => -SCALE as i64,
            Flag::TerminalDraw | Flag::TablebaseDraw => 0,
            Flag::Standard => evaln,
        };

        Self::finish_playout(&path, evaln);

        let depth = path.len();
        let num_nodes = self.num_nodes.fetch_add(depth, Ordering::Relaxed) + depth;
        self.max_depth.fetch_max(depth, Ordering::Relaxed);
        let playouts = self.playouts.fetch_add(1, Ordering::Relaxed) + 1;

        if num_nodes >= time_management.node_limit() {
            self.print_info(&time_management);
            return false;
        }

        if playouts % 128 == 0 && time_management.is_after_end() {
            self.print_info(&time_management);
            return false;
        }

        if playouts % 65536 == 0 {
            let elapsed = time_management.elapsed().as_secs();

            let next_info = self.next_info.fetch_max(elapsed, Ordering::Relaxed);

            if next_info < elapsed {
                self.print_info(&time_management);
            }
        }

        true
    }

    fn descend<'a>(
        &'a self,
        state: &State,
        choice: &'a MoveEdge,
        tld: &mut ThreadData<'a>,
    ) -> Result<&'a PositionNode, ArenaError> {
        // If the child is already there, return it.
        if let Some(child) = choice.child() {
            return Ok(child);
        }

        // Lookup to see if we already have this position in the ttable
        if let Some(node) = self.ttable.lookup(state) {
            return Ok(choice.set_or_get_child(node).unwrap_or(node));
        };

        // Create the child
        let mut created_here = create_node(
            state,
            &self.tb_hits,
            |sz| tld.allocator.alloc_move_info(sz),
            self.policy_t,
        )?;

        self.ttable.lookup_into(state, &mut created_here);

        let created = tld.allocator.alloc_node()?;

        *created = created_here;

        if let Some(node) = choice.set_or_get_child(created) {
            return Ok(node);
        }

        let inserted = self.ttable.insert(state, created);
        let inserted_ptr = (inserted as *const PositionNode).cast_mut();
        choice.child.store(inserted_ptr, Ordering::Relaxed);
        Ok(inserted)
    }

    fn finish_playout(path: &[&MoveEdge], evaln: i64) {
        let mut evaln_value = -evaln;
        for move_info in path.iter().rev() {
            move_info.up(evaln_value);
            evaln_value = -evaln_value;
        }
    }

    pub fn root_state(&self) -> &State {
        &self.root_state
    }

    pub fn root_node(&self) -> &PositionNode {
        &self.root_node
    }

    pub fn principal_variation(&self, num_moves: usize) -> Vec<&MoveEdge> {
        let mut result = Vec::new();
        let mut crnt = &self.root_node;
        while !crnt.hots().is_empty() && result.len() < num_moves {
            let choice = select_child_after_search(crnt.hots());
            result.push(choice);
            let child = choice.child.load(Ordering::SeqCst).cast_const();
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

        let nodes = self.num_nodes();
        let depth = nodes / self.playouts();
        let sel_depth = self.max_depth();
        let pv = self.principal_variation(depth.max(1));
        let pv_string: String = pv.into_iter().fold(String::new(), |mut out, x| {
            write!(out, " {}", to_uci(x.get_move())).unwrap();
            out
        });

        let nps = if search_time_ms == 0 {
            nodes
        } else {
            nodes * 1000 / search_time_ms as usize
        };

        let info_str = format!(
            "info depth {} seldepth {} nodes {} nps {} tbhits {} score {} time {} pv{}",
            depth.max(1),
            sel_depth.max(1),
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
            .map_or(0., |x| x.average_reward().unwrap_or(-SCALE) / SCALE)
    }

    fn eval_in_cp(&self) -> String {
        eval_in_cp(self.eval())
    }
}

fn select_child_after_search(children: &[MoveEdge]) -> &MoveEdge {
    let k = get_cvisits_selection();

    let reward = |child: &MoveEdge| {
        let visits = child.visits();

        if visits == 0 {
            return -SCALE;
        }

        let sum_rewards = child.sum_rewards();

        sum_rewards as f32 / visits as f32 - (k * 2. * SCALE) / (visits as f32).sqrt()
    };

    let mut best = &children[0];
    let mut best_reward = reward(best);

    for child in children.iter().skip(1) {
        let reward = reward(child);
        if reward > best_reward {
            best = child;
            best_reward = reward;
        }
    }

    best
}

pub fn print_size_list() {
    println!(
        "info string PositionNode {} MoveEdge {}",
        mem::size_of::<PositionNode>(),
        mem::size_of::<MoveEdge>(),
    );
}

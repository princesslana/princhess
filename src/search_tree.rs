use arrayvec::ArrayVec;
use std::fmt::Write;
use std::mem;
use std::ptr::{self, null_mut};
use std::sync::atomic::{
    AtomicBool, AtomicI64, AtomicPtr, AtomicU32, AtomicU64, AtomicUsize, Ordering,
};

use crate::arena::Error as ArenaError;
use crate::chess;
use crate::evaluation::{self, Flag};
use crate::options::{MctsOptions, SearchOptions, TimeManagementOptions};
use crate::search::{eval_in_cp, ThreadData};
use crate::search::{TimeManagement, SCALE};
use crate::state::State;
use crate::transposition_table::{LRAllocator, LRTable, TranspositionTable};
use crate::tree_policy;

const MAX_PLAYOUT_LENGTH: usize = 256;

pub struct SearchTree {
    root_node: PositionNode,
    root_state: State,

    search_options: SearchOptions,

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
    mov: chess::Move,
    child: AtomicPtr<PositionNode>,
}

pub struct PositionNode {
    hots: *const [MoveEdge],
    flag: Flag,
}

pub struct Reward {
    pub average: i32,
    pub visits: u32,
}

impl Reward {
    const ZERO: Self = Self {
        average: -SCALE as i32,
        visits: 0,
    };
}

unsafe impl Sync for PositionNode {}

static WIN_NODE: PositionNode = PositionNode::new(&[], Flag::TerminalWin);
static DRAW_NODE: PositionNode = PositionNode::new(&[], Flag::TerminalDraw);
static LOSS_NODE: PositionNode = PositionNode::new(&[], Flag::TerminalLoss);

static TABLEBASE_WIN_NODE: PositionNode = PositionNode::new(&[], Flag::TablebaseWin);
static TABLEBASE_DRAW_NODE: PositionNode = PositionNode::new(&[], Flag::TablebaseDraw);
static TABLEBASE_LOSS_NODE: PositionNode = PositionNode::new(&[], Flag::TablebaseLoss);

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

    pub fn visits(&self) -> u64 {
        self.hots().iter().map(|x| u64::from(x.visits())).sum()
    }

    pub fn clear_children_links(&self) {
        for h in self.hots() {
            h.child.store(null_mut(), Ordering::Relaxed);
        }
    }

    pub fn select_child_by_rewards(&self) -> &MoveEdge {
        self.hots()
            .iter()
            .max_by_key(|x| x.reward().average)
            .unwrap()
    }
}

impl MoveEdge {
    fn new(policy: u16, mov: chess::Move) -> Self {
        Self {
            policy,
            sum_evaluations: AtomicI64::default(),
            visits: AtomicU32::default(),
            mov,
            child: AtomicPtr::default(),
        }
    }

    pub fn get_move(&self) -> &chess::Move {
        &self.mov
    }

    pub fn visits(&self) -> u32 {
        self.visits.load(Ordering::Relaxed)
    }

    pub fn policy(&self) -> u16 {
        self.policy
    }

    pub fn reward(&self) -> Reward {
        let visits = self.visits.load(Ordering::Relaxed);

        if visits == 0 {
            return Reward::ZERO;
        }

        let sum = self.sum_evaluations.load(Ordering::Relaxed);

        Reward {
            average: (sum / i64::from(visits)) as i32,
            visits,
        }
    }

    pub fn down(&self) {
        self.visits.fetch_add(1, Ordering::Relaxed);
    }

    pub fn up(&self, evaln: i64) {
        self.sum_evaluations.fetch_add(evaln, Ordering::Relaxed);
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
            ptr::from_ref(new_child).cast_mut(),
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
    alloc_slice: F,
    policy_t: f32,
) -> Result<PositionNode, ArenaError>
where
    F: FnOnce(usize) -> Result<&'a mut [MoveEdge], ArenaError>,
{
    let moves = state.available_moves();

    let state_flag = evaluation::evaluate_state_flag(state, !moves.is_empty());
    let move_eval = evaluation::policy(state, &moves, policy_t);

    let hots = alloc_slice(move_eval.len())?;

    #[allow(clippy::cast_sign_loss)]
    for (i, x) in hots.iter_mut().enumerate() {
        *x = MoveEdge::new((move_eval[i] * SCALE) as u16, moves[i]);
    }

    Ok(PositionNode::new(hots, state_flag))
}

impl SearchTree {
    pub fn new(state: State, table: LRTable, search_options: SearchOptions) -> Self {
        let root_table = TranspositionTable::for_root();

        let mut root_node = create_node(
            &state,
            |sz| root_table.arena().allocator().alloc_slice(sz),
            search_options.mcts_options.policy_temperature_root,
        )
        .expect("Unable to create root node");

        table.lookup_into_from_all(&state, &mut root_node);

        Self {
            root_state: state,
            root_node,
            search_options,
            root_table,
            ttable: table,
            num_nodes: 1.into(),
            playouts: 0.into(),
            max_depth: 0.into(),
            tb_hits: 0.into(),
            next_info: 0.into(),
        }
    }

    pub fn table(self) -> LRTable {
        self.ttable
    }

    pub fn num_nodes(&self) -> usize {
        self.num_nodes.load(Ordering::Relaxed)
    }

    pub fn playouts(&self) -> usize {
        self.playouts.load(Ordering::Relaxed)
    }

    pub fn depth(&self) -> usize {
        match self.playouts() {
            0 => 0,
            _ => self.num_nodes() / self.playouts(),
        }
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
        cpuct: f32,
        time_management: &'a TimeManagement,
        stop_signal: &'a AtomicBool,
    ) -> bool {
        let mut state = self.root_state.clone();
        let mut node = &self.root_node;
        let mut path: ArrayVec<&MoveEdge, MAX_PLAYOUT_LENGTH> = ArrayVec::new();
        let mut evaln = 0;

        loop {
            self.ttable.wait_if_flipping();

            if node.is_terminal() || node.hots().is_empty() {
                break;
            }
            if node.is_tablebase() && state.halfmove_clock() == 0 {
                break;
            }
            if path.len() >= MAX_PLAYOUT_LENGTH {
                break;
            }

            let fpu = path.last().map_or(0, |x| -x.reward().average);

            let mcts_options = MctsOptions {
                cpuct,
                ..self.search_options.mcts_options
            };

            let choice = tree_policy::choose_child(node.hots(), fpu, &mcts_options);
            choice.down();
            path.push(choice);
            state.make_move(choice.mov);

            if choice.visits() == 1 {
                let flag = evaluation::evaluate_state_flag(&state, state.is_available_move());

                node = match flag {
                    Flag::TerminalWin => &WIN_NODE,
                    Flag::TerminalLoss => &LOSS_NODE,
                    Flag::TerminalDraw => &DRAW_NODE,
                    Flag::TablebaseWin => &TABLEBASE_WIN_NODE,
                    Flag::TablebaseLoss => &TABLEBASE_LOSS_NODE,
                    Flag::TablebaseDraw => &TABLEBASE_DRAW_NODE,
                    Flag::Standard => {
                        evaln = evaluation::value(&state);
                        &UNEXPANDED_NODE
                    }
                };
                break;
            }

            if state.is_repetition()
                || state.drawn_by_fifty_move_rule()
                || state.board().is_insufficient_material()
            {
                node = &DRAW_NODE;
                break;
            }

            let new_node = match self.descend(&state, choice, tld) {
                Ok(r) => r,
                Err(ArenaError::Full) => {
                    self.ttable
                        .flip_if_full(|| self.root_node.clear_children_links());
                    return true;
                }
            };

            node = new_node;
        }

        evaln = node.flag.adjust_eval(evaln);

        Self::finish_playout(&path, evaln);

        let depth = path.len();
        let num_nodes = self.num_nodes.fetch_add(depth, Ordering::Relaxed) + depth;
        self.max_depth.fetch_max(depth, Ordering::Relaxed);
        self.playouts.fetch_add(1, Ordering::Relaxed);
        tld.playouts += 1;

        if node.is_tablebase() {
            self.tb_hits.fetch_add(1, Ordering::Relaxed);
        }

        if num_nodes >= time_management.node_limit() {
            return false;
        }

        if tld.playouts % 128 == 0 && stop_signal.load(Ordering::Relaxed) {
            return false;
        }

        let elapsed = time_management.elapsed();

        if tld.playouts % 128 == 0 {
            if let Some(hard_limit) = time_management.hard_limit() {
                if elapsed >= hard_limit {
                    return false;
                }
            }

            if let Some(soft_limit) = time_management.soft_limit() {
                let opts = &self.search_options.time_management_options;

                if elapsed >= soft_limit.mul_f32(self.soft_time_multiplier(opts)) {
                    return false;
                }
            }
        }

        if tld.playouts % 65536 == 0 {
            let elapsed = time_management.elapsed().as_secs();

            let next_info = self.next_info.fetch_max(elapsed, Ordering::Relaxed);

            if next_info < elapsed && !stop_signal.load(Ordering::Relaxed) {
                self.print_info(time_management);
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
            |sz| tld.allocator.alloc_move_info(sz),
            self.search_options.mcts_options.policy_temperature,
        )?;

        // Copy any history
        self.ttable.lookup_into(state, &mut created_here);

        // Copy node to arena memory
        let created = tld.allocator.alloc_node()?;
        *created = created_here;

        // Insert the child into the ttable
        let inserted = self.ttable.insert(state, created);
        let inserted_ptr = ptr::from_ref(inserted).cast_mut();
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

    pub fn best_move(&self) -> chess::Move {
        *self.best_edge().get_move()
    }

    pub fn best_edge(&self) -> &MoveEdge {
        sort_moves(
            self.root_node.hots(),
            self.search_options.c_visits_selection,
        )[0]
    }

    fn soft_time_multiplier(&self, opts: &TimeManagementOptions) -> f32 {
        let mut m = 1.0;

        let bm_frac = self.root_node().select_child_by_rewards().visits() as f32
            / self.root_node().visits() as f32;

        m *= (opts.visits_base - bm_frac) * opts.visits_m;

        m = m.clamp(opts.min_m, opts.max_m);

        m
    }

    pub fn print_info(&self, time_management: &TimeManagement) {
        let mut info_str = String::with_capacity(256);

        let search_time_ms = time_management.elapsed().as_millis();

        let nodes = self.num_nodes();
        let depth = self.depth();
        let sel_depth = self.max_depth();
        let nps = if search_time_ms == 0 {
            nodes
        } else {
            nodes * 1000 / search_time_ms as usize
        };

        let moves = sort_moves(
            self.root_node.hots(),
            self.search_options.c_visits_selection,
        );

        let is_chess960 = self.search_options.is_chess960;

        for (idx, edge) in moves.iter().enumerate().take(self.search_options.multi_pv) {
            info_str.clear();
            info_str.push_str("info ");
            write!(info_str, "depth {} ", depth.max(1)).unwrap();
            write!(info_str, "seldepth {} ", sel_depth.max(1)).unwrap();
            write!(info_str, "nodes {nodes} ").unwrap();
            write!(info_str, "nps {nps} ").unwrap();
            write!(info_str, "tbhits {} ", self.tb_hits()).unwrap();
            write!(info_str, "hashfull {} ", self.ttable.full()).unwrap();

            if self.search_options.show_movesleft {
                write!(info_str, "movesleft {} ", self.root_state.moves_left()).unwrap();
            }

            write!(
                info_str,
                "score {} ",
                eval_in_cp(self.best_edge().reward().average as f32 / SCALE)
            )
            .unwrap();
            write!(info_str, "time {search_time_ms} ").unwrap();
            write!(info_str, "multipv {} ", idx + 1).unwrap();

            let pv = match edge.child() {
                Some(child) => principal_variation(child, depth.max(1) - 1),
                None => vec![],
            };

            write!(info_str, "pv {}", edge.get_move().to_uci(is_chess960)).unwrap();

            for m in &pv {
                write!(info_str, " {}", m.get_move().to_uci(is_chess960)).unwrap();
            }

            println!("{info_str}");
        }
    }
}

fn principal_variation(from: &PositionNode, num_moves: usize) -> Vec<&MoveEdge> {
    let mut result = Vec::with_capacity(num_moves);
    let mut crnt = from;

    while !crnt.hots().is_empty() && result.len() < num_moves {
        let choice = crnt.select_child_by_rewards();
        result.push(choice);

        match choice.child() {
            Some(child) => crnt = child,
            None => break,
        }
    }

    result
}

pub fn sort_moves(children: &[MoveEdge], k: f32) -> Vec<&MoveEdge> {
    let reward = |child: &MoveEdge| {
        let reward = child.reward();

        if reward.visits == 0 {
            return -(2. * SCALE) + f32::from(child.policy());
        }

        reward.average as f32 - (k * 2. * SCALE) / (reward.visits as f32).sqrt()
    };

    let mut result = Vec::with_capacity(children.len());

    for child in children {
        result.push((child, reward(child)));
    }

    result.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    result.reverse();

    result.into_iter().map(|x| x.0).collect()
}

pub fn print_size_list() {
    println!(
        "info string PositionNode {} MoveEdge {}",
        mem::size_of::<PositionNode>(),
        mem::size_of::<MoveEdge>(),
    );
}

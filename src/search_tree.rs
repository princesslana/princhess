use arrayvec::ArrayVec;
use std::fmt::{Display, Write};
use std::mem;
use std::ptr::{self, null_mut};
use std::sync::atomic::{
    AtomicBool, AtomicI64, AtomicPtr, AtomicU32, AtomicU64, AtomicUsize, Ordering,
};

use crate::arena::{ArenaRef, Error as ArenaError};
use crate::chess;
use crate::evaluation::{self, Flag};
use crate::options::{MctsOptions, SearchOptions, TimeManagementOptions};
use crate::search::{eval_in_cp, ThreadData, SCALE};
use crate::state::State;
use crate::time_management::TimeManagement;
use crate::transposition_table::{AllocNodeResult, LRTable, TranspositionTable};
use crate::tree_policy;

const MAX_PLAYOUT_LENGTH: usize = 256;

const PV_EVAL_MIN_DEPTH: usize = 4;

pub struct SearchTree {
    root_node: ArenaRef<PositionNode>,
    root_state: State,

    search_options: SearchOptions,

    #[allow(dead_code)]
    root_table: TranspositionTable,

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

#[derive(Clone)]
pub struct PositionNode {
    hots: *const [MoveEdge],
    flag: Flag,
    generation: u32,
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

// These static nodes are not allocated in an Arena, so their generation is 0 (invalid).
// This is fine as they are never looked up in the TT or cleared.
static WIN_NODE: PositionNode = PositionNode::new(&[], Flag::TerminalWin, 0);
static DRAW_NODE: PositionNode = PositionNode::new(&[], Flag::TerminalDraw, 0);
static LOSS_NODE: PositionNode = PositionNode::new(&[], Flag::TerminalLoss, 0);

static TABLEBASE_WIN_NODE: PositionNode = PositionNode::new(&[], Flag::TablebaseWin, 0);
static TABLEBASE_DRAW_NODE: PositionNode = PositionNode::new(&[], Flag::TablebaseDraw, 0);
static TABLEBASE_LOSS_NODE: PositionNode = PositionNode::new(&[], Flag::TablebaseLoss, 0);

static UNEXPANDED_NODE: PositionNode = PositionNode::new(&[], Flag::Standard, 0);

impl PositionNode {
    const fn new(hots: &[MoveEdge], flag: Flag, generation: u32) -> Self {
        Self {
            hots,
            flag,
            generation,
        }
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

    pub fn select_child_by_visits(&self) -> &MoveEdge {
        self.hots().iter().max_by_key(|x| x.visits()).unwrap()
    }

    pub fn generation(&self) -> u32 {
        self.generation
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
    pub fn set_or_get_child(&self, new_child: &PositionNode) -> Option<&PositionNode> {
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

fn create_node<F>(
    state: &State,
    alloc: F,
    policy_t: f32,
) -> Result<ArenaRef<PositionNode>, ArenaError>
where
    F: FnOnce(usize) -> AllocNodeResult,
{
    let moves = state.available_moves();

    let state_flag = evaluation::evaluate_state_flag(state, !moves.is_empty());
    let move_eval = evaluation::policy(state, &moves, policy_t);

    let (node_uninit_ref, hots_uninit_ref) = alloc(move_eval.len())?;

    #[allow(clippy::cast_sign_loss)]
    let hots_arena_ref = hots_uninit_ref.init_each(|i| {
        let policy_val = (move_eval[i] * SCALE) as u16;
        MoveEdge::new(policy_val, moves[i])
    });

    // Capture the generation before `node_uninit_ref` is moved by `write`.
    let node_generation = node_uninit_ref.generation();

    // SAFETY: `node_uninit_ref` points to valid, uninitialized memory for a single PositionNode.
    // We are writing it exactly once.
    let node_arena_ref = node_uninit_ref.write(PositionNode::new(
        &hots_arena_ref,
        state_flag,
        node_generation,
    ));

    Ok(node_arena_ref)
}

impl SearchTree {
    pub fn new(state: State, table: &LRTable, search_options: SearchOptions) -> Self {
        let root_table = TranspositionTable::for_root();

        let root_allocator = |sz| {
            let allocator = root_table.arena().allocator();
            Ok((allocator.alloc_one()?, allocator.alloc_slice(sz)?))
        };

        let mut root_node_arena_ref = create_node(
            &state,
            root_allocator,
            search_options.mcts_options.policy_temperature_root,
        )
        .expect("Unable to create root node");

        table.lookup_into_from_all(&state, &mut root_node_arena_ref);

        Self {
            root_state: state,
            root_node: root_node_arena_ref,
            search_options,
            root_table,
            num_nodes: 1.into(),
            playouts: 0.into(),
            max_depth: 0.into(),
            tb_hits: 0.into(),
            next_info: 0.into(),
        }
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

    #[inline(never)]
    pub fn playout<'a>(
        &'a self,
        tld: &mut ThreadData<'a>,
        cpuct: f32,
        time_management: &TimeManagement,
        stop_signal: &AtomicBool,
    ) -> bool {
        let mut state = self.root_state.clone();
        let mut node: &'a PositionNode = &self.root_node;
        let mut path: ArrayVec<&'a MoveEdge, MAX_PLAYOUT_LENGTH> = ArrayVec::new();
        let mut evaln = 0;

        loop {
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
                    tld.ttable
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
                self.print_info(time_management, tld.ttable.full());
            }
        }

        true
    }

    fn descend<'a>(
        &self,
        state: &State,
        choice: &'a MoveEdge,
        tld: &mut ThreadData<'a>,
    ) -> Result<&'a PositionNode, ArenaError> {
        let current_arena_generation = tld.ttable.current_table().generation();

        // If the child is already there, check its generation.
        if let Some(child) = choice.child() {
            if child.generation() == current_arena_generation {
                // Child is valid and current, return it.
                return Ok(child);
            }
            // Child exists but is from an old generation (stale).
            // Clear the pointer so it can be correctly re-set later.
            choice.child.store(null_mut(), Ordering::Relaxed);
        }

        // At this point, `choice.child` is either `null_mut` (was never set, or just cleared because it was stale).

        // Lookup to see if we already have this position in the ttable.
        // The `lookup` method already ensures the node's generation matches the current table's generation.
        if let Some(node) = tld.ttable.lookup(state) {
            // If found in TT, it's guaranteed to be current generation.
            // Try to set the child pointer to this node. If it was already set
            // by another thread (which would have set it to a current node), use the existing one.
            return Ok(choice.set_or_get_child(node).unwrap_or(node));
        }

        // Create the child
        let mut created_node_arena_ref = create_node(
            state,
            |sz| tld.allocator.alloc_node(sz),
            self.search_options.mcts_options.policy_temperature,
        )?;

        // Copy any history
        tld.ttable.lookup_into(state, &mut created_node_arena_ref);

        // Insert the child into the ttable
        let inserted = tld.ttable.insert(state, created_node_arena_ref);
        let inserted_ptr = ptr::from_ref::<PositionNode>(inserted).cast_mut();
        // Unconditionally store the new node, as `choice.child` is either `null_mut` or was just nulled.
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
        self.sort_moves(self.root_node.hots())[0]
    }

    fn sort_moves<'b>(&self, children: &'b [MoveEdge]) -> Vec<&'b MoveEdge> {
        let reward = |child: &MoveEdge| {
            let reward = child.reward();

            if reward.visits == 0 {
                return -(2. * SCALE) + f32::from(child.policy());
            }

            let visits_adj = (self.search_options.c_visits_selection * 2. * SCALE)
                / (reward.visits as f32).sqrt();

            reward.average as f32 - visits_adj
        };

        let mut result = Vec::with_capacity(children.len());

        for child in children {
            result.push((child, reward(child)));
        }

        result.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        result.reverse();

        result.into_iter().map(|x| x.0).collect()
    }

    fn soft_time_multiplier(&self, opts: &TimeManagementOptions) -> f32 {
        let mut m = 1.0;

        let bm = self.root_node().select_child_by_rewards();
        let bm_reward = bm.reward();

        let bm_frac = bm_reward.visits as f32 / self.root_node().visits() as f32;

        m *= (opts.visits_base - bm_frac) * opts.visits_m;

        let pv_eval_depth = self.depth() / 2;
        if pv_eval_depth >= PV_EVAL_MIN_DEPTH {
            let bm_eval = bm_reward.average;
            let bm_pv_eval = pv_eval(self.root_state.clone(), bm, pv_eval_depth);

            if (bm_eval - bm_pv_eval).abs() > (opts.pv_diff_c * SCALE) as i32 {
                m *= opts.pv_diff_m;
            }
        }

        m = m.clamp(opts.min_m, opts.max_m);

        m
    }

    pub fn print_info(&self, time_management: &TimeManagement, hash_full: usize) {
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

        let moves = self.sort_moves(self.root_node.hots());

        let is_chess960 = self.search_options.is_chess960;

        for (idx, edge) in moves.iter().enumerate().take(self.search_options.multi_pv) {
            info_str.clear();
            info_str.push_str("info ");
            write!(info_str, "depth {} ", depth.max(1)).unwrap();
            write!(info_str, "seldepth {} ", sel_depth.max(1)).unwrap();
            write!(info_str, "nodes {nodes} ").unwrap();
            write!(info_str, "nps {nps} ").unwrap();
            write!(info_str, "tbhits {} ", self.tb_hits()).unwrap();
            write!(info_str, "hashfull {hash_full} ").unwrap();

            if self.search_options.show_movesleft {
                write!(info_str, "movesleft {} ", self.root_state.moves_left()).unwrap();
            }

            if self.search_options.show_wdl {
                let wdl = UciWdl::from_eval(
                    edge.reward().average as f32 / SCALE,
                    self.root_state.phase(),
                );
                write!(info_str, "wdl {wdl} ").unwrap();
            }

            write!(
                info_str,
                "score {} ",
                eval_in_cp(edge.reward().average as f32 / SCALE)
            )
            .unwrap();
            write!(info_str, "time {search_time_ms} ").unwrap();
            write!(info_str, "multipv {} ", idx + 1).unwrap();

            let pv = match edge.child() {
                Some(child) => {
                    let mut state = self.root_state.clone();
                    state.make_move(*edge.get_move());
                    principal_variation(state, child, depth.max(1) - 1)
                }
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

fn principal_variation<'a>(
    mut state: State,
    from: &'a PositionNode,
    num_moves: usize,
) -> Vec<&'a MoveEdge> {
    let mut result: Vec<&'a MoveEdge> = Vec::with_capacity(num_moves);
    let mut crnt = from;

    while !crnt.hots().is_empty() && result.len() < num_moves {
        let choice = crnt.select_child_by_rewards();

        if result.iter().any(|x| x.get_move() == choice.get_move()) {
            break;
        }

        result.push(choice);

        state.make_move(*choice.get_move());
        if state.is_repetition()
            || state.drawn_by_fifty_move_rule()
            || state.board().is_insufficient_material()
        {
            break;
        }

        match choice.child() {
            Some(child) => crnt = child,
            None => break,
        }
    }

    result
}

fn pv_eval(mut state: State, mv: &MoveEdge, pv_depth: usize) -> i32 {
    match mv.child() {
        Some(child) => {
            state.make_move(*mv.get_move());
            let pv = principal_variation(state, child, pv_depth);
            let eval = pv
                .last()
                .map_or(mv.reward().average, |x| x.reward().average);

            eval * [1, -1][pv.len() % 2]
        }
        None => -SCALE as i32,
    }
}

pub fn print_size_list() {
    println!(
        "info string PositionNode {} MoveEdge {}",
        mem::size_of::<PositionNode>(),
        mem::size_of::<MoveEdge>(),
    );
}

#[derive(Debug, PartialEq, Eq)]
struct UciWdl {
    white: u16,
    draw: u16,
    black: u16,
}

impl Display for UciWdl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {} {}", self.white, self.draw, self.black)
    }
}

impl UciWdl {
    // eval here is white relative [-1.0, 1.0]
    #[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
    pub fn from_eval(eval: f32, phase: usize) -> Self {
        let phase = phase as i16;

        let mut win = ((1000. * eval.abs()) as u16).clamp(0, 1000);
        let mut draw = (-33 * phase + 1000).clamp(0, 1000) as u16;
        let mut loss = 0;

        if win + draw > 1000 {
            draw = 1000 - win;
        } else {
            let adj = (1000 - win - draw) / 3;
            win += adj;
            loss = adj;
            draw = 1000 - win - loss;
        }

        let result = Self {
            white: win,
            draw,
            black: loss,
        };

        if eval.is_sign_positive() {
            result
        } else {
            result.flip()
        }
    }

    fn flip(&self) -> Self {
        Self {
            white: self.black,
            draw: self.draw,
            black: self.white,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uciwdl_from_eval() {
        let wdl = UciWdl::from_eval(0.5, 12);
        assert_eq!(wdl.white, 500);
        assert_eq!(wdl.draw, 500);
        assert_eq!(wdl.black, 0);

        let wdl = UciWdl::from_eval(-0.5, 12);
        assert_eq!(wdl.white, 0);
        assert_eq!(wdl.draw, 500);
        assert_eq!(wdl.black, 500);

        let wdl = UciWdl::from_eval(0.0, 12);
        assert_eq!(wdl.white, 132);
        assert_eq!(wdl.draw, 736);
        assert_eq!(wdl.black, 132);
    }

    #[test]
    fn test_uciwdl_from_eval_symmetry() {
        let test_cases = [(0.3, 7), (0.6, 12), (0.9, 3), (0.1, 22)];

        for &(eval, phase) in &test_cases {
            let white = UciWdl::from_eval(eval, phase);
            let black = UciWdl::from_eval(-eval, phase);

            assert_eq!(white, black.flip());
        }
    }
}

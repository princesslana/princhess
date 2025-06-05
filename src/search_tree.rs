use arrayvec::ArrayVec;
use std::fmt::{Display, Write};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

use crate::arena::{ArenaRef, Error as ArenaError};
use crate::chess;
use crate::evaluation;
use crate::graph::{
    create_node, MoveEdge, PositionNode, DRAW_NODE, LOSS_NODE, TABLEBASE_DRAW_NODE,
    TABLEBASE_LOSS_NODE, TABLEBASE_WIN_NODE, UNEXPANDED_NODE, WIN_NODE,
};
use crate::options::{MctsOptions, SearchOptions, TimeManagementOptions};
use crate::search::{eval_in_cp, ThreadData, SCALE};
use crate::state::State;
use crate::time_management::TimeManagement;
use crate::transposition_table::{LRTable, TranspositionTable};
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
            if node.is_terminal() || node.edges().is_empty() {
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

            let choice = tree_policy::choose_child(node.edges(), fpu, &mcts_options);
            choice.down();
            path.push(choice);
            state.make_move(*choice.get_move());

            if choice.visits() == 1 {
                let flag = evaluation::evaluate_state_flag(&state, state.is_available_move());

                node = match flag {
                    evaluation::Flag::TerminalWin => &*WIN_NODE,
                    evaluation::Flag::TerminalLoss => &*LOSS_NODE,
                    evaluation::Flag::TerminalDraw => &*DRAW_NODE,
                    evaluation::Flag::TablebaseWin => &*TABLEBASE_WIN_NODE,
                    evaluation::Flag::TablebaseLoss => &*TABLEBASE_LOSS_NODE,
                    evaluation::Flag::TablebaseDraw => &*TABLEBASE_DRAW_NODE,
                    evaluation::Flag::Standard => {
                        evaln = evaluation::value(&state);
                        &*UNEXPANDED_NODE
                    }
                };
                break;
            }

            if state.is_repetition()
                || state.drawn_by_fifty_move_rule()
                || state.board().is_insufficient_material()
            {
                node = &*DRAW_NODE;
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

        evaln = node.flag().adjust_eval(evaln);

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
        }

        // Lookup to see if we already have this position in the ttable.
        // The `lookup` method already ensures the node's generation matches the current table's generation.
        if let Some(node) = tld.ttable.lookup(state) {
            // If found in TT, it's guaranteed to be current generation.
            // Set the child pointer to this node.
            choice.set_child_ptr(node);
            return Ok(node);
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
        // Unconditionally store the new node
        choice.set_child_ptr(inserted);
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
        self.sort_moves(self.root_node.edges())
            .into_iter()
            .next()
            .expect("Root node must have moves to determine best edge")
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

        let bm = self
            .root_node()
            .select_child_by_rewards()
            .expect("Root node must have moves during active search for time management");
        let bm_reward = bm.reward();

        let bm_frac = bm_reward.visits as f32 / self.root_node().visits() as f32;

        m *= (opts.visits_base - bm_frac) * opts.visits_m;

        let pv_eval_depth = self.depth() / 2;
        if pv_eval_depth >= PV_EVAL_MIN_DEPTH {
            let bm_eval = bm_reward.average;
            let bm_pv_eval = pv_eval(self.root_state.clone(), bm, pv_eval_depth);

            if (bm_eval - bm_pv_eval).abs() > (opts.pv_diff_c * SCALE) as i64 {
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

        let moves = self.sort_moves(self.root_node.edges());

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

pub fn principal_variation<'a>(
    mut state: State,
    from: &'a PositionNode,
    num_moves: usize,
) -> Vec<&'a MoveEdge> {
    let mut result: Vec<&'a MoveEdge> = Vec::with_capacity(num_moves);
    let mut crnt = from;

    while !crnt.edges().is_empty() && result.len() < num_moves {
        let choice_option = crnt.select_child_by_rewards();

        // Unwrap the option here, as the loop condition implies it won't be None
        let choice = choice_option.expect("Expected a child move, but node had no edges.");

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

pub fn pv_eval(mut state: State, mv: &MoveEdge, pv_depth: usize) -> i64 {
    match mv.child() {
        Some(child) => {
            state.make_move(*mv.get_move());
            let pv = principal_variation(state, child, pv_depth);
            let eval = pv
                .last()
                .map_or(mv.reward().average, |x| x.reward().average);

            eval * [1, -1][pv.len() % 2]
        }
        None => -SCALE as i64,
    }
}

use std::f32;
use std::fmt::{self, Display, Formatter, Write};
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, AtomicUsize, Ordering};

use arrayvec::ArrayVec;
use fastapprox::faster;

use crate::arena;
use crate::chess;
use crate::engine::{self, RootEdge, ThreadData, MATE_SCORE, SCALE};
use crate::evaluation::{self, Flag};
use crate::graph::{
    self, MoveEdge, PositionNode, DRAW_NODE, LOSS_NODE, PROVEN_MATE, TABLEBASE_DRAW_NODE,
    TABLEBASE_LOSS_NODE, TABLEBASE_WIN_NODE, UNEXPANDED_NODE, WIN_NODE,
};
use crate::math;
use crate::options::{EngineOptions, MctsOptions, TimeManagementOptions};
use crate::state::State;
use crate::time_management::TimeManagement;
use crate::transposition_table::LRTable;

pub const MAX_PLAYOUT_LENGTH: usize = 256;

const PV_EVAL_MIN_DEPTH: usize = 4;

/// Calculate exploration bonus U(s,a) for PUCT
pub fn exploration_bonus(explore_coef: i64, policy: u16, visits: u32) -> i64 {
    explore_coef * i64::from(policy) / ((i64::from(visits) + 1) * SCALE as i64)
}

/// Monte Carlo Tree Search implementation using PUCT algorithm.
/// Despite the name "tree search", this forms a graph due to transposition handling
/// where the same position can be reached via different move sequences.
pub struct Mcts {
    root_edges: Box<[MoveEdge]>,
    root_state: State,
    searchable_moves: ArrayVec<bool, 256>,
    tablebase_score: Option<i64>,

    engine_options: EngineOptions,

    num_nodes: AtomicUsize,
    playouts: AtomicUsize,
    max_depth: AtomicUsize,
    tb_hits: AtomicUsize,
    next_info: AtomicU64,

    winning_trend: AtomicBool,
    last_root_reward: AtomicI64,
}

impl Mcts {
    /// Creates a new MCTS instance with the given root state.
    pub fn new(state: State, table: &LRTable, engine_options: EngineOptions) -> Self {
        let moves = state.available_moves();
        let mut root_edges = Vec::with_capacity(moves.len());
        for mv in &moves {
            root_edges.push(MoveEdge::new(0, *mv));
        }

        // Warm-start: copy statistics from transposition table if available
        if let Some(cached) = table.lookup_from_all(&state) {
            graph::copy_edge_stats(&root_edges, cached.edges());
        }

        Self {
            root_state: state,
            root_edges: root_edges.into_boxed_slice(),
            searchable_moves: moves.iter().map(|_| true).collect(),
            tablebase_score: None,
            engine_options,
            num_nodes: 1.into(),
            playouts: 0.into(),
            max_depth: 0.into(),
            tb_hits: 0.into(),
            next_info: 0.into(),
            winning_trend: false.into(),
            last_root_reward: 0.into(),
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
    #[allow(clippy::too_many_lines)]
    /// Single MCTS playout: selection → expansion → evaluation → backpropagation
    pub fn playout<'a>(
        &'a self,
        tld: &mut ThreadData<'a>,
        options: &MctsOptions,
        time_management: &TimeManagement,
        stop_signal: &AtomicBool,
    ) -> bool {
        let total_visits: u64 = tld.root_edges.iter().map(|re| u64::from(re.visits())).sum();

        if tld.playouts.is_multiple_of(1024) {
            tld.root_gini = math::gini(tld.root_edges.iter().map(RootEdge::visits), total_visits);
        }

        let root_edge_idx = tld
            .top_two_state
            .next(&tld.root_edges, &self.searchable_moves);
        let root_edge_ref = &self.root_edges[root_edge_idx];

        let mut state = self.root_state.clone();
        state.make_move(*root_edge_ref.get_move());

        let mut node: &'a PositionNode;
        let mut evaln = 0;

        if tld.root_edges[root_edge_idx].visits() == 0 {
            let Some((expanded_node, eval)) = self.expand_edge(&state) else {
                return true;
            };
            node = expanded_node;
            evaln = eval;
        } else if state.is_repetition()
            || state.drawn_by_fifty_move_rule()
            || state.board().is_insufficient_material()
        {
            node = &*DRAW_NODE;
        } else {
            node = match self.descend(&state, root_edge_ref, tld) {
                Ok(r) => r,
                Err(arena::Error::Full) => {
                    tld.ttable.flip_if_full(|| self.clear_root_children_links());
                    return true;
                }
            };
        }

        let mut path: ArrayVec<(&'a MoveEdge, i64), MAX_PLAYOUT_LENGTH> = ArrayVec::new();

        let search_proven = node.proof() != 0;

        loop {
            if node.is_stale(tld.ttable.current_generation()) {
                return true;
            }

            if node.edges().is_empty() || (node.proof() != 0 && !search_proven) {
                break;
            }

            if !node.flag().is_standard() {
                if !node.flag().is_valid() {
                    return true;
                }
                if node.is_terminal() {
                    break;
                }
                if node.is_tablebase()
                    && state.halfmove_clock() == 0
                    && self.tablebase_score.is_none()
                {
                    break;
                }
            }

            if path.len() >= MAX_PLAYOUT_LENGTH {
                evaln = evaluation::value(&state, self.engine_options.evaluation_options);
                break;
            }

            let parent_q = path.last().map_or(0, |(x, _)| -x.reward().average);

            let choice = self.select(node, parent_q, tld, options);
            choice.down(parent_q);
            path.push((choice, parent_q));
            state.make_move(*choice.get_move());

            if choice.visits() == 1 {
                let Some((expanded_node, eval)) = self.expand_edge(&state) else {
                    return true;
                };
                node = expanded_node;
                evaln = eval;
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
                Err(arena::Error::Full) => {
                    tld.ttable.flip_if_full(|| self.clear_root_children_links());
                    return true;
                }
            };

            node = new_node;
        }

        evaln = match node.proof() {
            0 => node.flag().adjust_eval(evaln),
            proof => i64::from(proof.signum()) * MATE_SCORE,
        };

        Self::finish_playout(&mut tld.root_edges[root_edge_idx], &path, evaln);

        if node.proof() != 0 {
            Self::propagate_proof(root_edge_ref, &path, node.proof());
        }

        let depth = path.len() + 1;
        tld.num_nodes += depth;
        tld.max_depth = tld.max_depth.max(depth);
        tld.playouts += 1;

        if node.is_tablebase() {
            tld.tb_hits += 1;
        }

        if tld.is_main_thread()
            && self.num_nodes.load(Ordering::Relaxed) + tld.num_nodes
                >= time_management.node_limit()
        {
            return false;
        }

        if tld.playouts.is_multiple_of(1024) {
            self.flush_root_edges(tld);
            self.flush_thread_stats(tld);

            if tld.is_main_thread() {
                let elapsed = time_management.elapsed().as_secs();
                let next_info = self.next_info.fetch_max(elapsed, Ordering::Relaxed);

                if next_info < elapsed && !stop_signal.load(Ordering::Relaxed) {
                    self.print_info(time_management, tld.ttable.full());
                }
            }
        }

        if tld.playouts.is_multiple_of(128) && stop_signal.load(Ordering::Relaxed) {
            return false;
        }

        let elapsed = time_management.elapsed();

        if tld.playouts.is_multiple_of(128) {
            if let Some(hard_limit) = time_management.hard_limit() {
                if elapsed >= hard_limit {
                    return false;
                }
            }

            if let Some(soft_limit) = time_management.soft_limit() {
                let opts = &self.engine_options.time_management_options;

                if elapsed
                    >= soft_limit
                        .mul_f32(self.soft_time_multiplier(opts, tld.top_two_state.last_tc_sq()))
                {
                    return false;
                }
            }
        }

        if tld.is_main_thread() && tld.playouts.is_multiple_of(128) {
            let current_reward = self.best_edge().reward().average;
            let last_reward = self.last_root_reward.load(Ordering::Relaxed);
            self.last_root_reward
                .store(current_reward, Ordering::Relaxed);
            self.winning_trend
                .store(current_reward > last_reward, Ordering::Relaxed);
        }

        true
    }

    fn descend<'a>(
        &self,
        state: &State,
        choice: &'a MoveEdge,
        tld: &mut ThreadData<'a>,
    ) -> Result<&'a PositionNode, arena::Error> {
        let current_arena_generation = tld.ttable.current_generation();

        // If the child is already there, check its generation.
        if let Some(child) = choice.child() {
            if !child.is_stale(current_arena_generation) {
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
        let mut created_node_arena_ref = graph::create_node(
            state,
            |sz| tld.allocator.alloc_node(sz),
            self.engine_options.mcts_options.policy_temperature,
        )?;

        // Copy any history
        tld.ttable.lookup_into(state, &mut created_node_arena_ref);

        // Insert the child into the ttable
        let inserted = tld.ttable.insert(state, created_node_arena_ref);
        // Unconditionally store the new node
        choice.set_child_ptr(inserted);
        Ok(inserted)
    }

    fn finish_playout(root_edge: &mut RootEdge, path: &[(&MoveEdge, i64)], evaln: i64) {
        let mut evaln_value = -evaln;
        for (move_info, virtual_loss) in path.iter().rev() {
            move_info.up(evaln_value, *virtual_loss);
            evaln_value = -evaln_value;
        }
        root_edge.up(evaln_value);
    }

    fn propagate_proof(root_edge: &MoveEdge, path: &[(&MoveEdge, i64)], mut proof: i16) {
        for i in (0..path.len()).rev() {
            let parent_edge = if i == 0 { root_edge } else { path[i - 1].0 };

            let Some(parent) = parent_edge.child() else {
                return;
            };

            if !parent.update_proof(proof) {
                return;
            }

            proof = parent.proof();

            // Root edge stats are buffered per thread, so only edges within the tree are set
            if i > 0 {
                parent_edge.set_proven_value(-i64::from(proof.signum()) * MATE_SCORE);
            }
        }
    }

    pub fn root_state(&self) -> &State {
        &self.root_state
    }

    pub fn root_edges(&self) -> &[MoveEdge] {
        &self.root_edges
    }

    pub fn root_edges_mut(&mut self) -> &mut [MoveEdge] {
        &mut self.root_edges
    }

    pub fn set_root_filter(
        &mut self,
        searchable_moves: ArrayVec<bool, 256>,
        tablebase_score: Option<i64>,
    ) {
        for (edge, &searchable) in self.root_edges.iter().zip(&searchable_moves) {
            if !searchable {
                edge.clear_stats();
            }
        }

        self.searchable_moves = searchable_moves;
        self.tablebase_score = tablebase_score;
    }

    pub fn root_visits(&self) -> u64 {
        self.root_edges.iter().map(|x| u64::from(x.visits())).sum()
    }

    pub fn clear_root_children_links(&self) {
        graph::clear_edge_children(&self.root_edges);
    }

    pub fn flush_root_edges(&self, tld: &mut ThreadData) {
        for (root_edge, edge) in tld.root_edges.iter_mut().zip(&self.root_edges) {
            root_edge.flush(edge);
        }
    }

    pub fn flush_thread_stats(&self, tld: &mut ThreadData) {
        self.num_nodes.fetch_add(tld.num_nodes, Ordering::Relaxed);
        self.max_depth.fetch_max(tld.max_depth, Ordering::Relaxed);
        self.playouts.fetch_add(tld.playouts, Ordering::Relaxed);
        self.tb_hits.fetch_add(tld.tb_hits, Ordering::Relaxed);

        tld.num_nodes = 0;
        tld.max_depth = 0;
        tld.playouts = 0;
        tld.tb_hits = 0;
    }

    pub fn best_move(&self) -> chess::Move {
        *self.best_edge().get_move()
    }

    /// Returns the best move edge from the root position.
    ///
    /// # Panics
    ///
    /// Panics if the root node has no moves (e.g., checkmate or stalemate positions).
    pub fn best_edge(&self) -> &MoveEdge {
        self.sort_edges_by_score(&self.root_edges)
            .into_iter()
            .next()
            .expect("Root node must have moves to determine best edge")
    }

    fn move_score(&self, edge: &MoveEdge) -> f32 {
        let reward = edge.reward();

        if reward.visits == 0 {
            return -(2. * SCALE) + f32::from(edge.policy());
        }

        let visits_adj =
            (self.engine_options.c_visits_selection * 2. * SCALE) / (reward.visits as f32).sqrt();

        reward.average as f32 - visits_adj
    }

    fn sort_edges_by_score<'b>(&self, edges: &'b [MoveEdge]) -> Vec<&'b MoveEdge> {
        let mut result: Vec<&MoveEdge> = edges.iter().collect();
        result.sort_by(|a, b| {
            let (proof_a, proof_b) = (edge_proof(a), edge_proof(b));

            proof_b.signum().cmp(&proof_a.signum()).then_with(|| {
                if proof_a == 0 && proof_b == 0 {
                    self.move_score(b).total_cmp(&self.move_score(a))
                } else {
                    proof_b.cmp(&proof_a)
                }
            })
        });
        result
    }

    fn soft_time_multiplier(&self, opts: &TimeManagementOptions, last_tc_sq: f32) -> f32 {
        if self.searchable_moves.iter().filter(|&&s| s).count() == 1 {
            return 0.0;
        }

        if self.root_visits() == 0 {
            return 1.0;
        }

        let mut m = 1.0;

        let bm = graph::select_edge_by_rewards(&self.root_edges)
            .expect("Root node must have moves during active search for time management");
        let bm_reward = bm.reward();

        let bm_frac = bm_reward.visits as f32 / self.root_visits() as f32;

        m *= (opts.visits_base - bm_frac) * opts.visits_m;

        let pv_eval_depth = self.depth() / 2;
        if pv_eval_depth >= PV_EVAL_MIN_DEPTH {
            let bm_eval = bm_reward.average;
            let bm_pv_eval = pv_eval(self.root_state.clone(), bm, pv_eval_depth);

            let diff_abs_normalized = (bm_eval - bm_pv_eval).abs() as f32 / SCALE;

            // `opts.pv_diff_c` is the threshold for the normalized PV difference.
            // `opts.pv_diff_m` is the scaling factor for the time multiplier adjustment.
            // The adjustment can be positive (increase time) or negative (decrease time)
            // depending on whether `diff_abs_normalized` is above or below `opts.pv_diff_c`.
            let adjustment = (diff_abs_normalized - opts.pv_diff_c) * opts.pv_diff_m;

            m *= 1.0 + adjustment;
        }

        let p_challenger = 1.0 / (2.0 + last_tc_sq.max(0.0).sqrt());
        let top_two_adjustment = (p_challenger - opts.top_two_c) * opts.top_two_m;
        m *= 1.0 + top_two_adjustment.max(0.0);

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

        let moves = self.sort_edges_by_score(&self.root_edges);

        let is_chess960 = self.engine_options.is_chess960;

        for (idx, edge) in moves.iter().enumerate().take(self.engine_options.multi_pv) {
            info_str.clear();
            info_str.push_str("info ");
            write!(info_str, "depth {} ", depth.max(1)).unwrap();
            write!(info_str, "seldepth {} ", sel_depth.max(1)).unwrap();
            write!(info_str, "nodes {nodes} ").unwrap();
            write!(info_str, "nps {nps} ").unwrap();
            write!(info_str, "tbhits {} ", self.tb_hits()).unwrap();
            write!(info_str, "hashfull {hash_full} ").unwrap();

            if self.engine_options.show_movesleft {
                write!(info_str, "movesleft {} ", self.root_state.moves_left()).unwrap();
            }

            let proof = edge_proof(edge);
            let average = edge.reward().average;
            let eval = if proof == 0 {
                self.tablebase_score
                    .map_or(average, |tb| average.midpoint(tb)) as f32
                    / SCALE
            } else {
                f32::from(proof.signum())
            };

            if self.engine_options.show_wdl {
                let wdl = UciWdl::from_eval(eval, self.root_state.phase());
                write!(info_str, "wdl {wdl} ").unwrap();
            }

            write!(info_str, "score {} ", format_score(proof, eval)).unwrap();
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

    /// Expands an edge that's being visited for the first time.
    /// Returns the node and evaluation, or None if the flag is invalid.
    fn expand_edge<'a>(&self, state: &State) -> Option<(&'a PositionNode, i64)> {
        let flag = evaluation::evaluate_state_flag(state, state.is_available_move());

        let (node, eval) = match flag {
            Flag::TERMINAL_WIN => (&*WIN_NODE, 0),
            Flag::TERMINAL_LOSS => (&*LOSS_NODE, 0),
            Flag::TERMINAL_DRAW => (&*DRAW_NODE, 0),
            Flag::TABLEBASE_WIN => (&*TABLEBASE_WIN_NODE, 0),
            Flag::TABLEBASE_LOSS => (&*TABLEBASE_LOSS_NODE, 0),
            Flag::TABLEBASE_DRAW => (&*TABLEBASE_DRAW_NODE, 0),
            Flag::STANDARD => {
                let eval = evaluation::value(state, self.engine_options.evaluation_options);
                (&*UNEXPANDED_NODE, eval)
            }
            _ => return None,
        };

        Some((node, eval))
    }

    /// Calculate exploration coefficient using GPUCT formula with trend adjustment and Gini scaling
    pub fn exploration_coefficient(
        &self,
        options: &MctsOptions,
        total_visits: u64,
        apply_trend_adjustment: bool,
        gini: f32,
    ) -> i64 {
        // Apply trend adjustment (main thread only)
        let mut cpuct = if apply_trend_adjustment {
            let winning_trend = self.winning_trend.load(Ordering::Relaxed);
            let trend_factor = [
                -options.cpuct_trend_adjustment,
                options.cpuct_trend_adjustment,
            ][usize::from(winning_trend)];
            options.cpuct * (1.0 + trend_factor)
        } else {
            options.cpuct
        };

        // Apply Gini impurity scaling
        let gini_scale = (options.cpuct_gini_base
            - options.cpuct_gini_factor * faster::ln(gini + 0.001))
        .clamp(0.0, options.cpuct_gini_max);
        cpuct *= gini_scale;

        // Calculate exploration coefficient
        (cpuct * faster::exp(options.cpuct_tau * faster::ln((total_visits + 1) as f32)) * SCALE)
            as i64
    }

    /// PUCT selection: choose best child using Q(action) + U(action)
    /// where U incorporates policy priors and visit counts
    fn select<'a>(
        &self,
        node: &'a PositionNode,
        fpu: i64,
        tld: &ThreadData,
        options: &MctsOptions,
    ) -> &'a MoveEdge {
        let moves = node.edges();
        let total_visits = node.visits();
        let gini = f32::from(node.gini()) / SCALE;
        let explore_coef =
            self.exploration_coefficient(options, total_visits, tld.is_main_thread(), gini);

        let mut best_move = &moves[0];
        let mut best_score = i64::MIN;

        for mov in moves {
            let reward = mov.reward();
            let q = if reward.visits > 0 {
                reward.average
            } else {
                fpu
            };
            let u = exploration_bonus(explore_coef, mov.policy(), reward.visits);
            let score = q + u;

            if score > best_score {
                best_score = score;
                best_move = mov;
            }
        }

        best_move
    }
}

#[derive(Debug, PartialEq, Eq)]
struct UciWdl {
    white: u16,
    draw: u16,
    black: u16,
}

impl Display for UciWdl {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{} {} {}", self.white, self.draw, self.black)
    }
}

impl UciWdl {
    // eval here is white relative [-1.0, 1.0]
    // a=0.5: eval where win rate hits 50% (self-consistent with MCTS win=+1/draw=0/loss=-1)
    // b=phase/48: spread; at phase=24 (startpos) gives ~46% draw at eval=0
    #[allow(clippy::cast_sign_loss)]
    pub fn from_eval(eval: f32, phase: usize) -> Self {
        let b = phase as f32 / 48.0 + 1e-6;
        let win = (1000.0 / (1.0 + ((0.5 - eval) / b).exp())).round() as u16;
        let loss = (1000.0 / (1.0 + ((0.5 + eval) / b).exp())).round() as u16;
        let draw = 1000_u16.saturating_sub(win).saturating_sub(loss);

        Self {
            white: win,
            draw,
            black: loss,
        }
    }
}

#[must_use]
pub fn edge_proof(edge: &MoveEdge) -> i16 {
    edge.child()
        .map_or(0, |child| graph::parent_proof(child.proof()))
}

#[must_use]
pub fn format_score(proof: i16, eval: f32) -> String {
    if proof == 0 {
        engine::eval_in_cp(eval)
    } else {
        let moves = (PROVEN_MATE - proof.abs() + 1) / 2;
        format!("mate {}", proof.signum() * moves)
    }
}

/// Extracts the principal variation (best line of play) from the search tree.
///
/// # Panics
///
/// Panics if there's a logic error where a node has edges but selecting by rewards returns None.
#[must_use]
pub fn principal_variation<'a>(
    mut state: State,
    from: &'a PositionNode,
    num_moves: usize,
) -> Vec<&'a MoveEdge> {
    let mut result: Vec<&'a MoveEdge> = Vec::with_capacity(num_moves);
    let mut crnt = from;

    if state.is_repetition()
        || state.drawn_by_fifty_move_rule()
        || state.board().is_insufficient_material()
    {
        return result;
    }

    while !crnt.edges().is_empty() && result.len() < num_moves {
        let choice = graph::select_edge_by_rewards(crnt.edges())
            .expect("Expected a child move, but node had no edges.");

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uciwdl_from_eval() {
        let wdl = UciWdl::from_eval(0.5, 12);
        assert_eq!(wdl.white, 500);
        assert_eq!(wdl.draw, 482);
        assert_eq!(wdl.black, 18);

        let wdl = UciWdl::from_eval(-0.5, 12);
        assert_eq!(wdl.white, 18);
        assert_eq!(wdl.draw, 482);
        assert_eq!(wdl.black, 500);

        let wdl = UciWdl::from_eval(0.0, 12);
        assert_eq!(wdl.white, 119);
        assert_eq!(wdl.draw, 762);
        assert_eq!(wdl.black, 119);
    }

    #[test]
    fn test_uciwdl_from_eval_symmetry() {
        let test_cases = [(0.3, 7), (0.6, 12), (0.9, 3), (0.1, 22)];

        for &(eval, phase) in &test_cases {
            let white = UciWdl::from_eval(eval, phase);
            let black = UciWdl::from_eval(-eval, phase);

            assert_eq!(white.white, black.black);
            assert_eq!(white.black, black.white);
            assert_eq!(white.draw, black.draw);
        }
    }
}

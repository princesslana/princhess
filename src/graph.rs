use std::mem;
use std::ptr::{self, NonNull};
use std::sync::atomic::{AtomicI64, AtomicPtr, AtomicU32, Ordering};
use std::sync::LazyLock;

use crate::arena::{ArenaRef, Error as ArenaError};
use crate::chess;
use crate::engine::SCALE;
use crate::evaluation::{self, Flag};
use crate::state::State;
use crate::transposition_table::AllocNodeResult;

#[repr(C)]
/// Edge in the MCTS graph storing move, policy prior, and accumulated statistics.
/// Unlike standard MCTS, visit counts and rewards are stored on edges rather than nodes.
pub struct MoveEdge {
    sum_evaluations: AtomicI64,
    visits: AtomicU32,
    policy: u16,
    mov: chess::Move,
    child: AtomicPtr<PositionNode>,
}

#[repr(C)]
#[derive(Clone)]
pub struct PositionNode {
    edges_ptr: NonNull<MoveEdge>,
    edges_count: u8,
    flag: Flag,
    gini: u16,
    generation: u32,
    _padding: [u8; 8],
}

pub struct Reward {
    pub average: i64,
    pub visits: u32, // This is fine as visits count up, not related to evaluation magnitude
}

impl Reward {
    pub const ZERO: Self = Self {
        average: -SCALE as i64,
        visits: 0,
    };
}

// SAFETY: Manual Send/Sync required due to the cycle (PositionNode → MoveEdge → PositionNode).
// This is safe because:
// - edges_ptr always points to arena or static allocations that outlive cross-thread uses
// - all mutation of edge data goes through atomics (visits, sum_evaluations, child pointer)
// - edges_count == 0 permits edges_ptr to be dangling (never dereferenced)
unsafe impl Send for PositionNode {}
unsafe impl Sync for PositionNode {}

// Macro to define static PositionNode instances
macro_rules! define_static_node {
    ($name:ident, $flag:expr) => {
        pub static $name: LazyLock<PositionNode> =
            LazyLock::new(|| PositionNode::new_static($flag));
    };
}

// These static nodes, created via `PositionNode::new_static`, are not allocated in an Arena,
// so their generation is 0 (invalid). This is fine as they are never looked up in the TT or cleared.
define_static_node!(WIN_NODE, Flag::TERMINAL_WIN);
define_static_node!(DRAW_NODE, Flag::TERMINAL_DRAW);
define_static_node!(LOSS_NODE, Flag::TERMINAL_LOSS);

define_static_node!(TABLEBASE_WIN_NODE, Flag::TABLEBASE_WIN);
define_static_node!(TABLEBASE_DRAW_NODE, Flag::TABLEBASE_DRAW);
define_static_node!(TABLEBASE_LOSS_NODE, Flag::TABLEBASE_LOSS);

define_static_node!(UNEXPANDED_NODE, Flag::STANDARD);

impl PositionNode {
    fn new(edges: NonNull<[MoveEdge]>, flag: Flag, gini: u16, generation: u32) -> Self {
        // SAFETY: Chess positions have < 255 legal moves, so this cast is always safe
        #[allow(clippy::cast_possible_truncation)]
        let edges_count = edges.len() as u8;

        Self {
            edges_ptr: edges.cast::<MoveEdge>(),
            edges_count,
            flag,
            gini,
            generation,
            _padding: [0; 8],
        }
    }

    pub fn new_static(flag: Flag) -> Self {
        Self {
            edges_ptr: NonNull::dangling(),
            edges_count: 0,
            flag,
            gini: 0,
            generation: 0,
            _padding: [0; 8],
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

    pub fn edges(&self) -> &[MoveEdge] {
        // SAFETY: The NonNull guarantees the pointer is valid and non-null.
        // The count is guaranteed to be <= the actual allocation size.
        // For empty edges (count = 0), we use a dangling pointer which is safe.
        unsafe { std::slice::from_raw_parts(self.edges_ptr.as_ptr(), self.edges_count as usize) }
    }

    pub fn visits(&self) -> u64 {
        self.edges().iter().map(|x| u64::from(x.visits())).sum()
    }

    pub fn generation(&self) -> u32 {
        self.generation
    }

    /// Checks if this `PositionNode` is stale relative to the given current arena generation.
    ///
    /// A node is considered stale if its generation does not match the current active
    /// arena's generation. This typically means the arena it was allocated in has
    /// been cleared or is no longer the active one for new allocations.
    #[must_use]
    pub fn is_stale(&self, current_arena_generation: u32) -> bool {
        self.generation != 0 && self.generation != current_arena_generation
    }

    pub fn gini(&self) -> u16 {
        self.gini
    }
}

impl MoveEdge {
    pub fn new(policy: u16, mov: chess::Move) -> Self {
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

    pub fn sum_evaluations(&self) -> i64 {
        self.sum_evaluations.load(Ordering::Relaxed)
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
            average: sum / i64::from(visits),
            visits,
        }
    }

    pub fn down(&self, virtual_loss: i64) {
        self.visits.fetch_add(1, Ordering::Relaxed);
        self.sum_evaluations
            .fetch_add(virtual_loss, Ordering::Relaxed);
    }

    pub fn up(&self, evaln: i64, virtual_loss: i64) {
        self.sum_evaluations
            .fetch_add(evaln - virtual_loss, Ordering::Relaxed);
    }

    pub fn add_visits(&self, delta: u32) -> u32 {
        self.visits.fetch_add(delta, Ordering::Relaxed) + delta
    }

    pub fn add_sum_evaluations(&self, delta: i64) -> i64 {
        self.sum_evaluations.fetch_add(delta, Ordering::Relaxed) + delta
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
        let child = self.child.load(Ordering::Acquire).cast_const();
        if child.is_null() {
            None
        } else {
            unsafe { Some(&*child) }
        }
    }

    pub fn set_child_ptr(&self, node: &PositionNode) {
        self.child
            .store(ptr::from_ref(node).cast_mut(), Ordering::Release);
    }
}

/// Select the edge with the most visits
#[inline]
pub fn select_edge_by_visits(edges: &[MoveEdge]) -> Option<&MoveEdge> {
    edges.iter().max_by_key(|x| x.visits())
}

/// Select the edge with the highest average reward
#[inline]
pub fn select_edge_by_rewards(edges: &[MoveEdge]) -> Option<&MoveEdge> {
    edges.iter().max_by_key(|x| x.reward().average)
}

/// Copy visit and evaluation statistics from src edges to dest edges
pub fn copy_edge_stats(dest: &[MoveEdge], src: &[MoveEdge]) {
    for i in 0..dest.len().min(src.len()) {
        dest[i].replace(&src[i]);
    }
}

/// Clear all child pointers for the given edges
pub fn clear_edge_children(edges: &[MoveEdge]) {
    for edge in edges {
        edge.child.store(ptr::null_mut(), Ordering::Relaxed);
    }
}

pub fn create_node<F>(
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

    let (node_uninit_ref, edges_uninit_ref) = alloc(move_eval.len())?;

    let mut sum_squares = 0.0_f32;
    #[allow(clippy::cast_sign_loss)]
    let edges_arena_ref = edges_uninit_ref.init_each(|i| {
        let p = move_eval[i];
        sum_squares += p * p;
        let policy_val = (p * SCALE) as u16;
        MoveEdge::new(policy_val, moves[i])
    });

    // Gini impurity (Gini-Simpson index)
    let gini_f32 = (1.0 - sum_squares).clamp(0.0, 1.0);
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    let gini = (gini_f32 * SCALE) as u16;

    // Capture the generation before `node_uninit_ref` is moved by `write`.
    let node_generation = node_uninit_ref.generation();

    // SAFETY: `node_uninit_ref` points to valid, uninitialized memory for a single PositionNode.
    // We are writing it exactly once.
    let node_arena_ref = node_uninit_ref.write(PositionNode::new(
        edges_arena_ref.into_non_null(),
        state_flag,
        gini,
        node_generation,
    ));

    Ok(node_arena_ref)
}

const _: () = assert!(mem::size_of::<PositionNode>() == 24);
const _: () = assert!(mem::size_of::<MoveEdge>() == 24);

pub fn print_size_list() {
    println!(
        "info string PositionNode {} MoveEdge {}",
        mem::size_of::<PositionNode>(),
        mem::size_of::<MoveEdge>(),
    );
}

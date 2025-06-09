use std::mem;
use std::ops::Deref;
use std::ptr::{self, null_mut, NonNull};
use std::sync::atomic::{AtomicI64, AtomicPtr, AtomicU32, Ordering};
use std::sync::LazyLock;

use crate::arena::{ArenaRef, Error as ArenaError};
use crate::chess;
use crate::evaluation::{self, Flag};
use crate::search::SCALE;
use crate::state::State;
use crate::transposition_table::AllocNodeResult;

pub struct MoveEdge {
    sum_evaluations: AtomicI64,
    visits: AtomicU32,
    policy: u16,
    mov: chess::Move,
    child: AtomicPtr<PositionNode>,
}

#[derive(Clone, Copy)]
pub struct Edges(NonNull<[MoveEdge]>);

impl Deref for Edges {
    type Target = [MoveEdge];

    fn deref(&self) -> &Self::Target {
        // SAFETY: The NonNull guarantees the pointer is valid and non-null.
        // The lifetime 'static is appropriate as the underlying data is either
        // from a static empty slice or from the Arena, which outlives the Edges.
        unsafe { self.0.as_ref() }
    }
}

// SAFETY: Edges wraps NonNull<[MoveEdge]>.
// MoveEdge is Send and Sync (due to Atomic types and chess::Move).
// The pointer itself is just a reference, not owning the data,
// and the underlying data (from static EMPTY_EDGES or Arena) is safely managed
// and accessible across threads.
unsafe impl Send for Edges {}
unsafe impl Sync for Edges {}

#[derive(Clone)]
pub struct PositionNode {
    edges: Edges,
    flag: Flag,
    generation: u32,
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

unsafe impl Sync for PositionNode {}

// Static empty slice for nodes with no moves
static EMPTY_EDGES: LazyLock<Edges> = LazyLock::new(|| {
    let s: &'static [MoveEdge] = &[];
    // SAFETY: A static empty slice is guaranteed to be non-null.
    Edges(NonNull::from(s))
});

// Macro to define static PositionNode instances
macro_rules! define_static_node {
    ($name:ident, $flag:expr) => {
        pub static $name: LazyLock<PositionNode> =
            LazyLock::new(|| PositionNode::new_static($flag));
    };
}

// These static nodes, created via `PositionNode::new_static`, are not allocated in an Arena,
// so their generation is 0 (invalid). This is fine as they are never looked up in the TT or cleared.
define_static_node!(WIN_NODE, Flag::TerminalWin);
define_static_node!(DRAW_NODE, Flag::TerminalDraw);
define_static_node!(LOSS_NODE, Flag::TerminalLoss);

define_static_node!(TABLEBASE_WIN_NODE, Flag::TablebaseWin);
define_static_node!(TABLEBASE_DRAW_NODE, Flag::TablebaseDraw);
define_static_node!(TABLEBASE_LOSS_NODE, Flag::TablebaseLoss);

define_static_node!(UNEXPANDED_NODE, Flag::Standard);

impl PositionNode {
    fn new(edges: Edges, flag: Flag, generation: u32) -> Self {
        Self {
            edges,
            flag,
            generation,
        }
    }

    pub fn new_static(flag: Flag) -> Self {
        Self {
            edges: *EMPTY_EDGES,
            flag,
            generation: 0,
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
        &self.edges
    }

    pub fn visits(&self) -> u64 {
        self.edges().iter().map(|x| u64::from(x.visits())).sum()
    }

    pub fn clear_children_links(&self) {
        for h in self.edges() {
            h.child.store(null_mut(), Ordering::Relaxed);
        }
    }

    pub fn select_child_by_rewards(&self) -> Option<&MoveEdge> {
        self.edges().iter().max_by_key(|x| x.reward().average)
    }

    pub fn select_child_by_visits(&self) -> Option<&MoveEdge> {
        self.edges().iter().max_by_key(|x| x.visits())
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
            average: sum / i64::from(visits),
            visits,
        }
    }

    /// Apply virtual loss during descent to discourage other threads from taking this path.
    /// This increments visits and subtracts virtual_loss from sum_evaluations.
    pub fn down(&self, virtual_loss: i64) {
        self.visits.fetch_add(1, Ordering::Relaxed);
        self.sum_evaluations.fetch_sub(virtual_loss, Ordering::Relaxed);
    }

    /// Update edge statistics during backpropagation.
    /// Adds the evaluation value and removes the virtual loss that was applied during descent.
    /// 
    /// Note: virtual_loss must be the same value that was passed to down() for this edge.
    /// The virtual loss is always added back (regardless of evaln's sign) to restore
    /// the original state before the temporary penalty.
    pub fn up(&self, evaln: i64, virtual_loss: i64) {
        // Add the actual evaluation (which may be positive or negative)
        self.sum_evaluations.fetch_add(evaln, Ordering::Relaxed);
        // Remove the virtual loss by adding it back (always positive)
        self.sum_evaluations.fetch_add(virtual_loss, Ordering::Relaxed);
    }

    /// Replaces this edge's statistics with values from another edge.
    /// 
    /// This is used when copying historical data from the transposition table.
    /// The operation uses Acquire/Release ordering to minimize the window where
    /// the two values could be inconsistent, though perfect atomicity is not
    /// guaranteed. This is acceptable for MCTS as small inconsistencies are
    /// absorbed by the algorithm's inherent noise.
    pub fn replace(&self, other: &MoveEdge) {
        // Load both values with Acquire ordering to ensure we see a consistent
        // view of any prior updates to the source edge
        let visits = other.visits.load(Ordering::Acquire);
        let sum = other.sum_evaluations.load(Ordering::Acquire);
        
        // Store both values with Release ordering to ensure they become visible
        // to other threads as close together as possible
        self.sum_evaluations.store(sum, Ordering::Release);
        self.visits.store(visits, Ordering::Release);
    }

    /// Debug version that attempts to detect concurrent modifications.
    /// Returns true if the copy appeared consistent, false if a concurrent
    /// modification was detected (in which case it still performs the copy).
    #[cfg(debug_assertions)]
    pub fn replace_checked(&self, other: &MoveEdge) -> bool {
        const MAX_RETRIES: u32 = 3;
        
        // Try to get a consistent snapshot
        for _ in 0..MAX_RETRIES {
            let visits1 = other.visits.load(Ordering::Acquire);
            let sum = other.sum_evaluations.load(Ordering::Acquire);
            let visits2 = other.visits.load(Ordering::Acquire);
            
            // If visits didn't change, we likely got a consistent snapshot
            if visits1 == visits2 {
                self.sum_evaluations.store(sum, Ordering::Release);
                self.visits.store(visits1, Ordering::Release);
                return true;
            }
        }
        
        // Fall back to best-effort copy
        self.replace(other);
        false
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

    #[allow(clippy::cast_sign_loss)]
    let edges_arena_ref = edges_uninit_ref.init_each(|i| {
        let policy_val = (move_eval[i] * SCALE) as u16;
        MoveEdge::new(policy_val, moves[i])
    });

    // Capture the generation before `node_uninit_ref` is moved by `write`.
    let node_generation = node_uninit_ref.generation();

    // SAFETY: `node_uninit_ref` points to valid, uninitialized memory for a single PositionNode.
    // We are writing it exactly once.
    let node_arena_ref = node_uninit_ref.write(PositionNode::new(
        Edges(edges_arena_ref.into_non_null()),
        state_flag,
        node_generation,
    ));

    Ok(node_arena_ref)
}

pub fn print_size_list() {
    println!(
        "info string PositionNode {} MoveEdge {}",
        mem::size_of::<PositionNode>(),
        mem::size_of::<MoveEdge>(),
    );
}

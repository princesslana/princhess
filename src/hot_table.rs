use std::array;
use std::ptr;

use crate::graph::{MoveEdge, PositionNode, Reward};

pub const FLUSH_PLAYOUTS: usize = 1024;

const HOT_SLOTS: usize = 64;

// A node is hot from K × threads × FLUSH_PLAYOUTS visits, or 1 / ROOT_SHARE of the root's visits
const HOT_THRESHOLD_K: u64 = 4;
const HOT_THRESHOLD_ROOT_SHARE: u64 = 100;

/// An edge's shared stats as of the last flush, plus this thread's updates since.
#[derive(Clone, Copy)]
pub struct HotEdge {
    visits: u32,
    sum_evaluations: i64,
    average: i64,
    synced_visits: u32,
    synced_sum_evaluations: i64,
}

impl HotEdge {
    fn new(edge: &MoveEdge) -> Self {
        let mut hot_edge = Self {
            visits: 0,
            sum_evaluations: 0,
            average: 0,
            synced_visits: 0,
            synced_sum_evaluations: 0,
        };
        hot_edge.flush(edge);
        hot_edge
    }

    pub fn visits(&self) -> u32 {
        self.visits
    }

    pub fn sum_evaluations(&self) -> i64 {
        self.sum_evaluations
    }

    pub fn average(&self) -> i64 {
        self.average
    }

    pub fn reward(&self) -> Reward {
        Reward {
            average: self.average,
            visits: self.visits,
        }
    }

    fn down(&mut self, virtual_loss: i64) {
        self.visits += 1;
        self.sum_evaluations += virtual_loss;
    }

    fn up(&mut self, evaln: i64, virtual_loss: i64) {
        self.sum_evaluations += evaln - virtual_loss;
        self.update_average();
    }

    fn flush(&mut self, edge: &MoveEdge) {
        let delta_visits = self.visits - self.synced_visits;
        let delta_sum = self.sum_evaluations - self.synced_sum_evaluations;

        self.visits = if delta_visits > 0 {
            edge.add_visits(delta_visits)
        } else {
            edge.visits()
        };
        self.sum_evaluations = if delta_sum != 0 {
            edge.add_sum_evaluations(delta_sum)
        } else {
            edge.sum_evaluations()
        };
        self.synced_visits = self.visits;
        self.synced_sum_evaluations = self.sum_evaluations;
        self.update_average();
    }

    fn update_average(&mut self) {
        self.average = if self.visits == 0 {
            Reward::ZERO.average
        } else {
            self.sum_evaluations / i64::from(self.visits)
        };
    }
}

pub struct HotSlot<'a> {
    node: Option<&'a PositionNode>,
    generation: u32,
    used: bool,
    visits: u64,
    edges: Vec<HotEdge>,
}

impl HotSlot<'_> {
    fn empty() -> Self {
        Self {
            node: None,
            generation: 0,
            used: false,
            visits: 0,
            edges: Vec::new(),
        }
    }

    fn claim(&mut self, edges: &[MoveEdge]) {
        self.edges.clear();
        self.edges.extend(edges.iter().map(HotEdge::new));
        self.visits = self.edges.iter().map(|e| u64::from(e.visits)).sum();
    }

    fn flush(&mut self, edges: &[MoveEdge]) {
        for (hot_edge, edge) in self.edges.iter_mut().zip(edges) {
            hot_edge.flush(edge);
        }
        self.visits = self.edges.iter().map(|e| u64::from(e.visits)).sum();
    }

    pub fn visits(&self) -> u64 {
        self.visits
    }

    pub fn edges(&self) -> &[HotEdge] {
        &self.edges
    }

    pub fn down(&mut self, idx: usize, virtual_loss: i64) {
        self.visits += 1;
        self.edges[idx].down(virtual_loss);
    }

    pub fn up(&mut self, idx: usize, evaln: i64, virtual_loss: i64) {
        self.edges[idx].up(evaln, virtual_loss);
    }
}

/// Stores the most visited nodes' edge stats per thread, to avoid contention on them.
pub struct HotTable<'a> {
    pub threshold: u64,
    min_threshold: u64,
    slots: [HotSlot<'a>; HOT_SLOTS + 1],
}

impl<'a> HotTable<'a> {
    pub const ROOT: usize = HOT_SLOTS;

    pub fn new(threads: usize, root_edges: &[MoveEdge]) -> Self {
        // A single thread has no contention to avoid
        let min_threshold = if threads == 1 {
            u64::MAX
        } else {
            HOT_THRESHOLD_K * threads as u64 * FLUSH_PLAYOUTS as u64
        };

        let mut slots: [HotSlot<'a>; HOT_SLOTS + 1] = array::from_fn(|_| HotSlot::empty());
        slots[Self::ROOT].claim(root_edges);

        Self {
            threshold: min_threshold,
            min_threshold,
            slots,
        }
    }

    pub fn root(&self) -> &HotSlot<'a> {
        &self.slots[Self::ROOT]
    }

    pub fn slot(&mut self, node: &'a PositionNode) -> Option<usize> {
        let hash = (ptr::from_ref(node).addr() >> 3).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let idx = hash >> (usize::BITS - HOT_SLOTS.trailing_zeros());
        let slot = &mut self.slots[idx];

        match slot.node {
            Some(held) if ptr::eq(held, node) && slot.generation == node.generation() => {}
            Some(_) => return None,
            None => {
                slot.node = Some(node);
                slot.generation = node.generation();
                slot.claim(node.edges());
            }
        }

        slot.used = true;
        Some(idx)
    }

    pub fn get(&self, idx: usize) -> &HotSlot<'a> {
        &self.slots[idx]
    }

    pub fn get_mut(&mut self, idx: usize) -> &mut HotSlot<'a> {
        &mut self.slots[idx]
    }

    pub fn flush(&mut self, root_edges: &[MoveEdge], generation: u32) {
        self.slots[Self::ROOT].flush(root_edges);

        for slot in &mut self.slots[..HOT_SLOTS] {
            let Some(node) = slot.node else {
                continue;
            };

            if !slot.used || slot.generation != generation {
                slot.node = None;
                continue;
            }

            slot.used = false;
            slot.flush(node.edges());
        }

        self.threshold = self
            .min_threshold
            .max(self.root().visits() / HOT_THRESHOLD_ROOT_SHARE);
    }
}

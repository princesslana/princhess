use bytemuck::{allocation, Pod, Zeroable};
use goober::{FeedForwardNetwork, OutputLayer, SparseVector};
use std::fmt::{self, Display};
use std::ops::AddAssign;
use std::path::Path;

use crate::chess::{Piece, Square};
use crate::nets::save_to_bin;
use crate::subnets::{LayerNetwork, LinearNetwork, QuantizedLayerNetwork, QuantizedLinearNetwork};

#[must_use]
#[derive(Debug, Copy, Clone)]
pub struct MoveIndex {
    piece: Piece,
    from_sq: Square,
    to_sq: Square,
    from_threats: u8,
    to_threats: u8,
}

impl MoveIndex {
    const FROM_BUCKETS: usize = 4;
    const TO_BUCKETS: usize = 10;

    pub const FROM_COUNT: usize = 64 * Self::FROM_BUCKETS;
    pub const TO_COUNT: usize = 64 * Self::TO_BUCKETS;

    const THREAT_SHIFT: u8 = 0;
    const DEFEND_SHIFT: u8 = 1;
    const SEE_SHIFT: u8 = 0;

    pub fn new(piece: Piece, from: Square, to: Square) -> Self {
        Self {
            piece,
            from_sq: from,
            to_sq: to,
            from_threats: 0,
            to_threats: 0,
        }
    }

    pub fn set_from_threat(&mut self, is_threat: bool) {
        self.from_threats |= u8::from(is_threat) << Self::THREAT_SHIFT;
    }

    pub fn set_from_defend(&mut self, is_defend: bool) {
        self.from_threats |= u8::from(is_defend) << Self::DEFEND_SHIFT;
    }

    pub fn set_to_good_see(&mut self, is_good_see: bool) {
        self.to_threats |= u8::from(is_good_see) << Self::SEE_SHIFT;
    }

    #[must_use]
    pub fn from_index(&self) -> usize {
        let bucket = usize::from(self.from_threats);
        bucket * 64 + self.from_sq.index()
    }

    #[must_use]
    pub fn to_index(&self) -> usize {
        let bucket = match self.piece {
            Piece::KING => 0,
            Piece::PAWN => 1,
            p => 2 + usize::from(self.to_threats) * 4 + p.index() - 1,
        };
        bucket * 64 + self.to_sq.index()
    }
}

type FromNetwork = LinearNetwork;
type ToNetwork = LayerNetwork;

#[allow(clippy::module_name_repetitions)]
#[derive(Zeroable)]
pub struct PolicyNetwork {
    from: [FromNetwork; MoveIndex::FROM_COUNT],
    to: [ToNetwork; MoveIndex::TO_COUNT],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct QuantizedPolicyNetwork {
    from: QuantizedLinearNetwork<{ MoveIndex::FROM_COUNT }>,
    to: QuantizedLayerNetwork<{ MoveIndex::TO_COUNT }>,
}

#[allow(clippy::module_name_repetitions)]
pub struct PolicyCount {
    pub from: [u64; MoveIndex::FROM_COUNT],
    pub to: [u64; MoveIndex::TO_COUNT],
}

impl Display for PolicyNetwork {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let from = format!("from: [{}; {}]", self.from[0], MoveIndex::FROM_COUNT);
        let to = format!("to: [{}; {}]", self.to[0], MoveIndex::TO_COUNT);
        write!(f, "{from} * {to}")
    }
}

impl AddAssign<&Self> for PolicyNetwork {
    fn add_assign(&mut self, rhs: &Self) {
        for (lhs_subnet, rhs_subnet) in self.from.iter_mut().zip(&rhs.from) {
            *lhs_subnet += rhs_subnet;
        }

        for (lhs_subnet, rhs_subnet) in self.to.iter_mut().zip(&rhs.to) {
            *lhs_subnet += rhs_subnet;
        }
    }
}

impl PolicyNetwork {
    #[must_use]
    pub fn zeroed() -> Box<Self> {
        allocation::zeroed_box()
    }

    #[must_use]
    pub fn random() -> Box<Self> {
        let mut network = Self::zeroed();

        for subnetwork in &mut network.from {
            subnetwork.randomize();
        }

        for subnetwork in &mut network.to {
            subnetwork.randomize();
        }

        network
    }

    fn get_from(&self, from_idx: usize) -> &FromNetwork {
        unsafe { self.from.get_unchecked(from_idx) }
    }

    fn get_to(&self, to_idx: usize) -> &ToNetwork {
        unsafe { self.to.get_unchecked(to_idx) }
    }

    pub fn get_all<I: Iterator<Item = MoveIndex>>(
        &self,
        features: &SparseVector,
        move_idxes: I,
        out: &mut Vec<f32>,
    ) {
        let mut from_logits = [None; Square::COUNT];

        for move_idx in move_idxes {
            let from_idx = move_idx.from_index();
            let to_idx = move_idx.to_index();

            let from = from_logits[from_idx % Square::COUNT]
                .get_or_insert_with(|| self.get_from(from_idx).out(features));
            let to = self.get_to(to_idx).out(features);

            out.push(from.dot(&to));
        }
    }

    pub fn adam(&mut self, g: &Self, m: &mut Self, v: &mut Self, count: &PolicyCount, lr: f32) {
        for subnet_idx in 0..self.from.len() {
            match count.from[subnet_idx] {
                0 => continue,
                n => self.from[subnet_idx].adam(
                    &g.from[subnet_idx],
                    &mut m.from[subnet_idx],
                    &mut v.from[subnet_idx],
                    1.0 / n as f32,
                    lr,
                ),
            }
        }

        for subnet_idx in 0..self.to.len() {
            match count.to[subnet_idx] {
                0 => continue,
                n => self.to[subnet_idx].adam(
                    &g.to[subnet_idx],
                    &mut m.to[subnet_idx],
                    &mut v.to[subnet_idx],
                    1.0 / n as f32,
                    lr,
                ),
            }
        }
    }

    pub fn backprop(&self, features: &SparseVector, g: &mut Self, move_idx: MoveIndex, error: f32) {
        let from = &self.from[move_idx.from_index()];
        let to = &self.to[move_idx.to_index()];

        let from_out = from.out_with_layers(features);
        let to_out = to.out_with_layers(features);

        from.backprop(
            features,
            &mut g.from[move_idx.from_index()],
            error * to_out.output_layer(),
            &from_out,
        );

        to.backprop(
            features,
            &mut g.to[move_idx.to_index()],
            error * from_out.output_layer(),
            &to_out,
        );
    }

    #[must_use]
    pub fn to_boxed_and_quantized(&self) -> Box<QuantizedPolicyNetwork> {
        let mut result: Box<QuantizedPolicyNetwork> = allocation::zeroed_box();

        result.from = *QuantizedLinearNetwork::boxed_from(&self.from);
        result.to = *QuantizedLayerNetwork::boxed_from(&self.to);

        result
    }
}

impl QuantizedPolicyNetwork {
    #[must_use]
    pub fn zeroed() -> Box<Self> {
        allocation::zeroed_box()
    }

    pub fn save_to_bin(&self, dir: &Path) {
        save_to_bin(dir, "policy.bin", self);
    }

    pub fn get_all<I: Iterator<Item = MoveIndex>>(
        &self,
        features: &SparseVector,
        move_idxes: I,
        out: &mut Vec<f32>,
    ) {
        for move_idx in move_idxes {
            let from_idx = move_idx.from_index();
            let to_idx = move_idx.to_index();

            let mut from = self.from.get_bias(from_idx);
            let mut to = self.to.get_bias(to_idx);

            for f in features.iter() {
                self.from.set(from_idx, *f, &mut from);
                self.to.set(to_idx, *f, &mut to);
            }

            let to_out = self.to.out(to_idx, &to);

            out.push(from.dot_relu(&to_out));
        }
    }
}

impl PolicyCount {
    #[must_use]
    pub fn zeroed() -> Self {
        Self {
            from: [0; MoveIndex::FROM_COUNT],
            to: [0; MoveIndex::TO_COUNT],
        }
    }

    pub fn increment(&mut self, move_idx: MoveIndex) {
        self.from[move_idx.from_index()] += 1;
        self.to[move_idx.to_index()] += 1;
    }
}

impl AddAssign<&Self> for PolicyCount {
    fn add_assign(&mut self, rhs: &Self) {
        for (lhs, rhs) in self.from.iter_mut().zip(&rhs.from) {
            *lhs += rhs;
        }

        for (lhs, rhs) in self.to.iter_mut().zip(&rhs.to) {
            *lhs += rhs;
        }
    }
}

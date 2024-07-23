use goober::activation::ReLU;
use goober::layer::SparseConnected;
use goober::{FeedForwardNetwork, OutputLayer, SparseVector};
use std::fmt::{self, Display};
use std::mem;
use std::ops::AddAssign;
use std::path::Path;

use crate::chess::{MoveIndex, Square};
use crate::math::{randomize_sparse, Rng};
use crate::mem::{boxed_and_zeroed, Align16};
use crate::nets::{q_i16, save_to_bin, Accumulator};
use crate::state;

const INPUT_SIZE: usize = state::POLICY_NUMBER_FEATURES;
const ATTENTION_SIZE: usize = 8;

const QA: i32 = 256;
const QAA: i32 = QA * QA;

type Output = SparseConnected<ReLU, INPUT_SIZE, ATTENTION_SIZE>;

type QuantizedOutputWeights = [Align16<Accumulator<ATTENTION_SIZE>>; INPUT_SIZE];
type QuantizedOutputBias = Align16<Accumulator<ATTENTION_SIZE>>;

type RawOutputWeights = [[i16; ATTENTION_SIZE]; INPUT_SIZE];
type RawOutputBias = [i16; ATTENTION_SIZE];

#[repr(C)]
#[derive(FeedForwardNetwork)]
pub struct FromNetwork {
    output: Output,
}

impl FromNetwork {
    pub fn randomize(&mut self) {
        let mut rng = Rng::default();

        randomize_sparse(&mut self.output, &mut rng);
    }
}

#[repr(C)]
#[derive(FeedForwardNetwork)]
pub struct ToNetwork {
    output: Output,
}

impl ToNetwork {
    pub fn randomize(&mut self) {
        let mut rng = Rng::default();

        randomize_sparse(&mut self.output, &mut rng);
    }
}

#[allow(clippy::module_name_repetitions)]
pub struct PolicyNetwork {
    from: [FromNetwork; MoveIndex::FROM_COUNT],
    to: [ToNetwork; MoveIndex::TO_COUNT],
}

#[repr(C)]
pub struct QuantizedPolicyNetwork {
    from_weights: [QuantizedOutputWeights; MoveIndex::FROM_COUNT],
    from_bias: [QuantizedOutputBias; MoveIndex::FROM_COUNT],
    to_weights: [QuantizedOutputWeights; MoveIndex::TO_COUNT],
    to_bias: [QuantizedOutputBias; MoveIndex::TO_COUNT],
}

impl Display for PolicyNetwork {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let from = format!(
            "from: [{INPUT_SIZE}->{ATTENTION_SIZE}; {}]",
            MoveIndex::FROM_COUNT
        );
        let to = format!(
            "to: [{INPUT_SIZE}->{ATTENTION_SIZE}; {}]",
            MoveIndex::TO_COUNT
        );
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
        boxed_and_zeroed()
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

    pub fn adam(&mut self, g: &Self, m: &mut Self, v: &mut Self, adj: f32, lr: f32) {
        for subnet_idx in 0..self.from.len() {
            self.from[subnet_idx].adam(
                &g.from[subnet_idx],
                &mut m.from[subnet_idx],
                &mut v.from[subnet_idx],
                adj,
                lr,
            );
        }

        for subnet_idx in 0..self.to.len() {
            self.to[subnet_idx].adam(
                &g.to[subnet_idx],
                &mut m.to[subnet_idx],
                &mut v.to[subnet_idx],
                adj,
                lr,
            );
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
        let mut from_weights: Box<[RawOutputWeights; MoveIndex::FROM_COUNT]> = boxed_and_zeroed();
        let mut from_bias: Box<[RawOutputBias; MoveIndex::FROM_COUNT]> = boxed_and_zeroed();
        let mut to_weights: Box<[RawOutputWeights; MoveIndex::TO_COUNT]> = boxed_and_zeroed();
        let mut to_bias: Box<[RawOutputBias; MoveIndex::TO_COUNT]> = boxed_and_zeroed();

        for (subnet, raw) in self.from.iter().zip(from_weights.iter_mut()) {
            for (row_idx, weights) in raw.iter_mut().enumerate() {
                let row = subnet.output.weights_row(row_idx);
                for weight_idx in 0..ATTENTION_SIZE {
                    weights[weight_idx] = q_i16(row[weight_idx], QA);
                }
            }
        }

        for (subnet, raw) in self.from.iter().zip(from_bias.iter_mut()) {
            for (weight_idx, bias) in raw.iter_mut().enumerate() {
                *bias = q_i16(subnet.output.bias()[weight_idx], QA);
            }
        }

        for (subnet, raw) in self.to.iter().zip(to_weights.iter_mut()) {
            for (row_idx, weights) in raw.iter_mut().enumerate() {
                let row = subnet.output.weights_row(row_idx);
                for weight_idx in 0..ATTENTION_SIZE {
                    weights[weight_idx] = q_i16(row[weight_idx], QA);
                }
            }
        }

        for (subnet, raw) in self.to.iter().zip(to_bias.iter_mut()) {
            for (weight_idx, bias) in raw.iter_mut().enumerate() {
                *bias = q_i16(subnet.output.bias()[weight_idx], QA);
            }
        }

        QuantizedPolicyNetwork::from_slices(&from_weights, &from_bias, &to_weights, &to_bias)
    }
}

impl QuantizedPolicyNetwork {
    #[must_use]
    pub fn zeroed() -> Box<Self> {
        boxed_and_zeroed()
    }

    #[must_use]
    pub fn from_slices(
        from_weights: &[RawOutputWeights; MoveIndex::FROM_COUNT],
        from_bias: &[RawOutputBias; MoveIndex::FROM_COUNT],
        to_weights: &[RawOutputWeights; MoveIndex::TO_COUNT],
        to_bias: &[RawOutputBias; MoveIndex::TO_COUNT],
    ) -> Box<Self> {
        let mut network = Self::zeroed();

        network.from_weights = unsafe {
            mem::transmute::<
                [RawOutputWeights; MoveIndex::FROM_COUNT],
                [QuantizedOutputWeights; MoveIndex::FROM_COUNT],
            >(*from_weights)
        };

        network.from_bias = unsafe {
            mem::transmute::<
                [RawOutputBias; MoveIndex::FROM_COUNT],
                [QuantizedOutputBias; MoveIndex::FROM_COUNT],
            >(*from_bias)
        };

        network.to_weights = unsafe {
            mem::transmute::<
                [RawOutputWeights; MoveIndex::TO_COUNT],
                [QuantizedOutputWeights; MoveIndex::TO_COUNT],
            >(*to_weights)
        };

        network.to_bias = unsafe {
            mem::transmute::<
                [RawOutputBias; MoveIndex::TO_COUNT],
                [QuantizedOutputBias; MoveIndex::TO_COUNT],
            >(*to_bias)
        };

        network
    }

    pub fn save_to_bin(&self, dir: &Path) {
        save_to_bin(dir, "policy.bin", self);
    }

    fn get_from_bias(&self, from_idx: usize) -> Accumulator<ATTENTION_SIZE> {
        unsafe { **self.from_bias.get_unchecked(from_idx) }
    }

    fn get_from_weights(&self, from_idx: usize, feat_idx: usize) -> &Accumulator<ATTENTION_SIZE> {
        unsafe {
            self.from_weights
                .get_unchecked(from_idx)
                .get_unchecked(feat_idx)
        }
    }

    fn get_to_bias(&self, to_idx: usize) -> Accumulator<ATTENTION_SIZE> {
        unsafe { **self.to_bias.get_unchecked(to_idx) }
    }

    fn get_to_weights(&self, to_idx: usize, feat_idx: usize) -> &Accumulator<ATTENTION_SIZE> {
        unsafe {
            self.to_weights
                .get_unchecked(to_idx)
                .get_unchecked(feat_idx)
        }
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

            let mut from = self.get_from_bias(from_idx);
            let mut to = self.get_to_bias(to_idx);

            for f in features.iter() {
                let from_weight = self.get_from_weights(from_idx, *f);
                let to_weight = self.get_to_weights(to_idx, *f);

                from.set(from_weight);
                to.set(to_weight);
            }

            out.push(from.dot_relu(&to) as f32 / QAA as f32);
        }
    }
}

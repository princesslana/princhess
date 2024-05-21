#![allow(clippy::module_name_repetitions)]

use goober::activation::{Activation, Identity, ReLU};
use goober::layer::SparseConnected;
use goober::{FeedForwardNetwork, Vector};
use std::fmt::{self, Display, Formatter};
use std::ops::AddAssign;

use crate::evaluation;
use crate::math::Rng;
use crate::mem::boxed_and_zeroed;
use crate::state;
use crate::train::{q_i16, randomize_sparse};

const INPUT_SIZE: usize = state::POLICY_NUMBER_FEATURES;
const OUTPUT_SIZE: usize = 384;

const QA: f32 = 256.;

type ConstantLayer = SparseConnected<Identity, INPUT_SIZE, 1>;
type LeftLayer = SparseConnected<Identity, INPUT_SIZE, 1>;
type RightLayer = SparseConnected<ReLU, INPUT_SIZE, 1>;

pub struct PolicyNetwork {
    pub constant: [ConstantLayer; OUTPUT_SIZE],
    pub left: [LeftLayer; OUTPUT_SIZE],
    pub right: [RightLayer; OUTPUT_SIZE],
}

impl AddAssign<&Self> for PolicyNetwork {
    fn add_assign(&mut self, rhs: &Self) {
        for (s, r) in self.constant.iter_mut().zip(&rhs.constant) {
            *s += r;
        }
        for (s, r) in self.left.iter_mut().zip(&rhs.left) {
            *s += r;
        }
        for (s, r) in self.right.iter_mut().zip(&rhs.right) {
            *s += r;
        }
    }
}

impl Display for PolicyNetwork {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let layer = format!("[{INPUT_SIZE}->1; {OUTPUT_SIZE}]");
        write!(f, "c: {layer} + l: {layer} * r: {layer}",)
    }
}

impl PolicyNetwork {
    pub const INPUT_SIZE: usize = INPUT_SIZE;
    pub const OUTPUT_SIZE: usize = OUTPUT_SIZE;

    #[must_use]
    pub fn zeroed() -> Box<Self> {
        boxed_and_zeroed()
    }

    #[must_use]
    pub fn random() -> Box<Self> {
        let mut rng = Rng::default();

        let mut network = Self::zeroed();

        for constant in &mut network.constant {
            randomize_sparse(constant, &mut rng);
        }

        for left in &mut network.left {
            randomize_sparse(left, &mut rng);
        }

        for right in &mut network.right {
            randomize_sparse(right, &mut rng);
        }

        network
    }

    #[must_use]
    fn constant_weights(&self, input_idx: usize) -> Vector<OUTPUT_SIZE> {
        weights(&self.constant, input_idx)
    }

    #[must_use]
    pub fn constant_bias(&self) -> Vector<OUTPUT_SIZE> {
        bias(&self.constant)
    }

    #[must_use]
    pub fn left_weights(&self, input_idx: usize) -> Vector<OUTPUT_SIZE> {
        weights(&self.left, input_idx)
    }

    #[must_use]
    pub fn left_bias(&self) -> Vector<OUTPUT_SIZE> {
        bias(&self.left)
    }

    #[must_use]
    pub fn right_weights(&self, input_idx: usize) -> Vector<OUTPUT_SIZE> {
        weights(&self.right, input_idx)
    }

    #[must_use]
    pub fn right_bias(&self) -> Vector<OUTPUT_SIZE> {
        bias(&self.right)
    }

    pub fn adam(&mut self, g: &Self, m: &mut Self, v: &mut Self, adj: f32, lr: f32) {
        for idx in 0..OUTPUT_SIZE {
            self.constant[idx].adam(
                &g.constant[idx],
                &mut m.constant[idx],
                &mut v.constant[idx],
                adj,
                lr,
            );

            self.left[idx].adam(&g.left[idx], &mut m.left[idx], &mut v.left[idx], adj, lr);

            self.right[idx].adam(&g.right[idx], &mut m.right[idx], &mut v.right[idx], adj, lr);
        }
    }

    pub fn decay_weights(&mut self, decay: f32) {
        for idx in 0..OUTPUT_SIZE {
            for row_idx in 0..INPUT_SIZE {
                let cr = self.constant[idx].weights_row_mut(row_idx);
                let lr = self.left[idx].weights_row_mut(row_idx);
                let rr = self.right[idx].weights_row_mut(row_idx);

                cr[0] *= decay;
                lr[0] *= decay;
                rr[0] *= decay;
            }

            self.constant[idx].bias_mut()[0] *= decay;
            self.left[idx].bias_mut()[0] *= decay;
            self.right[idx].bias_mut()[0] *= decay;
        }
    }

    #[must_use]
    pub fn to_boxed_evaluation_network(&self) -> Box<evaluation::PolicyNetwork> {
        let mut left_weights: Box<[[i16; OUTPUT_SIZE]; INPUT_SIZE]> = boxed_and_zeroed();
        let mut left_bias = [0; OUTPUT_SIZE];
        let mut right_weights: Box<[[i16; OUTPUT_SIZE]; INPUT_SIZE]> = boxed_and_zeroed();
        let mut right_bias = [0; OUTPUT_SIZE];
        let mut constant_weights: Box<[[i16; OUTPUT_SIZE]; INPUT_SIZE]> = boxed_and_zeroed();
        let mut constant_bias = [0; OUTPUT_SIZE];

        for (row_idx, weights) in left_weights.iter_mut().enumerate() {
            let row = self.left_weights(row_idx);
            for weight_idx in 0..OUTPUT_SIZE {
                weights[weight_idx] = q_i16(row[weight_idx], QA);
            }
        }

        for (weight_idx, bias) in left_bias.iter_mut().enumerate() {
            *bias = q_i16(self.left_bias()[weight_idx], QA);
        }

        for (row_idx, weights) in right_weights.iter_mut().enumerate() {
            let row = self.right_weights(row_idx);
            for weight_idx in 0..OUTPUT_SIZE {
                weights[weight_idx] = q_i16(row[weight_idx], QA);
            }
        }

        for (weight_idx, bias) in right_bias.iter_mut().enumerate() {
            *bias = q_i16(self.right_bias()[weight_idx], QA);
        }

        for (row_idx, weights) in constant_weights.iter_mut().enumerate() {
            let row = self.constant_weights(row_idx);
            for weight_idx in 0..OUTPUT_SIZE {
                weights[weight_idx] = q_i16(row[weight_idx], QA);
            }
        }

        for (weight_idx, bias) in constant_bias.iter_mut().enumerate() {
            *bias = q_i16(self.constant_bias()[weight_idx], QA);
        }

        evaluation::PolicyNetwork::from_slices(
            &left_weights,
            &left_bias,
            &right_weights,
            &right_bias,
            &constant_weights,
            &constant_bias,
        )
    }
}

fn weights<A: Activation, const I: usize>(
    layers: &[SparseConnected<A, I, 1>; OUTPUT_SIZE],
    input_idx: usize,
) -> Vector<OUTPUT_SIZE> {
    let mut weights = Vector::zeroed();

    for idx in 0..OUTPUT_SIZE {
        weights[idx] = layers[idx].weights_row(input_idx)[0];
    }

    weights
}

fn bias<A: Activation, const I: usize>(
    layers: &[SparseConnected<A, I, 1>; OUTPUT_SIZE],
) -> Vector<OUTPUT_SIZE> {
    let mut bias = Vector::zeroed();

    for (i, layer) in layers.iter().enumerate() {
        bias[i] = layer.bias()[0];
    }

    bias
}

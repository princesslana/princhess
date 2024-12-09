use bytemuck::{allocation, Pod, Zeroable};
use goober::activation::ReLU;
use goober::layer::SparseConnected;
use goober::FeedForwardNetwork;
use std::fmt::{self, Display};

use crate::math::{randomize_sparse, Rng};
use crate::mem::Align16;
use crate::nets::{q_i16, Accumulator};
use crate::state;

const INPUT_SIZE: usize = state::POLICY_NUMBER_FEATURES;
const ATTENTION_SIZE: usize = 8;

pub const QA: i32 = 256;
pub const QAA: i32 = QA * QA;

type Linear = SparseConnected<ReLU, INPUT_SIZE, ATTENTION_SIZE>;

type QuantizedLinearWeights = [Align16<Accumulator<i16, ATTENTION_SIZE>>; INPUT_SIZE];
type QuantizedLinearBias = Align16<Accumulator<i16, ATTENTION_SIZE>>;

type RawLinearWeights = Align16<[[i16; ATTENTION_SIZE]; INPUT_SIZE]>;
type RawLinearBias = Align16<[i16; ATTENTION_SIZE]>;

#[repr(C)]
#[derive(FeedForwardNetwork)]
pub struct LinearNetwork {
    output: Linear,
}

unsafe impl Zeroable for LinearNetwork {}

impl LinearNetwork {
    pub fn randomize(&mut self) {
        let mut rng = Rng::default();

        randomize_sparse(&mut self.output, &mut rng);
    }
}

impl Display for LinearNetwork {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{INPUT_SIZE}->{ATTENTION_SIZE}")
    }
}

#[repr(C)]
#[derive(Copy, Clone, Zeroable)]
pub struct QuantizedLinearNetwork<const N: usize> {
    weights: [QuantizedLinearWeights; N],
    bias: [QuantizedLinearBias; N],
}

unsafe impl<const N: usize> Pod for QuantizedLinearNetwork<N> {}

impl<const N: usize> QuantizedLinearNetwork<N> {
    #[must_use]
    pub fn boxed_from(subnets: &[LinearNetwork; N]) -> Box<Self> {
        let mut weights: Box<[RawLinearWeights; N]> = allocation::zeroed_box();
        let mut bias: Box<[RawLinearBias; N]> = allocation::zeroed_box();

        for (subnet, raw) in subnets.iter().zip(weights.iter_mut()) {
            for (row_idx, weights) in raw.iter_mut().enumerate() {
                let row = subnet.output.weights_row(row_idx);
                for weight_idx in 0..ATTENTION_SIZE {
                    weights[weight_idx] = q_i16(row[weight_idx], QA);
                }
            }
        }

        for (subnet, raw) in subnets.iter().zip(bias.iter_mut()) {
            for (weight_idx, bias) in raw.iter_mut().enumerate() {
                *bias = q_i16(subnet.output.bias()[weight_idx], QA);
            }
        }

        let mut result: Box<Self> = allocation::zeroed_box();

        result.weights = *bytemuck::must_cast_ref(&*weights);
        result.bias = *bytemuck::must_cast_ref(&*bias);

        result
    }

    pub fn get_bias(&self, idx: usize) -> Accumulator<i16, ATTENTION_SIZE> {
        unsafe { **self.bias.get_unchecked(idx) }
    }

    fn get_weights(&self, idx: usize, feat_idx: usize) -> &Accumulator<i16, ATTENTION_SIZE> {
        unsafe { self.weights.get_unchecked(idx).get_unchecked(feat_idx) }
    }

    pub fn set(&self, idx: usize, feat_idx: usize, acc: &mut Accumulator<i16, ATTENTION_SIZE>) {
        acc.set(self.get_weights(idx, feat_idx));
    }
}

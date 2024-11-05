use bytemuck::{allocation, Pod, Zeroable};
use goober::activation::ReLU;
use goober::layer::{DenseConnected, SparseConnected};
use goober::FeedForwardNetwork;
use std::fmt::{self, Display};

use crate::math::{randomize_dense, randomize_sparse, Rng};
use crate::mem::Align16;
use crate::nets::{q_i16, q_i32, relu, Accumulator};
use crate::state;

const INPUT_SIZE: usize = state::POLICY_NUMBER_FEATURES;
const ATTENTION_SIZE: usize = 8;

pub const QA: i32 = 256;
pub const QB: i32 = 256;
pub const QAA: i32 = QA * QA;
pub const QAB: i32 = QA * QB;

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

const HIDDEN_SIZE: usize = 8;

type Feature = SparseConnected<ReLU, INPUT_SIZE, HIDDEN_SIZE>;
type Output = DenseConnected<ReLU, HIDDEN_SIZE, ATTENTION_SIZE>;

type QuantizedFeatureWeights = [Align16<Accumulator<i16, HIDDEN_SIZE>>; INPUT_SIZE];
type QuantizedFeatureBias = Align16<Accumulator<i16, HIDDEN_SIZE>>;
type QuantizedOutputWeights = [Align16<Accumulator<i16, ATTENTION_SIZE>>; HIDDEN_SIZE];
type QuantizedOutputBias = Align16<Accumulator<i32, ATTENTION_SIZE>>;

type RawFeatureWeights = Align16<[[i16; HIDDEN_SIZE]; INPUT_SIZE]>;
type RawFeatureBias = Align16<[i16; HIDDEN_SIZE]>;
type RawOutputWeights = Align16<[[i16; ATTENTION_SIZE]; HIDDEN_SIZE]>;
type RawOutputBias = Align16<[i32; ATTENTION_SIZE]>;

#[repr(C)]
#[derive(FeedForwardNetwork)]
pub struct LayerNetwork {
    feature: Feature,
    output: Output,
}

unsafe impl Zeroable for LayerNetwork {}

impl Display for LayerNetwork {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{INPUT_SIZE}->{HIDDEN_SIZE}->{ATTENTION_SIZE}")
    }
}

impl LayerNetwork {
    pub fn randomize(&mut self) {
        let mut rng = Rng::default();

        randomize_sparse(&mut self.feature, &mut rng);
        randomize_dense(&mut self.output, &mut rng);
    }
}

#[repr(C)]
#[derive(Copy, Clone, Zeroable)]
pub struct QuantizedLayerNetwork<const N: usize> {
    feature_weights: [QuantizedFeatureWeights; N],
    feature_bias: [QuantizedFeatureBias; N],
    output_weights: [QuantizedOutputWeights; N],
    output_bias: [QuantizedOutputBias; N],
}

unsafe impl<const N: usize> Pod for QuantizedLayerNetwork<N> {}

impl<const N: usize> QuantizedLayerNetwork<N> {
    #[must_use]
    pub fn boxed_from(subnets: &[LayerNetwork; N]) -> Box<Self> {
        let mut feature_weights: Box<[RawFeatureWeights; N]> = allocation::zeroed_box();
        let mut feature_bias: Box<[RawFeatureBias; N]> = allocation::zeroed_box();
        let mut output_weights: Box<[RawOutputWeights; N]> = allocation::zeroed_box();
        let mut output_bias: Box<[RawOutputBias; N]> = allocation::zeroed_box();

        for (subnet, raw) in subnets.iter().zip(feature_weights.iter_mut()) {
            for (row_idx, weights) in raw.iter_mut().enumerate() {
                let row = subnet.feature.weights_row(row_idx);
                for weight_idx in 0..HIDDEN_SIZE {
                    weights[weight_idx] = q_i16(row[weight_idx], QA);
                }
            }
        }

        for (subnet, raw) in subnets.iter().zip(feature_bias.iter_mut()) {
            for (weight_idx, bias) in raw.iter_mut().enumerate() {
                *bias = q_i16(subnet.feature.bias()[weight_idx], QA);
            }
        }

        for (subnet, raw) in subnets.iter().zip(output_weights.iter_mut()) {
            for a in 0..ATTENTION_SIZE {
                let col = subnet.output.weights_col(a);
                for h in 0..HIDDEN_SIZE {
                    raw[h][a] = q_i16(col[h], QB);
                }
            }
        }

        for (subnet, raw) in subnets.iter().zip(output_bias.iter_mut()) {
            for (weight_idx, bias) in raw.iter_mut().enumerate() {
                *bias = q_i32(subnet.output.bias()[weight_idx], QAB);
            }
        }

        let mut result: Box<Self> = allocation::zeroed_box();

        result.feature_weights = *bytemuck::must_cast_ref(&*feature_weights);
        result.feature_bias = *bytemuck::must_cast_ref(&*feature_bias);
        result.output_weights = *bytemuck::must_cast_ref(&*output_weights);
        result.output_bias = *bytemuck::must_cast_ref(&*output_bias);

        result
    }

    pub fn get_bias(&self, idx: usize) -> Accumulator<i16, HIDDEN_SIZE> {
        unsafe { **self.feature_bias.get_unchecked(idx) }
    }

    pub fn set(&self, idx: usize, feat_idx: usize, acc: &mut Accumulator<i16, HIDDEN_SIZE>) {
        let weights = unsafe {
            **self
                .feature_weights
                .get_unchecked(idx)
                .get_unchecked(feat_idx)
        };
        acc.set(&weights);
    }

    pub fn out(
        &self,
        idx: usize,
        acc: &Accumulator<i16, HIDDEN_SIZE>,
    ) -> Accumulator<i32, ATTENTION_SIZE> {
        assert!(idx < N);

        let mut outs = *self.output_bias[idx];
        let weights = self.output_weights[idx];

        for (out, weight) in outs.vals.iter_mut().zip(weights.iter()) {
            for (a, b) in acc.vals.iter().zip(weight.vals) {
                *out += relu(*a) * i32::from(b);
            }
        }

        outs
    }
}

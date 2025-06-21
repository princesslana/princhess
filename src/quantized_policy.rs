use arrayvec::ArrayVec;
use bytemuck::{allocation, Pod, Zeroable};
use std::path::Path;

use crate::mem::Align16;
use crate::nets::{save_to_bin, Accumulator, MoveIndex};
use crate::state::{self, State};

pub const INPUT_SIZE: usize = state::POLICY_NUMBER_FEATURES;
pub const ATTENTION_SIZE: usize = 8;

pub const QA: i32 = 256;
pub const QAA: i32 = QA * QA;

pub type RawPolicySqWeights = [RawLinearWeights; MoveIndex::SQ_COUNT];
pub type RawPolicySqBias = [RawLinearBias; MoveIndex::SQ_COUNT];
pub type RawPolicyPieceSqWeights = [RawLinearWeights; MoveIndex::TO_PIECE_SQ_COUNT];
pub type RawPolicyPieceSqBias = [RawLinearBias; MoveIndex::TO_PIECE_SQ_COUNT];

type RawLinearWeights = Align16<[[i16; ATTENTION_SIZE]; INPUT_SIZE]>;
type RawLinearBias = Align16<[i16; ATTENTION_SIZE]>;

type QuantizedLinearWeights = [Align16<Accumulator<i16, ATTENTION_SIZE>>; INPUT_SIZE];
type QuantizedLinearBias = Align16<Accumulator<i16, ATTENTION_SIZE>>;

// Max number of features is 1 per piece
type FeatureVector = ArrayVec<usize, 32>;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct QuantizedPolicyNetwork {
    sq: QuantizedLinearNetwork<{ MoveIndex::SQ_COUNT }>,
    piece_sq: QuantizedLinearNetwork<{ MoveIndex::TO_PIECE_SQ_COUNT }>,
}

impl QuantizedPolicyNetwork {
    #[must_use]
    pub fn zeroed() -> Box<Self> {
        allocation::zeroed_box()
    }

    #[must_use]
    pub fn boxed_from_slices(
        sq_weights: &RawPolicySqWeights,
        sq_bias: &RawPolicySqBias,
        piece_sq_weights: &RawPolicyPieceSqWeights,
        piece_sq_bias: &RawPolicyPieceSqBias,
    ) -> Box<Self> {
        let mut result: Box<QuantizedPolicyNetwork> = allocation::zeroed_box();

        result.sq = *QuantizedLinearNetwork::boxed_from_slices(sq_weights, sq_bias);
        result.piece_sq =
            *QuantizedLinearNetwork::boxed_from_slices(piece_sq_weights, piece_sq_bias);
        result
    }

    pub fn save_to_bin(&self, dir: &Path, name: &str) {
        save_to_bin(dir, name, self);
    }

    pub fn get_all<I: Iterator<Item = MoveIndex>>(
        &self,
        state: &State,
        move_idxes: I,
        out: &mut [f32],
    ) {
        let mut features = FeatureVector::new();

        state.policy_features_map(|feature| {
            features.push(feature);
        });

        for (i, move_idx) in move_idxes.enumerate() {
            let from_sq_idx = move_idx.from_sq().index();
            let to_sq_idx = move_idx.to_sq().index();

            let from_piece_sq_idx = move_idx.from_piece_sq_index();
            let to_piece_sq_idx = move_idx.to_piece_sq_index();

            let mut from_sq = self.sq.get_bias(from_sq_idx);
            let mut to_sq = self.sq.get_bias(to_sq_idx);

            let mut from_piece_sq = self.piece_sq.get_bias(from_piece_sq_idx);
            let mut to_piece_sq = self.piece_sq.get_bias(to_piece_sq_idx);

            for f in &features {
                self.sq.set(from_sq_idx, *f, &mut from_sq);
                self.sq.set(to_sq_idx, *f, &mut to_sq);

                self.piece_sq.set(from_piece_sq_idx, *f, &mut from_piece_sq);
                self.piece_sq.set(to_piece_sq_idx, *f, &mut to_piece_sq);
            }

            out[i] = to_sq.dot_relu::<QAA>(&to_piece_sq) - from_sq.dot_relu::<QAA>(&from_piece_sq);
        }
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
    pub fn boxed_from_slices(
        weights: &[RawLinearWeights; N],
        bias: &[RawLinearBias; N],
    ) -> Box<Self> {
        let mut result: Box<Self> = allocation::zeroed_box();

        result.weights = *bytemuck::must_cast_ref(weights);
        result.bias = *bytemuck::must_cast_ref(bias);

        result
    }

    #[must_use]
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

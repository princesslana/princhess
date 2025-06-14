use bytemuck::{allocation, Pod, Zeroable};
use std::path::Path;

use crate::nets::{save_to_bin, MoveIndex};
use crate::subnets::{LinearNetwork, QuantizedLinearNetwork, RawLinearBias, RawLinearWeights};
use goober::SparseVector;

pub type RawPolicySqWeights = [RawLinearWeights; MoveIndex::SQ_COUNT];
pub type RawPolicySqBias = [RawLinearBias; MoveIndex::SQ_COUNT];
pub type RawPolicyPieceSqWeights = [RawLinearWeights; MoveIndex::TO_PIECE_SQ_COUNT];
pub type RawPolicyPieceSqBias = [RawLinearBias; MoveIndex::TO_PIECE_SQ_COUNT];

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
        sq: &[LinearNetwork; MoveIndex::SQ_COUNT],
        piece_sq: &[LinearNetwork; MoveIndex::TO_PIECE_SQ_COUNT],
    ) -> Box<Self> {
        let mut result: Box<QuantizedPolicyNetwork> = allocation::zeroed_box();

        result.sq = *QuantizedLinearNetwork::boxed_from(sq);
        result.piece_sq = *QuantizedLinearNetwork::boxed_from(piece_sq);

        result
    }

    pub fn save_to_bin(&self, dir: &Path) {
        save_to_bin(dir, "policy.bin", self);
    }

    pub fn get_all<I: Iterator<Item = MoveIndex>>(
        &self,
        features: &SparseVector,
        move_idxes: I,
        out: &mut [f32],
    ) {
        for (i, move_idx) in move_idxes.enumerate() {
            let from_sq_idx = move_idx.from_sq().index();
            let to_sq_idx = move_idx.to_sq().index();

            let from_piece_sq_idx = move_idx.from_piece_sq_index();
            let to_piece_sq_idx = move_idx.to_piece_sq_index();

            let mut from_sq = self.sq.get_bias(from_sq_idx);
            let mut to_sq = self.sq.get_bias(to_sq_idx);

            let mut from_piece_sq = self.piece_sq.get_bias(from_piece_sq_idx);
            let mut to_piece_sq = self.piece_sq.get_bias(to_piece_sq_idx);

            for f in features.iter() {
                self.sq.set(from_sq_idx, *f, &mut from_sq);
                self.sq.set(to_sq_idx, *f, &mut to_sq);

                self.piece_sq.set(from_piece_sq_idx, *f, &mut from_piece_sq);
                self.piece_sq.set(to_piece_sq_idx, *f, &mut to_piece_sq);
            }

            out[i] = to_sq.dot_relu(&to_piece_sq) - from_sq.dot_relu(&from_piece_sq);
        }
    }
}

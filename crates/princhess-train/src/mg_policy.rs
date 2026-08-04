use std::fmt::{self, Display};
use std::ops::{AddAssign, DivAssign};
use std::ptr;

use bytemuck::{allocation, Zeroable};
use princhess::chess::Square;
use princhess::nets::MoveIndex;
use princhess::quantized_mg_policy::{
    QuantizedMgPolicyNetwork, RawPolicyPieceSqBias, RawPolicyPieceSqWeights, RawPolicySqBias,
    RawPolicySqWeights, ATTENTION_SIZE, INPUT_SIZE, QA,
};
use princhess::state::State;

use crate::nets;
use crate::neural::{
    AdamWOptimizer, FeedForwardNetwork, LRScheduler, LinearNetwork, OutputLayer, SparseVector,
};

type MgLinearNetwork = LinearNetwork<INPUT_SIZE, ATTENTION_SIZE>;

#[must_use]
pub fn is_training_position(state: &State) -> bool {
    let board = state.board();
    let major_pieces_count =
        (board.queens() | board.rooks() | board.bishops() | board.knights()).count();
    major_pieces_count > 6
}

#[allow(clippy::module_name_repetitions)]
#[derive(Zeroable)]
pub struct MgPolicyNetwork {
    sq: [MgLinearNetwork; MoveIndex::SQ_COUNT],
    piece_sq: [MgLinearNetwork; MoveIndex::TO_PIECE_SQ_COUNT],
}

impl Display for MgPolicyNetwork {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let sq = format!("sq: [{}; {}]", self.sq[0], Square::COUNT);
        let piece_sq = format!("piece_sq: [{}; {}]", self.piece_sq[0], self.piece_sq.len());
        write!(f, "{sq} * {piece_sq}")
    }
}

impl AddAssign<&Self> for MgPolicyNetwork {
    fn add_assign(&mut self, rhs: &Self) {
        for (lhs_subnet, rhs_subnet) in self.sq.iter_mut().zip(&rhs.sq) {
            *lhs_subnet += rhs_subnet;
        }

        for (lhs_subnet, rhs_subnet) in self.piece_sq.iter_mut().zip(&rhs.piece_sq) {
            *lhs_subnet += rhs_subnet;
        }
    }
}

impl DivAssign<f32> for MgPolicyNetwork {
    fn div_assign(&mut self, rhs: f32) {
        for subnet in &mut self.sq {
            *subnet /= rhs;
        }
        for subnet in &mut self.piece_sq {
            *subnet /= rhs;
        }
    }
}

impl MgPolicyNetwork {
    #[must_use]
    pub fn zeroed() -> Box<Self> {
        allocation::zeroed_box()
    }

    pub fn zero_out(&mut self) {
        // SAFETY: MgPolicyNetwork: Zeroable guarantees all-zeros is a valid bit pattern
        unsafe { ptr::write_bytes(ptr::from_mut::<Self>(self), 0, 1) }
    }

    #[must_use]
    pub fn random() -> Box<Self> {
        let mut network = Self::zeroed();

        for subnet in &mut network.sq {
            subnet.randomize();
        }

        for subnet in &mut network.piece_sq {
            subnet.randomize();
        }

        network
    }

    fn get_sq(&self, sq: Square) -> &MgLinearNetwork {
        &self.sq[sq]
    }

    fn get_piece_sq(&self, piece_sq_idx: usize) -> &MgLinearNetwork {
        unsafe { self.piece_sq.get_unchecked(piece_sq_idx) }
    }

    pub fn get_all<I: Iterator<Item = MoveIndex>>(
        &self,
        features: &SparseVector,
        move_idxes: I,
        out: &mut [f32],
    ) {
        for (i, move_idx) in move_idxes.enumerate() {
            let from_piece_sq = move_idx.from_piece_sq_index();

            let to_piece_sq = move_idx.to_piece_sq_index();

            let from_sq_logits = self.get_sq(move_idx.from_sq()).out(features);
            let to_sq_logits = self.get_sq(move_idx.to_sq()).out(features);

            let from_piece_sq_logits = self.get_piece_sq(from_piece_sq).out(features);
            let to_piece_sq_logits = self.get_piece_sq(to_piece_sq).out(features);

            out[i] =
                to_sq_logits.dot(&to_piece_sq_logits) - from_sq_logits.dot(&from_piece_sq_logits);
        }
    }

    pub fn adamw<S: LRScheduler>(
        &mut self,
        g: &Self,
        m: &mut Self,
        v: &mut Self,
        optimizer: &AdamWOptimizer<S>,
    ) {
        for subnet_idx in 0..self.sq.len() {
            self.sq[subnet_idx].adamw(
                &g.sq[subnet_idx],
                &mut m.sq[subnet_idx],
                &mut v.sq[subnet_idx],
                optimizer,
            );
        }

        for subnet_idx in 0..self.piece_sq.len() {
            self.piece_sq[subnet_idx].adamw(
                &g.piece_sq[subnet_idx],
                &mut m.piece_sq[subnet_idx],
                &mut v.piece_sq[subnet_idx],
                optimizer,
            );
        }
    }

    pub fn backprop(&self, features: &SparseVector, g: &mut Self, move_idx: MoveIndex, error: f32) {
        let from_sq = self.get_sq(move_idx.from_sq());
        let from_piece_sq = self.get_piece_sq(move_idx.from_piece_sq_index());

        let to_sq = self.get_sq(move_idx.to_sq());
        let to_piece_sq = self.get_piece_sq(move_idx.to_piece_sq_index());

        let from_sq_out = from_sq.out_with_layers(features);
        let from_piece_sq_out = from_piece_sq.out_with_layers(features);
        let to_sq_out = to_sq.out_with_layers(features);
        let to_piece_sq_out = to_piece_sq.out_with_layers(features);

        from_sq.backprop(
            features,
            &mut g.sq[move_idx.from_sq()],
            -error * from_piece_sq_out.output_layer(),
            &from_sq_out,
        );

        from_piece_sq.backprop(
            features,
            &mut g.piece_sq[move_idx.from_piece_sq_index()],
            -error * from_sq_out.output_layer(),
            &from_piece_sq_out,
        );

        to_sq.backprop(
            features,
            &mut g.sq[move_idx.to_sq()],
            error * to_piece_sq_out.output_layer(),
            &to_sq_out,
        );

        to_piece_sq.backprop(
            features,
            &mut g.piece_sq[move_idx.to_piece_sq_index()],
            error * to_sq_out.output_layer(),
            &to_piece_sq_out,
        );
    }

    #[must_use]
    pub fn to_boxed_and_quantized(&self) -> Box<QuantizedMgPolicyNetwork> {
        let mut sq_weights: Box<RawPolicySqWeights> = allocation::zeroed_box();
        let mut sq_bias: Box<RawPolicySqBias> = allocation::zeroed_box();
        let mut piece_sq_weights: Box<RawPolicyPieceSqWeights> = allocation::zeroed_box();
        let mut piece_sq_bias: Box<RawPolicyPieceSqBias> = allocation::zeroed_box();

        for (subnet, raw) in self.sq.iter().zip(sq_weights.iter_mut()) {
            for (row_idx, weights) in raw.iter_mut().enumerate() {
                let row = subnet.output.weights_row(row_idx);
                for weight_idx in 0..ATTENTION_SIZE {
                    weights[weight_idx] = nets::q_i16(row[weight_idx], QA);
                }
            }
        }

        for (subnet, raw) in self.sq.iter().zip(sq_bias.iter_mut()) {
            for (weight_idx, bias) in raw.iter_mut().enumerate() {
                *bias = nets::q_i16(subnet.output.bias()[weight_idx], QA);
            }
        }

        for (subnet, raw) in self.piece_sq.iter().zip(piece_sq_weights.iter_mut()) {
            for (row_idx, weights) in raw.iter_mut().enumerate() {
                let row = subnet.output.weights_row(row_idx);
                for weight_idx in 0..ATTENTION_SIZE {
                    weights[weight_idx] = nets::q_i16(row[weight_idx], QA);
                }
            }
        }

        for (subnet, raw) in self.piece_sq.iter().zip(piece_sq_bias.iter_mut()) {
            for (weight_idx, bias) in raw.iter_mut().enumerate() {
                *bias = nets::q_i16(subnet.output.bias()[weight_idx], QA);
            }
        }

        QuantizedMgPolicyNetwork::boxed_from_slices(
            &sq_weights,
            &sq_bias,
            &piece_sq_weights,
            &piece_sq_bias,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantization_does_not_crash() {
        let policy_net = MgPolicyNetwork::random();
        let _quantized_policy_net = policy_net.to_boxed_and_quantized();
    }
}

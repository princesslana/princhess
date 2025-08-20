use crate::neural::{
    AdamWOptimizer, FeedForwardNetwork, LRScheduler, OutputLayer, ReLU, SparseConnected,
    SparseVector,
};
use bytemuck::{allocation, Zeroable};
use princhess::chess::Square;
use princhess::math::Rng;
use princhess::nets::MoveIndex;
use princhess::quantized_policy::{
    QuantizedPolicyNetwork, RawPolicyPieceSqBias, RawPolicyPieceSqWeights, RawPolicySqBias,
    RawPolicySqWeights, ATTENTION_SIZE, INPUT_SIZE, QA,
};
use princhess::state::State;
use std::fmt::{self, Display};
use std::ops::AddAssign;

use crate::nets::q_i16;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Phase {
    MiddleGame,
    Endgame,
}

impl Phase {
    pub fn from_arg(arg: &str) -> Option<Self> {
        if arg.eq_ignore_ascii_case("mg") {
            Some(Self::MiddleGame)
        } else if arg.eq_ignore_ascii_case("eg") {
            Some(Self::Endgame)
        } else {
            None
        }
    }

    pub fn matches(&self, state: &State) -> bool {
        let board = state.board();
        let major_pieces_count =
            (board.queens() | board.rooks() | board.bishops() | board.knights()).count();

        match self {
            Self::MiddleGame => major_pieces_count > 6,
            Self::Endgame => major_pieces_count <= 8,
        }
    }
}

impl Display for Phase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MiddleGame => write!(f, "mg"),
            Self::Endgame => write!(f, "eg"),
        }
    }
}

#[allow(clippy::module_name_repetitions)]
#[derive(Zeroable)]
pub struct PolicyNetwork {
    sq: [LinearNetwork; MoveIndex::SQ_COUNT],
    piece_sq: [LinearNetwork; MoveIndex::TO_PIECE_SQ_COUNT],
}

#[allow(clippy::module_name_repetitions)]
#[derive(Zeroable)]
pub struct PolicyCount {
    pub sq: [u64; MoveIndex::SQ_COUNT],
    pub piece_sq: [u64; MoveIndex::TO_PIECE_SQ_COUNT],
}

impl Display for PolicyNetwork {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let sq = format!("sq: [{}; {}]", self.sq[0], Square::COUNT);
        let piece_sq = format!("piece_sq: [{}; {}]", self.piece_sq[0], self.piece_sq.len(),);
        write!(f, "{sq} * {piece_sq}")
    }
}

impl AddAssign<&Self> for PolicyNetwork {
    fn add_assign(&mut self, rhs: &Self) {
        for (lhs_subnet, rhs_subnet) in self.sq.iter_mut().zip(&rhs.sq) {
            *lhs_subnet += rhs_subnet;
        }

        for (lhs_subnet, rhs_subnet) in self.piece_sq.iter_mut().zip(&rhs.piece_sq) {
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

        for subnet in &mut network.sq {
            subnet.randomize();
        }

        for subnet in &mut network.piece_sq {
            subnet.randomize();
        }

        network
    }

    fn get_sq(&self, sq: Square) -> &LinearNetwork {
        &self.sq[sq]
    }

    fn get_piece_sq(&self, piece_sq_idx: usize) -> &LinearNetwork {
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

    pub fn scale_by_counts(&mut self, count: &PolicyCount) {
        for subnet_idx in 0..self.sq.len() {
            if count.sq[subnet_idx] > 0 {
                self.sq[subnet_idx] /= count.sq[subnet_idx] as f32;
            }
        }
        for subnet_idx in 0..self.piece_sq.len() {
            if count.piece_sq[subnet_idx] > 0 {
                self.piece_sq[subnet_idx] /= count.piece_sq[subnet_idx] as f32;
            }
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
    pub fn to_boxed_and_quantized(&self) -> Box<QuantizedPolicyNetwork> {
        let mut sq_weights: Box<RawPolicySqWeights> = allocation::zeroed_box();
        let mut sq_bias: Box<RawPolicySqBias> = allocation::zeroed_box();
        let mut piece_sq_weights: Box<RawPolicyPieceSqWeights> = allocation::zeroed_box();
        let mut piece_sq_bias: Box<RawPolicyPieceSqBias> = allocation::zeroed_box();

        for (subnet, raw) in self.sq.iter().zip(sq_weights.iter_mut()) {
            for (row_idx, weights) in raw.iter_mut().enumerate() {
                let row = subnet.output.weights_row(row_idx);
                for weight_idx in 0..ATTENTION_SIZE {
                    weights[weight_idx] = q_i16(row[weight_idx], QA);
                }
            }
        }

        for (subnet, raw) in self.sq.iter().zip(sq_bias.iter_mut()) {
            for (weight_idx, bias) in raw.iter_mut().enumerate() {
                *bias = q_i16(subnet.output.bias()[weight_idx], QA);
            }
        }

        for (subnet, raw) in self.piece_sq.iter().zip(piece_sq_weights.iter_mut()) {
            for (row_idx, weights) in raw.iter_mut().enumerate() {
                let row = subnet.output.weights_row(row_idx);
                for weight_idx in 0..ATTENTION_SIZE {
                    weights[weight_idx] = q_i16(row[weight_idx], QA);
                }
            }
        }

        for (subnet, raw) in self.piece_sq.iter().zip(piece_sq_bias.iter_mut()) {
            for (weight_idx, bias) in raw.iter_mut().enumerate() {
                *bias = q_i16(subnet.output.bias()[weight_idx], QA);
            }
        }

        QuantizedPolicyNetwork::boxed_from_slices(
            &sq_weights,
            &sq_bias,
            &piece_sq_weights,
            &piece_sq_bias,
        )
    }
}

impl PolicyCount {
    pub fn increment(&mut self, move_idx: MoveIndex) {
        self.sq[move_idx.from_sq()] += 1;
        self.sq[move_idx.to_sq()] += 1;
        self.piece_sq[move_idx.from_piece_sq_index()] += 1;
        self.piece_sq[move_idx.to_piece_sq_index()] += 1;
    }
}

impl AddAssign<&Self> for PolicyCount {
    fn add_assign(&mut self, rhs: &Self) {
        for (lhs, rhs) in self.sq.iter_mut().zip(&rhs.sq) {
            *lhs += rhs;
        }

        for (lhs, rhs) in self.piece_sq.iter_mut().zip(&rhs.piece_sq) {
            *lhs += rhs;
        }
    }
}

type Linear = SparseConnected<ReLU, INPUT_SIZE, ATTENTION_SIZE>;

#[repr(C)]
pub struct LinearNetwork {
    output: Linear,
}

unsafe impl Zeroable for LinearNetwork {}

impl std::ops::AddAssign<&LinearNetwork> for LinearNetwork {
    fn add_assign(&mut self, rhs: &LinearNetwork) {
        self.output += &rhs.output;
    }
}

impl std::ops::DivAssign<f32> for LinearNetwork {
    fn div_assign(&mut self, rhs: f32) {
        self.output /= rhs;
    }
}

impl LinearNetwork {
    pub fn randomize(&mut self) {
        let mut rng = Rng::default();

        self.output = *SparseConnected::randomized(&mut rng);
    }

    pub fn adamw<S: LRScheduler>(
        &mut self,
        g: &Self,
        m: &mut Self,
        v: &mut Self,
        optimizer: &AdamWOptimizer<S>,
    ) {
        self.output
            .adamw(&g.output, &mut m.output, &mut v.output, optimizer);
    }
}

impl Display for LinearNetwork {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{INPUT_SIZE}->{ATTENTION_SIZE}")
    }
}

impl FeedForwardNetwork for LinearNetwork {
    type InputType = SparseVector;
    type OutputType = crate::neural::Vector<ATTENTION_SIZE>;
    type Layers = <Linear as FeedForwardNetwork>::Layers;

    fn out_with_layers(&self, input: &Self::InputType) -> Self::Layers {
        self.output.out_with_layers(input)
    }

    fn backprop(
        &self,
        input: &Self::InputType,
        grad: &mut Self,
        out_err: Self::OutputType,
        layers: &Self::Layers,
    ) -> Self::InputType {
        self.output
            .backprop(input, &mut grad.output, out_err, layers)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantization_does_not_crash() {
        // This test ensures that the conversion to a quantized policy network does not crash.
        let policy_net = PolicyNetwork::random();
        let _quantized_policy_net = policy_net.to_boxed_and_quantized();
    }
}

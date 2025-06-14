use bytemuck::{allocation, Zeroable};
use goober::{FeedForwardNetwork, OutputLayer, SparseVector};
use std::fmt::{self, Display};
use std::ops::AddAssign;

use princhess::chess::Square;
use princhess::nets::MoveIndex;
use princhess::quantized_policy::QuantizedPolicyNetwork;
use princhess::subnets::LinearNetwork;

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

    pub fn adam(&mut self, g: &Self, m: &mut Self, v: &mut Self, count: &PolicyCount, lr: f32) {
        for subnet_idx in 0..self.sq.len() {
            match count.sq[subnet_idx] {
                0 => (),
                n => self.sq[subnet_idx].adam(
                    &g.sq[subnet_idx],
                    &mut m.sq[subnet_idx],
                    &mut v.sq[subnet_idx],
                    1.0 / n as f32,
                    lr,
                ),
            }
        }

        for subnet_idx in 0..self.piece_sq.len() {
            match count.piece_sq[subnet_idx] {
                0 => (),
                n => self.piece_sq[subnet_idx].adam(
                    &g.piece_sq[subnet_idx],
                    &mut m.piece_sq[subnet_idx],
                    &mut v.piece_sq[subnet_idx],
                    1.0 / n as f32,
                    lr,
                ),
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
        QuantizedPolicyNetwork::boxed_from_slices(&self.sq, &self.piece_sq)
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

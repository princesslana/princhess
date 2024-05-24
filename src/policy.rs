use goober::activation::ReLU;
use goober::layer::{DenseConnected, SparseConnected};
use goober::{FeedForwardNetwork, OutputLayer, SparseVector};
use std::fmt::{self, Display};
use std::fs;
use std::io::Write;
use std::mem;
use std::ops::AddAssign;
use std::path::Path;
use std::slice;

use crate::chess::MoveIndex;
use crate::math::{randomize_dense, randomize_sparse, Rng};
use crate::mem::boxed_and_zeroed;
use crate::state;

const INPUT_SIZE: usize = state::POLICY_NUMBER_FEATURES;
const HIDDEN_SIZE: usize = 16;
const ATTENTION_SIZE: usize = 16;

#[repr(C)]
#[derive(FeedForwardNetwork)]
pub struct SubNetwork {
    hidden: SparseConnected<ReLU, INPUT_SIZE, HIDDEN_SIZE>,
    output: DenseConnected<ReLU, HIDDEN_SIZE, ATTENTION_SIZE>,
}

impl SubNetwork {
    pub fn randomize(&mut self) {
        let mut rng = Rng::default();

        randomize_sparse(&mut self.hidden, &mut rng);
        randomize_dense(&mut self.output, &mut rng);
    }
}

#[allow(clippy::module_name_repetitions)]
#[repr(C)]
pub struct PolicyNetwork {
    from: [SubNetwork; 64],
    to: [SubNetwork; 64],
}

impl Display for PolicyNetwork {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let from = format!("from: [{INPUT_SIZE}->{HIDDEN_SIZE}->{ATTENTION_SIZE}; 64]");
        let to = format!("to: [{INPUT_SIZE}->{HIDDEN_SIZE}->{ATTENTION_SIZE}; 64]");
        write!(f, "{from} * {to}")
    }
}

impl AddAssign<&Self> for PolicyNetwork {
    fn add_assign(&mut self, rhs: &Self) {
        for (lhs_subnet, rhs_subnet) in self.from.iter_mut().zip(&rhs.from) {
            lhs_subnet.hidden += &rhs_subnet.hidden;
            lhs_subnet.output += &rhs_subnet.output;
        }

        for (lhs_subnet, rhs_subnet) in self.to.iter_mut().zip(&rhs.to) {
            lhs_subnet.hidden += &rhs_subnet.hidden;
            lhs_subnet.output += &rhs_subnet.output;
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

    #[must_use]
    pub fn get(&self, features: &SparseVector, move_idx: MoveIndex) -> f32 {
        let from = &self.from[move_idx.from_index()].out(features);
        let to = &self.to[move_idx.to_index()].out(features);

        from.dot(to)
    }

    pub fn get_all<I: Iterator<Item=MoveIndex>>(
        &self,
        features: &SparseVector,
        move_idxes: I,
        out: &mut Vec<f32>,
    ) {
        let mut from_logits = [None; 64];
        let mut to_logits = [None; 64];

        for move_idx in move_idxes {
            let from_idx = move_idx.from_index();
            let to_idx = move_idx.to_index();

            let from =
                from_logits[from_idx].get_or_insert_with(|| self.from[from_idx].out(features));
            let to = to_logits[to_idx].get_or_insert_with(|| self.to[to_idx].out(features));

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

    pub fn save_to_bin(&self, dir: &Path) {
        let mut file = fs::File::create(dir.join("policy.bin")).expect("Failed to create file");

        let size_of = mem::size_of::<Self>();

        unsafe {
            let ptr: *const Self = self;
            let slice_ptr: *const u8 = ptr.cast::<u8>();
            let slice = slice::from_raw_parts(slice_ptr, size_of);
            file.write_all(slice).unwrap();
        }
    }
}

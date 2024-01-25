#![allow(clippy::module_name_repetitions)]

use goober::activation::{Identity, ReLU};
use goober::layer::SparseConnected;
use goober::{FeedForwardNetwork, OutputLayer, SparseVector, Vector};
use std::fs;
use std::ops::AddAssign;
use std::path::Path;

use crate::math::Rng;
use crate::state;
use crate::train::{boxed_and_zeroed, randomize, write_layer};

const INPUT_SIZE: usize = state::POLICY_NUMBER_FEATURES;
const OUTPUT_SIZE: usize = 384;

const QA: f32 = 256.;

type ConstantLayer = SparseConnected<Identity, INPUT_SIZE, OUTPUT_SIZE>;
type LeftLayer = SparseConnected<Identity, INPUT_SIZE, OUTPUT_SIZE>;
type RightLayer = SparseConnected<ReLU, INPUT_SIZE, OUTPUT_SIZE>;

pub struct PolicyNetwork {
    constant: ConstantLayer,
    left: LeftLayer,
    right: RightLayer,
}

impl AddAssign<&Self> for PolicyNetwork {
    fn add_assign(&mut self, rhs: &Self) {
        self.constant += &rhs.constant;
        self.left += &rhs.left;
        self.right += &rhs.right;
    }
}

pub struct PolicyNetworkLayers {
    constant: <ConstantLayer as FeedForwardNetwork>::Layers,
    left: <LeftLayer as FeedForwardNetwork>::Layers,
    right: <RightLayer as FeedForwardNetwork>::Layers,
}

impl OutputLayer<Vector<OUTPUT_SIZE>> for PolicyNetworkLayers {
    fn output_layer(&self) -> Vector<OUTPUT_SIZE> {
        self.constant.output_layer() + self.left.output_layer() * self.right.output_layer()
    }
}

impl FeedForwardNetwork for PolicyNetwork {
    type InputType = SparseVector;
    type OutputType = Vector<OUTPUT_SIZE>;
    type Layers = PolicyNetworkLayers;

    fn adam(&mut self, g: &Self, m: &mut Self, v: &mut Self, adj: f32, lr: f32) {
        self.constant
            .adam(&g.constant, &mut m.constant, &mut v.constant, adj, lr);
        self.left.adam(&g.left, &mut m.left, &mut v.left, adj, lr);
        self.right
            .adam(&g.right, &mut m.right, &mut v.right, adj, lr);
    }

    fn out_with_layers(&self, input: &Self::InputType) -> Self::Layers {
        Self::Layers {
            constant: self.constant.out_with_layers(input),
            left: self.left.out_with_layers(input),
            right: self.right.out_with_layers(input),
        }
    }

    fn backprop(
        &self,
        i: &Self::InputType,
        g: &mut Self,
        e: Self::OutputType,
        l: &Self::Layers,
    ) -> Self::InputType {
        self.constant.backprop(i, &mut g.constant, e, &l.constant);

        self.left
            .backprop(i, &mut g.left, e * l.right.output_layer(), &l.left);
        self.right
            .backprop(i, &mut g.right, e * l.left.output_layer(), &l.right);

        SparseVector::with_capacity(0)
    }
}

impl PolicyNetwork {
    #[must_use]
    pub fn zeroed() -> Box<Self> {
        boxed_and_zeroed()
    }

    #[must_use]
    pub fn random() -> Box<Self> {
        let mut rng = Rng::default();

        let mut network = Self::zeroed();

        randomize(&mut network.constant, &mut rng);
        randomize(&mut network.left, &mut rng);
        randomize(&mut network.right, &mut rng);

        network
    }

    pub fn save(&self, path: &str) {
        fs::create_dir(path).expect("Failed to create directory");

        let dir = Path::new(path);

        write_layer(dir, "constant", &self.constant, QA);
        write_layer(dir, "left", &self.left, QA);
        write_layer(dir, "right", &self.right, QA);
    }
}

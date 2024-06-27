use goober::activation::{ReLU, Tanh};
use goober::layer::{DenseConnected, SparseConnected};
use goober::{FeedForwardNetwork, OutputLayer, SparseVector, Vector};
use std::boxed::Box;
use std::fmt::{self, Display, Formatter};
use std::fs;
use std::io::Write;
use std::mem;
use std::ops::AddAssign;
use std::path::Path;
use std::slice;

use crate::math::{randomize_dense, randomize_sparse, Rng};
use crate::mem::boxed_and_zeroed;
use crate::nets::{q_i16, q_i32, relu, Accumulator};
use crate::state::{self, State};

const INPUT_SIZE: usize = state::VALUE_NUMBER_FEATURES;
const HIDDEN_SIZE: usize = 256;
const OUTPUT_SIZE: usize = 1;

const QA: i32 = 256;
const QB: i32 = 256;
const QAB: i32 = QA * QB;

type Feature = SparseConnected<ReLU, INPUT_SIZE, HIDDEN_SIZE>;
type Output = DenseConnected<Tanh, { HIDDEN_SIZE * 2 }, OUTPUT_SIZE>;

type QuantizedFeatureWeights = [Accumulator<HIDDEN_SIZE>; INPUT_SIZE];
type QuantizedFeatureBias = Accumulator<HIDDEN_SIZE>;
type QuantizedOutputWeights = [Accumulator<HIDDEN_SIZE>; 2];

type RawFeatureWeights = [[i16; HIDDEN_SIZE]; INPUT_SIZE];
type RawFeatureBias = [i16; HIDDEN_SIZE];
type RawOutputWeights = [i16; HIDDEN_SIZE * 2];

#[allow(clippy::module_name_repetitions)]
pub struct ValueNetwork {
    stm: Feature,
    nstm: Feature,
    output: Output,
}

#[repr(C)]
pub struct QuantizedValueNetwork {
    stm_weights: QuantizedFeatureWeights,
    stm_bias: QuantizedFeatureBias,
    nstm_weights: QuantizedFeatureWeights,
    nstm_bias: QuantizedFeatureBias,
    output_weights: QuantizedOutputWeights,
    output_bias: i32,
}

impl Display for ValueNetwork {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "({INPUT_SIZE}->{HIDDEN_SIZE})*2->{OUTPUT_SIZE}")
    }
}

impl AddAssign<&Self> for ValueNetwork {
    fn add_assign(&mut self, rhs: &Self) {
        self.stm += &rhs.stm;
        self.nstm += &rhs.nstm;
        self.output += &rhs.output;
    }
}

impl ValueNetwork {
    #[must_use]
    pub fn zeroed() -> Box<Self> {
        boxed_and_zeroed()
    }

    #[must_use]
    pub fn random() -> Box<Self> {
        let mut rng = Rng::default();

        let mut network = Self::zeroed();

        randomize_sparse(&mut network.stm, &mut rng);
        randomize_sparse(&mut network.nstm, &mut rng);
        randomize_dense(&mut network.output, &mut rng);

        network
    }

    pub fn decay_weights(&mut self, decay: f32) {
        for row_idx in 0..INPUT_SIZE {
            let stm_row = self.stm.weights_row_mut(row_idx);
            let nstm_row = self.nstm.weights_row_mut(row_idx);

            for weight_idx in 0..HIDDEN_SIZE {
                stm_row[weight_idx] *= decay;
                nstm_row[weight_idx] *= decay;
            }
        }

        for weight_idx in 0..HIDDEN_SIZE {
            self.stm.bias_mut()[weight_idx] *= decay;
            self.nstm.bias_mut()[weight_idx] *= decay;
        }

        for col_idx in 0..HIDDEN_SIZE {
            let col = self.output.weights_col_mut(col_idx);
            for weight_idx in 0..OUTPUT_SIZE {
                col[weight_idx] *= decay;
            }
        }

        for weight_idx in 0..OUTPUT_SIZE {
            self.output.bias_mut()[weight_idx] *= decay;
        }
    }

    #[must_use]
    pub fn to_boxed_and_quantized(&self) -> Box<QuantizedValueNetwork> {
        let mut stm_weights: Box<RawFeatureWeights> = boxed_and_zeroed();
        let mut stm_bias = [0; HIDDEN_SIZE];
        let mut nstm_weights: Box<RawFeatureWeights> = boxed_and_zeroed();
        let mut nstm_bias = [0; HIDDEN_SIZE];
        let mut output_weights: Box<RawOutputWeights> = boxed_and_zeroed();

        for (row_idx, weights) in stm_weights.iter_mut().enumerate() {
            let row = self.stm.weights_row(row_idx);
            for weight_idx in 0..HIDDEN_SIZE {
                weights[weight_idx] = q_i16(row[weight_idx], QA);
            }
        }

        for (weight_idx, bias) in stm_bias.iter_mut().enumerate() {
            *bias = q_i16(self.stm.bias()[weight_idx], QA);
        }

        for (row_idx, weights) in nstm_weights.iter_mut().enumerate() {
            let row = self.nstm.weights_row(row_idx);
            for weight_idx in 0..HIDDEN_SIZE {
                weights[weight_idx] = q_i16(row[weight_idx], QA);
            }
        }

        for (weight_idx, bias) in nstm_bias.iter_mut().enumerate() {
            *bias = q_i16(self.nstm.bias()[weight_idx], QA);
        }

        for (col_idx, weights) in output_weights.iter_mut().enumerate() {
            let col = self.output.weights_col(col_idx);
            *weights = q_i16(col[0], QB);
        }

        let output_bias = q_i32(self.output.bias()[0], QAB);

        QuantizedValueNetwork::from_slices(
            &stm_weights,
            &stm_bias,
            &nstm_weights,
            &nstm_bias,
            &output_weights,
            output_bias,
        )
    }
}

#[allow(clippy::module_name_repetitions)]
pub struct ValueNetworkLayers {
    stm: <Feature as FeedForwardNetwork>::Layers,
    nstm: <Feature as FeedForwardNetwork>::Layers,
    output: <Output as FeedForwardNetwork>::Layers,
}

impl OutputLayer<Vector<OUTPUT_SIZE>> for ValueNetworkLayers {
    fn output_layer(&self) -> Vector<OUTPUT_SIZE> {
        self.output.output_layer()
    }
}

impl FeedForwardNetwork for ValueNetwork {
    type InputType = SparseVector;
    type OutputType = Vector<OUTPUT_SIZE>;
    type Layers = ValueNetworkLayers;

    fn adam(&mut self, g: &Self, m: &mut Self, v: &mut Self, adj: f32, lr: f32) {
        self.stm.adam(&g.stm, &mut m.stm, &mut v.stm, adj, lr);
        self.nstm.adam(&g.nstm, &mut m.nstm, &mut v.nstm, adj, lr);
        self.output
            .adam(&g.output, &mut m.output, &mut v.output, adj, lr);
    }

    fn out_with_layers(&self, input: &Self::InputType) -> Self::Layers {
        let (stm_input, nstm_input) = split_sparse(input);

        let (stm_layers, nstm_layers) = (
            self.stm.out_with_layers(&stm_input),
            self.nstm.out_with_layers(&nstm_input),
        );

        let output_layer = self.output.out_with_layers(&concat_dense(
            &stm_layers.output_layer(),
            &nstm_layers.output_layer(),
        ));

        ValueNetworkLayers {
            stm: stm_layers,
            nstm: nstm_layers,
            output: output_layer,
        }
    }

    fn backprop(
        &self,
        input: &Self::InputType,
        grad: &mut Self,
        err: Self::OutputType,
        layers: &Self::Layers,
    ) -> Self::InputType {
        let feature_errs = self.output.backprop(
            &concat_dense(&layers.stm.output_layer(), &layers.nstm.output_layer()),
            &mut grad.output,
            err,
            &layers.output,
        );

        let (stm_input, nstm_input) = split_sparse(input);
        let (stm_errs, nstm_errs) = split_dense(&feature_errs);

        self.stm
            .backprop(&stm_input, &mut grad.stm, stm_errs, &layers.stm);
        self.nstm
            .backprop(&nstm_input, &mut grad.nstm, nstm_errs, &layers.nstm);

        SparseVector::with_capacity(0)
    }
}

impl QuantizedValueNetwork {
    #[must_use]
    pub fn zeroed() -> Box<Self> {
        boxed_and_zeroed()
    }

    #[must_use]
    pub fn from_slices(
        stm_weights: &RawFeatureWeights,
        stm_bias: &RawFeatureBias,
        nstm_weights: &RawFeatureWeights,
        nstm_bias: &RawFeatureBias,
        output_weights: &RawOutputWeights,
        output_bias: i32,
    ) -> Box<Self> {
        let mut network = Self::zeroed();

        network.stm_weights = unsafe {
            std::mem::transmute::<RawFeatureWeights, QuantizedFeatureWeights>(*stm_weights)
        };
        network.stm_bias =
            unsafe { std::mem::transmute::<RawFeatureBias, QuantizedFeatureBias>(*stm_bias) };

        network.nstm_weights = unsafe {
            std::mem::transmute::<RawFeatureWeights, QuantizedFeatureWeights>(*nstm_weights)
        };
        network.nstm_bias =
            unsafe { std::mem::transmute::<RawFeatureBias, QuantizedFeatureBias>(*nstm_bias) };

        network.output_weights = unsafe {
            std::mem::transmute::<RawOutputWeights, QuantizedOutputWeights>(*output_weights)
        };
        network.output_bias = output_bias;

        network
    }

    pub fn save_to_bin(&self, dir: &Path) {
        let mut file = fs::File::create(dir.join("value.bin")).expect("Failed to create file");

        let size_of = mem::size_of::<Self>();

        unsafe {
            let ptr: *const Self = self;
            let slice_ptr: *const u8 = ptr.cast::<u8>();
            let slice = slice::from_raw_parts(slice_ptr, size_of);
            file.write_all(slice).unwrap();
        }
    }

    #[must_use]
    #[allow(clippy::cast_possible_wrap)]
    pub fn get(&self, state: &State) -> f32 {
        let mut stm = self.stm_bias;
        let mut nstm = self.nstm_bias;

        state.value_features_map(|idx| {
            if idx < INPUT_SIZE {
                stm.set(&self.stm_weights[idx]);
            } else {
                nstm.set(&self.nstm_weights[idx % INPUT_SIZE]);
            }
        });

        let mut result: i32 = self.output_bias;

        for (&x, w) in stm.vals.iter().zip(self.output_weights[0].vals) {
            result += relu(x) * i32::from(w);
        }

        for (&x, w) in nstm.vals.iter().zip(self.output_weights[1].vals) {
            result += relu(x) * i32::from(w);
        }

        result -= result * (state.fifty_move_counter() - 36).max(0) as i32 / 128;

        (result as f32 / QAB as f32).tanh()
    }
}

fn split_sparse(input: &SparseVector) -> (SparseVector, SparseVector) {
    let mut lhs = SparseVector::with_capacity(32);
    let mut rhs = SparseVector::with_capacity(32);
    for idx in input.iter() {
        if *idx < INPUT_SIZE {
            lhs.push(*idx);
        } else {
            rhs.push(*idx - INPUT_SIZE);
        }
    }
    (lhs, rhs)
}

fn concat_dense(
    lhs: &Vector<HIDDEN_SIZE>,
    rhs: &Vector<HIDDEN_SIZE>,
) -> Vector<{ HIDDEN_SIZE * 2 }> {
    let mut result = Vector::zeroed();
    for idx in 0..HIDDEN_SIZE {
        result[idx] = lhs[idx];
        result[idx + HIDDEN_SIZE] = rhs[idx];
    }
    result
}

fn split_dense(input: &Vector<{ HIDDEN_SIZE * 2 }>) -> (Vector<HIDDEN_SIZE>, Vector<HIDDEN_SIZE>) {
    let mut lhs = Vector::zeroed();
    let mut rhs = Vector::zeroed();
    for idx in 0..HIDDEN_SIZE {
        lhs[idx] = input[idx];
        rhs[idx] = input[idx + HIDDEN_SIZE];
    }
    (lhs, rhs)
}

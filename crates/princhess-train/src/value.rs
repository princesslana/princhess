use crate::neural::{
    AdamWOptimizer, DenseConnected, FeedForwardNetwork, LRScheduler, OutputLayer, SparseConnected,
    SparseVector, Tanh, Vector,
};
use bytemuck::{allocation, Zeroable};
use princhess::math::Rng;
use princhess::quantized_value::{
    QuantizedValueNetwork, RawFeatureBias, RawFeatureWeights, RawOutputWeights, HIDDEN_SIZE,
    INPUT_SIZE, QA, QAB, QB,
};
use std::boxed::Box;
use std::fmt::{self, Display, Formatter};
use std::ops::{AddAssign, DivAssign, MulAssign};

use crate::nets::{q_i16, q_i32};
use crate::neural::SCReLU;

pub const OUTPUT_SIZE: usize = 1;

type Feature = SparseConnected<SCReLU, INPUT_SIZE, HIDDEN_SIZE>;
type Output = DenseConnected<Tanh, { HIDDEN_SIZE * 2 }, OUTPUT_SIZE>;

#[allow(clippy::module_name_repetitions)]
pub struct ValueNetwork {
    stm: Feature,
    nstm: Feature,
    output: Output,
}

unsafe impl Zeroable for ValueNetwork {}

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

impl DivAssign<f32> for ValueNetwork {
    fn div_assign(&mut self, rhs: f32) {
        self.stm /= rhs;
        self.nstm /= rhs;
        self.output /= rhs;
    }
}

impl MulAssign<f32> for ValueNetwork {
    fn mul_assign(&mut self, rhs: f32) {
        self.stm *= rhs;
        self.nstm *= rhs;
        self.output *= rhs;
    }
}

impl ValueNetwork {
    #[must_use]
    pub fn zeroed() -> Box<Self> {
        allocation::zeroed_box()
    }

    /// Compute the L2 norm of all gradients
    pub fn gradient_norm(&self) -> f32 {
        let mut norm_sq: f32 = 0.0;

        // STM weights and bias
        for row in 0..INPUT_SIZE {
            let weights = self.stm.weights_row(row);
            for col in 0..HIDDEN_SIZE {
                let w = weights[col];
                norm_sq += w * w;
            }
        }
        for i in 0..HIDDEN_SIZE {
            let b = self.stm.bias()[i];
            norm_sq += b * b;
        }

        // NSTM weights and bias
        for row in 0..INPUT_SIZE {
            let weights = self.nstm.weights_row(row);
            for col in 0..HIDDEN_SIZE {
                let w = weights[col];
                norm_sq += w * w;
            }
        }
        for i in 0..HIDDEN_SIZE {
            let b = self.nstm.bias()[i];
            norm_sq += b * b;
        }

        // Output weights and bias
        for col in 0..(HIDDEN_SIZE * 2) {
            let weights = self.output.weights_col(col);
            let w = weights[0]; // OUTPUT_SIZE = 1
            norm_sq += w * w;
        }
        let b = self.output.bias()[0]; // OUTPUT_SIZE = 1
        norm_sq += b * b;

        norm_sq.sqrt()
    }

    /// Clip gradients to maximum norm
    pub fn clip_gradients(&mut self, max_norm: f32) {
        let norm = self.gradient_norm();
        if norm > max_norm {
            let scale = max_norm / norm;
            *self *= scale;
        }
    }

    /// Get weight statistics for monitoring
    pub fn output_weights_norm(&self) -> f32 {
        self.output.weights_norm()
    }

    pub fn output_bias(&self) -> f32 {
        self.output.bias()[0]
    }

    pub fn stm_weights_norm(&self) -> f32 {
        self.stm.weights_norm()
    }

    pub fn nstm_weights_norm(&self) -> f32 {
        self.nstm.weights_norm()
    }

    /// Apply optimization step with different optimizers for feature and output layers
    pub fn train_step<S: LRScheduler>(
        &mut self,
        gradients: &Self,
        momentum: &mut Self,
        velocity: &mut Self,
        feature_optimizer: &AdamWOptimizer<S>,
        output_optimizer: &AdamWOptimizer<S>,
    ) {
        self.stm.adamw(
            &gradients.stm,
            &mut momentum.stm,
            &mut velocity.stm,
            feature_optimizer,
        );
        self.nstm.adamw(
            &gradients.nstm,
            &mut momentum.nstm,
            &mut velocity.nstm,
            feature_optimizer,
        );
        self.output.adamw(
            &gradients.output,
            &mut momentum.output,
            &mut velocity.output,
            output_optimizer,
        );
    }

    #[must_use]
    pub fn random() -> Box<Self> {
        let mut rng = Rng::default();

        let mut network: Box<Self> = allocation::zeroed_box();
        network.stm = *SparseConnected::randomized(&mut rng);
        network.nstm = *SparseConnected::randomized(&mut rng);
        network.output = *DenseConnected::randomized(&mut rng);
        network
    }

    #[must_use]
    pub fn to_boxed_and_quantized(&self) -> Box<QuantizedValueNetwork> {
        let mut stm_weights: Box<RawFeatureWeights> = allocation::zeroed_box();
        let mut stm_bias: Box<RawFeatureBias> = allocation::zeroed_box();

        let mut nstm_weights: Box<RawFeatureWeights> = allocation::zeroed_box();
        let mut nstm_bias: Box<RawFeatureBias> = allocation::zeroed_box();

        let mut output_weights: Box<RawOutputWeights> = allocation::zeroed_box();

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

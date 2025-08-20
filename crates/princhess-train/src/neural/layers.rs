use std::marker::PhantomData;

use crate::neural::{
    activation::Activation, initialization::WeightInitializer, AdamWOptimizer, FeedForwardNetwork,
    LRScheduler, Matrix, OutputLayer, SparseVector, Vector,
};
use bytemuck::{allocation, Zeroable};
use princhess::math::Rng;

#[repr(C)]
#[derive(Clone, Copy, Zeroable)]
pub struct DenseConnected<T, const M: usize, const N: usize> {
    weights: Matrix<M, N>,
    bias: Vector<N>,
    phantom: PhantomData<T>,
}

impl<T: Activation, const M: usize, const N: usize> std::ops::AddAssign<&DenseConnected<T, M, N>>
    for DenseConnected<T, M, N>
{
    fn add_assign(&mut self, rhs: &DenseConnected<T, M, N>) {
        self.weights += &rhs.weights;
        self.bias += rhs.bias;
    }
}

impl<T: Activation, const M: usize, const N: usize> std::ops::DivAssign<f32>
    for DenseConnected<T, M, N>
{
    fn div_assign(&mut self, rhs: f32) {
        self.weights /= rhs;
        self.bias /= rhs;
    }
}

impl<T: Activation, const M: usize, const N: usize> std::ops::MulAssign<f32>
    for DenseConnected<T, M, N>
{
    fn mul_assign(&mut self, rhs: f32) {
        self.weights *= rhs;
        self.bias *= rhs;
    }
}

impl<T: Activation + Zeroable, const M: usize, const N: usize> DenseConnected<T, M, N> {
    pub const INPUT_SIZE: usize = M;
    pub const OUTPUT_SIZE: usize = N;

    pub fn weights_col(&self, idx: usize) -> &Vector<N> {
        &self.weights[idx]
    }

    pub fn weights_col_mut(&mut self, idx: usize) -> &mut Vector<N> {
        &mut self.weights[idx]
    }

    pub fn bias(&self) -> Vector<N> {
        self.bias
    }

    pub fn weights_norm(&self) -> f32 {
        let mut norm_sq = 0.0f32;
        for row in 0..M {
            for col in 0..N {
                let w = self.weights[row][col];
                norm_sq += w * w;
            }
        }
        for col in 0..N {
            let b = self.bias[col];
            norm_sq += b * b;
        }
        norm_sq.sqrt()
    }

    pub fn adamw<S: LRScheduler>(
        &mut self,
        gradients: &Self,
        momentum: &mut Self,
        velocity: &mut Self,
        optimizer: &AdamWOptimizer<S>,
    ) {
        optimizer.update_matrix(
            &mut self.weights,
            &gradients.weights,
            &mut momentum.weights,
            &mut velocity.weights,
        );
        optimizer.update_vector(
            &mut self.bias,
            &gradients.bias,
            &mut momentum.bias,
            &mut velocity.bias,
        );
    }

    pub fn bias_mut(&mut self) -> &mut Vector<N> {
        &mut self.bias
    }

    pub const fn zeroed() -> Self {
        Self::from_raw(Matrix::zeroed(), Vector::zeroed())
    }

    pub fn randomized(rng: &mut Rng) -> Box<Self> {
        let mut layer: Box<Self> = allocation::zeroed_box();
        T::Initializer::randomize_matrix(&mut layer.weights, rng);
        layer
    }

    pub const fn from_raw(weights: Matrix<M, N>, bias: Vector<N>) -> Self {
        Self {
            weights,
            bias,
            phantom: PhantomData,
        }
    }

    pub fn from_fn<W: FnMut(usize, usize) -> f32, B: FnMut(usize) -> f32>(w: W, b: B) -> Self {
        Self {
            weights: Matrix::from_fn(w),
            bias: Vector::from_fn(b),
            phantom: PhantomData,
        }
    }
}

pub struct DenseConnectedLayers<const N: usize> {
    pre_activation: Vector<N>,
    post_activation: Vector<N>,
}

impl<const N: usize> OutputLayer<Vector<N>> for DenseConnectedLayers<N> {
    fn output_layer(&self) -> Vector<N> {
        self.post_activation
    }
}

impl<T: Activation + Zeroable, const M: usize, const N: usize> FeedForwardNetwork
    for DenseConnected<T, M, N>
{
    type InputType = Vector<M>;
    type OutputType = Vector<N>;
    type Layers = DenseConnectedLayers<N>;

    fn out_with_layers(&self, input: &Self::InputType) -> Self::Layers {
        let pre_activation = self.weights.mul(input) + self.bias;
        Self::Layers {
            pre_activation,
            post_activation: pre_activation.activate::<T>(),
        }
    }

    fn backprop(
        &self,
        input: &Self::InputType,
        grad: &mut Self,
        out_err: Self::OutputType,
        layers: &Self::Layers,
    ) -> Self::InputType {
        let linear_err = out_err * layers.pre_activation.derivative::<T>();

        for (i, row) in grad.weights.iter_mut().enumerate() {
            row.madd(&linear_err, input[i]);
        }

        grad.bias += linear_err;
        self.weights.transpose_mul(&linear_err)
    }
}

#[repr(C)]
#[derive(Clone, Copy, Zeroable)]
pub struct SparseConnected<T, const M: usize, const N: usize> {
    weights: Matrix<M, N>,
    bias: Vector<N>,
    phantom: PhantomData<T>,
}

impl<T: Activation, const M: usize, const N: usize> std::ops::AddAssign<&SparseConnected<T, M, N>>
    for SparseConnected<T, M, N>
{
    fn add_assign(&mut self, rhs: &SparseConnected<T, M, N>) {
        self.weights += &rhs.weights;
        self.bias += rhs.bias;
    }
}

impl<T: Activation, const M: usize, const N: usize> std::ops::DivAssign<f32>
    for SparseConnected<T, M, N>
{
    fn div_assign(&mut self, rhs: f32) {
        self.weights /= rhs;
        self.bias /= rhs;
    }
}

impl<T: Activation, const M: usize, const N: usize> std::ops::MulAssign<f32>
    for SparseConnected<T, M, N>
{
    fn mul_assign(&mut self, rhs: f32) {
        self.weights *= rhs;
        self.bias *= rhs;
    }
}

impl<T: Activation + Zeroable, const M: usize, const N: usize> SparseConnected<T, M, N> {
    pub fn weights_row(&self, idx: usize) -> Vector<N> {
        self.weights[idx]
    }

    pub fn weights_row_mut(&mut self, idx: usize) -> &mut Vector<N> {
        &mut self.weights[idx]
    }

    pub fn bias(&self) -> Vector<N> {
        self.bias
    }

    pub fn weights_norm(&self) -> f32 {
        let mut norm_sq = 0.0f32;
        for row in 0..M {
            for col in 0..N {
                let w = self.weights[row][col];
                norm_sq += w * w;
            }
        }
        for col in 0..N {
            let b = self.bias[col];
            norm_sq += b * b;
        }
        norm_sq.sqrt()
    }

    pub fn adamw<S: LRScheduler>(
        &mut self,
        gradients: &Self,
        momentum: &mut Self,
        velocity: &mut Self,
        optimizer: &AdamWOptimizer<S>,
    ) {
        optimizer.update_matrix(
            &mut self.weights,
            &gradients.weights,
            &mut momentum.weights,
            &mut velocity.weights,
        );
        optimizer.update_vector(
            &mut self.bias,
            &gradients.bias,
            &mut momentum.bias,
            &mut velocity.bias,
        );
    }

    pub fn bias_mut(&mut self) -> &mut Vector<N> {
        &mut self.bias
    }

    pub const fn zeroed() -> Self {
        Self::from_raw(Matrix::zeroed(), Vector::zeroed())
    }

    pub fn randomized(rng: &mut Rng) -> Box<Self> {
        let mut layer: Box<Self> = allocation::zeroed_box();
        T::Initializer::randomize_matrix(&mut layer.weights, rng);
        layer
    }

    pub const fn from_raw(weights: Matrix<M, N>, bias: Vector<N>) -> Self {
        Self {
            weights,
            bias,
            phantom: PhantomData,
        }
    }

    pub fn from_fn<W: FnMut(usize, usize) -> f32, B: FnMut(usize) -> f32>(w: W, b: B) -> Self {
        Self {
            weights: Matrix::from_fn(w),
            bias: Vector::from_fn(b),
            phantom: PhantomData,
        }
    }
}

pub struct SparseConnectedLayers<const N: usize> {
    pre_activation: Vector<N>,
    post_activation: Vector<N>,
}

impl<const N: usize> OutputLayer<Vector<N>> for SparseConnectedLayers<N> {
    fn output_layer(&self) -> Vector<N> {
        self.post_activation
    }
}

impl<T: Activation + Zeroable, const M: usize, const N: usize> FeedForwardNetwork
    for SparseConnected<T, M, N>
{
    type InputType = SparseVector;
    type OutputType = Vector<N>;
    type Layers = SparseConnectedLayers<N>;

    fn out_with_layers(&self, input: &Self::InputType) -> Self::Layers {
        let mut pre_activation = self.bias;

        for &feat in input.iter() {
            pre_activation += self.weights[feat];
        }

        Self::Layers {
            pre_activation,
            post_activation: pre_activation.activate::<T>(),
        }
    }

    fn backprop(
        &self,
        input: &Self::InputType,
        grad: &mut Self,
        out_err: Self::OutputType,
        layers: &Self::Layers,
    ) -> Self::InputType {
        let linear_err = out_err * layers.pre_activation.derivative::<T>();

        for &feat in input.iter() {
            grad.weights[feat] += linear_err;
        }

        grad.bias += linear_err;
        SparseVector::with_capacity(0)
    }
}

use std::fmt::{self, Display};
use std::ops::{AddAssign, DivAssign};

use bytemuck::Zeroable;
use princhess::math::Rng;

use crate::neural::{
    AdamWOptimizer, FeedForwardNetwork, LRScheduler, ReLU, SparseConnected, SparseVector, Vector,
};

#[repr(C)]
#[derive(Zeroable)]
pub struct LinearNetwork<const I: usize, const A: usize> {
    pub output: SparseConnected<ReLU, I, A>,
}

impl<const I: usize, const A: usize> AddAssign<&Self> for LinearNetwork<I, A> {
    fn add_assign(&mut self, rhs: &Self) {
        self.output += &rhs.output;
    }
}

impl<const I: usize, const A: usize> DivAssign<f32> for LinearNetwork<I, A> {
    fn div_assign(&mut self, rhs: f32) {
        self.output /= rhs;
    }
}

impl<const I: usize, const A: usize> LinearNetwork<I, A> {
    pub fn randomize(&mut self) {
        let mut rng = Rng::default();
        self.output = *SparseConnected::randomized(&mut rng);
    }

    #[must_use]
    pub fn l1_norm(&self) -> f32 {
        self.output.l1_norm()
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

impl<const I: usize, const A: usize> Display for LinearNetwork<I, A> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{I}->{A}")
    }
}

impl<const I: usize, const A: usize> FeedForwardNetwork for LinearNetwork<I, A> {
    type InputType = SparseVector;
    type OutputType = Vector<A>;
    type Layers = <SparseConnected<ReLU, I, A> as FeedForwardNetwork>::Layers;

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

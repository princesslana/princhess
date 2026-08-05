use bytemuck::{allocation, Pod, Zeroable};

use crate::mem::Align16;
use crate::nets::Accumulator;

#[repr(C)]
#[derive(Copy, Clone, Zeroable)]
pub struct QuantizedLinearNetwork<const N: usize, const I: usize, const A: usize> {
    weights: [[Align16<Accumulator<i16, A>>; I]; N],
    bias: [Align16<Accumulator<i16, A>>; N],
}

unsafe impl<const N: usize, const I: usize, const A: usize> Pod
    for QuantizedLinearNetwork<N, I, A>
{
}

impl<const N: usize, const I: usize, const A: usize> QuantizedLinearNetwork<N, I, A> {
    #[must_use]
    pub fn boxed_from_slices(
        weights: &[Align16<[[i16; A]; I]>; N],
        bias: &[Align16<[i16; A]>; N],
    ) -> Box<Self> {
        let mut result: Box<Self> = allocation::zeroed_box();
        result.weights = *bytemuck::must_cast_ref(weights);
        result.bias = *bytemuck::must_cast_ref(bias);
        result
    }

    #[must_use]
    pub fn get_bias(&self, idx: usize) -> Accumulator<i16, A> {
        unsafe { **self.bias.get_unchecked(idx) }
    }

    #[must_use]
    pub fn get_weights(&self, idx: usize, feat_idx: usize) -> &Accumulator<i16, A> {
        unsafe { self.weights.get_unchecked(idx).get_unchecked(feat_idx) }
    }

    pub fn set(&self, idx: usize, feat_idx: usize, acc: &mut Accumulator<i16, A>) {
        acc.set(self.get_weights(idx, feat_idx));
    }
}

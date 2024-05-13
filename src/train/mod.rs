mod data;
mod policy;
mod value;

pub use crate::train::data::TrainingPosition;
pub use crate::train::policy::PolicyNetwork;
pub use crate::train::value::ValueNetwork;

use goober::activation::Activation;
use goober::layer::{DenseConnected, SparseConnected};
use std::alloc::{self, Layout};
use std::boxed::Box;

use crate::math::Rng;

fn randomize_sparse<A: Activation, const I: usize, const O: usize>(
    layer: &mut SparseConnected<A, I, O>,
    rng: &mut Rng,
) {
    let limit = (6. / (I + O) as f32).sqrt();

    for row_idx in 0..I {
        let row = layer.weights_row_mut(row_idx);
        for weight_idx in 0..O {
            row[weight_idx] = rng.next_f32_range(-limit, limit);
        }
    }
}

fn randomize_dense<A: Activation, const I: usize, const O: usize>(
    layer: &mut DenseConnected<A, I, O>,
    rng: &mut Rng,
) {
    let limit = (6. / (I + O) as f32).sqrt();

    for row_idx in 0..O {
        let row = layer.weights_row_mut(row_idx);
        for weight_idx in 0..I {
            row[weight_idx] = rng.next_f32_range(-limit, limit);
        }
    }
}

fn q_i16(x: f32, q: f32) -> i16 {
    let quantized = x * q;
    assert!(f32::from(i16::MIN) < quantized && quantized < f32::from(i16::MAX),);
    quantized as i16
}

fn q_i32(x: f32, q: f32) -> i32 {
    let quantized = x * q;
    assert!((i32::MIN as f32) < quantized && quantized < i32::MAX as f32);
    quantized as i32
}

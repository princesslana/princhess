use crate::neural::{Activation, DenseConnected, SparseConnected};
use princhess::math::Rng;

pub fn randomize_sparse<A: Activation, const I: usize, const O: usize>(
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

pub fn randomize_dense<A: Activation, const I: usize, const O: usize>(
    layer: &mut DenseConnected<A, I, O>,
    rng: &mut Rng,
) {
    let limit = (6. / (I + O) as f32).sqrt();

    for col_idx in 0..I {
        let col = layer.weights_col_mut(col_idx);
        for weight_idx in 0..O {
            col[weight_idx] = rng.next_f32_range(-limit, limit);
        }
    }
}

#[must_use]
pub fn q_i16(x: f32, q: i32) -> i16 {
    let quantized = x * q as f32;
    assert!(f32::from(i16::MIN) < quantized && quantized < f32::from(i16::MAX),);
    quantized as i16
}

#[must_use]
pub fn q_i32(x: f32, q: i32) -> i32 {
    let quantized = x * q as f32;
    assert!((i32::MIN as f32) < quantized && quantized < i32::MAX as f32);
    quantized as i32
}

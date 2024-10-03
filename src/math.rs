use goober::activation::Activation;
use goober::layer::{DenseConnected, SparseConnected};
use std::time::{SystemTime, UNIX_EPOCH};
use wide::f32x8;

use crate::simd;

pub fn softmax(arr: &mut [f32], t: f32) {
    let max = max(arr);

    let mut sum_x8 = f32x8::splat(0.);
    let max_x8 = f32x8::splat(max);
    let t_x8 = f32x8::splat(t);

    for c in arr.chunks_mut(8) {
        let c_x8 = simd::f32x8_from_slice_with_padding(c, f32::NEG_INFINITY);
        let c_x8 = ((c_x8 - max_x8) / t_x8).exp();

        sum_x8 += c_x8;

        c.copy_from_slice(&c_x8.as_array_ref()[..c.len()]);
    }

    let sum = sum_x8.reduce_add();
    let sum_x8 = f32x8::splat(sum);

    for c in arr.chunks_mut(8) {
        let c_x8 = simd::f32x8_from_slice_with_padding(c, 0.);
        let c_x8 = c_x8 / sum_x8;

        c.copy_from_slice(&c_x8.as_array_ref()[..c.len()]);
    }
}

fn max(arr: &[f32]) -> f32 {
    let mut max_x8 = f32x8::splat(f32::NEG_INFINITY);

    for chunk in arr.chunks(8) {
        let chunk = simd::f32x8_from_slice_with_padding(chunk, f32::NEG_INFINITY);
        max_x8 = max_x8.max(chunk);
    }

    let mut max = f32::NEG_INFINITY;
    for x in max_x8.as_array_ref() {
        max = max.max(*x);
    }
    max
}

#[must_use]
pub struct Rng {
    seed: u64,
}

impl Rng {
    fn with_seed(seed: u64) -> Self {
        Self { seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.seed ^= self.seed << 13;
        self.seed ^= self.seed >> 17;
        self.seed ^= self.seed << 5;
        self.seed
    }

    pub fn next_u32(&mut self) -> u32 {
        self.next_u64() as u32
    }

    pub fn next_usize(&mut self) -> usize {
        self.next_u64() as usize
    }

    pub fn next_i8(&mut self) -> i8 {
        self.next_u64() as i8
    }

    // Returns a random f32 in the range [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1 << 24) as f32
    }

    pub fn next_f32_range(&mut self, min: f32, max: f32) -> f32 {
        min + self.next_f32() * (max - min)
    }
}

impl Default for Rng {
    fn default() -> Self {
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis();

        Self::with_seed(seed as u64)
    }
}

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

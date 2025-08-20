use crate::neural::{Matrix, Vector};
use princhess::math::Rng;

pub trait WeightInitializer {
    fn randomize_matrix<const M: usize, const N: usize>(matrix: &mut Matrix<M, N>, rng: &mut Rng);
    fn randomize_vector<const N: usize>(vector: &mut Vector<N>, rng: &mut Rng);
}

pub struct Glorot;

impl WeightInitializer for Glorot {
    fn randomize_matrix<const M: usize, const N: usize>(matrix: &mut Matrix<M, N>, rng: &mut Rng) {
        let limit = (6.0 / (M + N) as f32).sqrt();
        for row in 0..M {
            for col in 0..N {
                matrix[row][col] = rng.next_f32_range(-limit, limit);
            }
        }
    }

    fn randomize_vector<const N: usize>(vector: &mut Vector<N>, rng: &mut Rng) {
        let limit = (1.0 / N as f32).sqrt();
        for i in 0..N {
            vector[i] = rng.next_f32_range(-limit, limit);
        }
    }
}

pub struct He;

impl WeightInitializer for He {
    fn randomize_matrix<const M: usize, const N: usize>(matrix: &mut Matrix<M, N>, rng: &mut Rng) {
        let limit = (6.0 / M as f32).sqrt();
        for row in 0..M {
            for col in 0..N {
                matrix[row][col] = rng.next_f32_range(-limit, limit);
            }
        }
    }

    fn randomize_vector<const N: usize>(vector: &mut Vector<N>, rng: &mut Rng) {
        let limit = (1.0 / N as f32).sqrt();
        for i in 0..N {
            vector[i] = rng.next_f32_range(-limit, limit);
        }
    }
}

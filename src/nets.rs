use bytemuck::{self, Pod, Zeroable};
use goober::activation::Activation;
use std::fs;
use std::io::Write;
use std::path::Path;

// Workaround for error in how goober handles an activation such as SCReLU
#[derive(Clone, Copy)]
pub struct SCReLU;

impl Activation for SCReLU {
    fn activate(x: f32) -> f32 {
        let clamped = x.clamp(0.0, 1.0);
        clamped * clamped
    }

    fn derivative(x: f32) -> f32 {
        if 0.0 < x && x < 1.0 {
            2.0 * x.sqrt()
        } else {
            0.0
        }
    }
}

#[derive(Clone, Copy, Debug, Zeroable)]
#[repr(C)]
pub struct Accumulator<const H: usize> {
    pub vals: [i16; H],
}

unsafe impl<const H: usize> Pod for Accumulator<H> {}

impl<const H: usize> Accumulator<H> {
    pub fn set(&mut self, weights: &Accumulator<H>) {
        for (i, d) in self.vals.iter_mut().zip(&weights.vals) {
            *i += *d;
        }
    }

    pub fn dot_relu(&self, rhs: &Accumulator<H>) -> i32 {
        let mut result = 0;

        for (a, b) in self.vals.iter().zip(&rhs.vals) {
            result += relu(*a) * relu(*b);
        }

        result
    }
}

pub fn relu(x: i16) -> i32 {
    i32::from(x).max(0)
}

pub fn screlu(x: i16, q: i32) -> i32 {
    let clamped = i32::from(x).clamp(0, q);
    clamped * clamped
}

pub fn q_i16(x: f32, q: i32) -> i16 {
    let quantized = x * q as f32;
    assert!(f32::from(i16::MIN) < quantized && quantized < f32::from(i16::MAX),);
    quantized as i16
}

pub fn q_i32(x: f32, q: i32) -> i32 {
    let quantized = x * q as f32;
    assert!((i32::MIN as f32) < quantized && quantized < i32::MAX as f32);
    quantized as i32
}

pub fn save_to_bin<T: Pod>(dir: &Path, file_name: &str, data: &T) {
    let mut file = fs::File::create(dir.join(file_name)).expect("Failed to create file");

    let slice = bytemuck::bytes_of(data);

    file.write_all(slice).unwrap();
}

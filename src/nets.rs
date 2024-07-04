use std::fs;
use std::io::Write;
use std::mem;
use std::path::Path;
use std::slice;

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Accumulator<const H: usize> {
    pub vals: [i16; H],
}

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

pub fn save_to_bin<T>(dir: &Path, file_name: &str, data: &T) {
    let mut file = fs::File::create(dir.join(file_name)).expect("Failed to create file");

    let size_of = mem::size_of::<T>();

    unsafe {
        let ptr: *const T = data;
        let slice_ptr: *const u8 = ptr.cast::<u8>();
        let slice = slice::from_raw_parts(slice_ptr, size_of);
        file.write_all(slice).unwrap();
    }
}

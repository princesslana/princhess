mod data;
mod value;

pub use crate::train::data::TrainingPosition;
pub use crate::train::value::ValueNetwork;

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

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

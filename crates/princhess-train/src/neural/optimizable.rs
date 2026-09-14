pub trait AsParams {
    fn params(&self) -> &[f32];
    fn params_mut(&mut self) -> &mut [f32];
}

/// # Safety
/// Implementors must guarantee their memory layout is `#[repr(C)]`, composed
/// entirely of `f32` values with no padding, and that `align_of::<Self>() == align_of::<f32>()`.
pub unsafe trait Optimizable: AsParams {}

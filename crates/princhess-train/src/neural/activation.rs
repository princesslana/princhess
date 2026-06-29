use crate::neural::initialization::{Glorot, He, WeightInitializer};
use bytemuck::Zeroable;

pub trait Activation: Copy + Zeroable {
    type Initializer: WeightInitializer;

    fn name() -> &'static str;

    fn activate(x: f32) -> f32;

    fn derivative(x: f32) -> f32;
}

#[derive(Clone, Copy, Zeroable)]
pub struct Identity;
impl Activation for Identity {
    type Initializer = Glorot;

    fn name() -> &'static str {
        "identity"
    }

    fn activate(x: f32) -> f32 {
        x
    }

    fn derivative(_: f32) -> f32 {
        1.0
    }
}

#[derive(Clone, Copy, Zeroable)]
pub struct ReLU;
impl Activation for ReLU {
    type Initializer = He;

    fn name() -> &'static str {
        "relu"
    }

    fn activate(x: f32) -> f32 {
        x.max(0.0)
    }

    fn derivative(x: f32) -> f32 {
        if x > 0.0 {
            1.0
        } else {
            0.0
        }
    }
}

#[derive(Clone, Copy, Zeroable)]
pub struct SCReLU;
impl Activation for SCReLU {
    type Initializer = He;

    fn name() -> &'static str {
        "screlu"
    }

    fn activate(x: f32) -> f32 {
        let clamped = x.clamp(0.0, 1.0);
        clamped * clamped
    }

    fn derivative(x: f32) -> f32 {
        if 0.0 < x && x < 1.0 {
            2.0 * x
        } else {
            0.0
        }
    }
}

#[derive(Clone, Copy, Zeroable)]
pub struct HardSwish;
impl Activation for HardSwish {
    type Initializer = He;

    fn name() -> &'static str {
        "hardswish"
    }

    fn activate(x: f32) -> f32 {
        x * (x + 3.0).clamp(0.0, 6.0) / 6.0
    }

    fn derivative(x: f32) -> f32 {
        if x <= -3.0 {
            0.0
        } else if x >= 3.0 {
            1.0
        } else {
            (2.0 * x + 3.0) / 6.0
        }
    }
}

#[derive(Clone, Copy, Zeroable)]
pub struct SoftSign;
impl Activation for SoftSign {
    type Initializer = Glorot;

    fn name() -> &'static str {
        "softsign"
    }

    fn activate(x: f32) -> f32 {
        x / (1.0 + x.abs())
    }

    fn derivative(x: f32) -> f32 {
        let d = 1.0 + x.abs();
        1.0 / (d * d)
    }
}

#[derive(Clone, Copy, Zeroable)]
pub struct Tanh;
impl Activation for Tanh {
    type Initializer = Glorot;

    fn name() -> &'static str {
        "tanh"
    }

    fn activate(x: f32) -> f32 {
        x.tanh()
    }

    fn derivative(x: f32) -> f32 {
        let t = x.tanh();
        1.0 - t * t
    }
}

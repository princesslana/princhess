use crate::neural::initialization::{Glorot, He, WeightInitializer};
use bytemuck::Zeroable;

pub trait Activation: Copy {
    type Initializer: WeightInitializer;

    fn activate(x: f32) -> f32;

    fn derivative(x: f32) -> f32;
}

#[derive(Clone, Copy, Zeroable)]
pub struct Identity;
impl Activation for Identity {
    type Initializer = Glorot;

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
pub struct Tanh;
impl Activation for Tanh {
    type Initializer = Glorot;

    fn activate(x: f32) -> f32 {
        x.tanh()
    }

    fn derivative(x: f32) -> f32 {
        let t = x.tanh();
        1.0 - t * t
    }
}

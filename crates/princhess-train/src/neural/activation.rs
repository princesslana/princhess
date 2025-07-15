pub trait Activation: Copy {
    fn activate(x: f32) -> f32;

    fn derivative(x: f32) -> f32;
}

#[derive(Clone, Copy)]
pub struct Identity;
impl Activation for Identity {
    fn activate(x: f32) -> f32 {
        x
    }

    fn derivative(_: f32) -> f32 {
        1.0
    }
}

#[derive(Clone, Copy)]
pub struct ReLU;
impl Activation for ReLU {
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

#[derive(Clone, Copy)]
pub struct SCReLU;
impl Activation for SCReLU {
    fn activate(x: f32) -> f32 {
        let clamped = x.clamp(0.0, 1.0);
        clamped * clamped
    }

    fn derivative(x: f32) -> f32 {
        // Workaround for error in how goober handles an activation such as SCReLU
        if 0.0 < x && x < 1.0 {
            2.0 * x.sqrt()
        } else {
            0.0
        }
    }
}

#[derive(Clone, Copy)]
pub struct Tanh;
impl Activation for Tanh {
    fn activate(x: f32) -> f32 {
        x.tanh()
    }

    fn derivative(x: f32) -> f32 {
        let t = x.tanh();
        1.0 - t * t
    }
}

use goober::activation::Activation;

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

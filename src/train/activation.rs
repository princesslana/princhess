use goober::activation::Activation;

#[derive(Copy, Clone)]
pub struct TanH;

impl Activation for TanH {
    fn activate(x: f32) -> f32 {
        x.tanh()
    }

    fn derivative(x: f32) -> f32 {
        let t = x.tanh();
        1.0 - t * t
    }
}

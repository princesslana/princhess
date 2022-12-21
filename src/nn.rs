use state;

const NUMBER_INPUTS: usize = state::NUMBER_FEATURES;
const NUMBER_HIDDEN: usize = 128;
const NUMBER_OUTPUTS: usize = 1;

#[allow(clippy::excessive_precision)]
const EVAL_HIDDEN_BIAS: [f32; NUMBER_HIDDEN] = include!("model/hidden_bias_0");

#[allow(clippy::excessive_precision)]
const EVAL_HIDDEN_WEIGHTS: [[f32; NUMBER_INPUTS]; NUMBER_HIDDEN] =
    include!("model/hidden_weights_0");

#[allow(clippy::excessive_precision)]
const EVAL_OUTPUT_WEIGHTS: [[f32; NUMBER_HIDDEN]; NUMBER_OUTPUTS] =
    include!("model/output_weights");

pub struct NN {
    hidden_layer: [f32; NUMBER_HIDDEN],
}

impl NN {
    pub fn new() -> Self {
        #[allow(clippy::uninit_assumed_init)]
        let hidden_layer = unsafe { std::mem::MaybeUninit::uninit().assume_init() };
        Self { hidden_layer }
    }

    pub fn set_inputs(&mut self, inputs: &[f32; state::NUMBER_FEATURES]) {
        self.hidden_layer.copy_from_slice(&EVAL_HIDDEN_BIAS);

        for i in 0..inputs.len() {
            if inputs[i] > 0.5 {
                for j in 0..self.hidden_layer.len() {
                    self.hidden_layer[j] += EVAL_HIDDEN_WEIGHTS[j][i];
                }
            }
        }
    }

    pub fn get_output(&self) -> f32 {
        let mut result = 0.;

        let weights = EVAL_OUTPUT_WEIGHTS[0];

        for i in 0..self.hidden_layer.len() {
            result += weights[i] * self.hidden_layer[i].max(0.);
        }

        result
    }
}

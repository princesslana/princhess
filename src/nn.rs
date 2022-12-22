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

pub struct NN;

impl NN {
    pub fn new() -> Self {
        Self
    }

    pub fn get_output(&self, inputs: &[f32; state::NUMBER_FEATURES]) -> f32 {
        #[allow(clippy::uninit_assumed_init)]
        let mut hidden_layer: [f32; NUMBER_HIDDEN] =
            unsafe { std::mem::MaybeUninit::uninit().assume_init() };

        hidden_layer.copy_from_slice(&EVAL_HIDDEN_BIAS);

        for i in 0..inputs.len() {
            if inputs[i] > 0.5 {
                for j in 0..hidden_layer.len() {
                    hidden_layer[j] += EVAL_HIDDEN_WEIGHTS[j][i];
                }
            }
        }

        let mut result = 0.;

        let weights = EVAL_OUTPUT_WEIGHTS[0];

        for i in 0..hidden_layer.len() {
            result += weights[i] * hidden_layer[i].max(0.);
        }

        result
    }
}

use state;

const NUMBER_INPUTS: usize = state::NUMBER_FEATURES;
const NUMBER_HIDDEN: usize = 128;

struct NNWeights {
    hidden_bias: &'static [f32],
    hidden: &'static [[f32; NUMBER_INPUTS]],
    output: &'static [[f32; NUMBER_HIDDEN]],
}

#[allow(clippy::excessive_precision)]
const EVAL_HIDDEN_BIAS: [f32; NUMBER_HIDDEN] = include!("model/hidden_bias");

#[allow(clippy::excessive_precision)]
const EVAL_HIDDEN_WEIGHTS: [[f32; NUMBER_INPUTS]; NUMBER_HIDDEN] = include!("model/hidden_weights");

#[allow(clippy::excessive_precision)]
const EVAL_OUTPUT_WEIGHTS: [[f32; NUMBER_HIDDEN]; 1] = include!("model/output_weights");

const EVAL_WEIGHTS: NNWeights = NNWeights {
    hidden_bias: &EVAL_HIDDEN_BIAS,
    hidden: &EVAL_HIDDEN_WEIGHTS,
    output: &EVAL_OUTPUT_WEIGHTS,
};

pub struct NN {
    weights: NNWeights,
    hidden_layer: [f32; NUMBER_HIDDEN],
}

impl NN {
    fn new(weights: NNWeights) -> Self {
        #[allow(clippy::uninit_assumed_init)]
        let hidden = unsafe { std::mem::MaybeUninit::uninit().assume_init() };
        Self {
            weights,
            hidden_layer: hidden,
        }
    }

    pub fn new_eval() -> Self {
        Self::new(EVAL_WEIGHTS)
    }

    pub fn set_inputs(&mut self, inputs: &[f32; state::NUMBER_FEATURES]) {
        self.hidden_layer.copy_from_slice(self.weights.hidden_bias);

        for i in 0..inputs.len() {
            if inputs[i] > 0.5 {
                for j in 0..self.hidden_layer.len() {
                    self.hidden_layer[j] += self.weights.hidden[j][i];
                }
            }
        }
    }

    pub fn get_output(&self, idx: usize) -> f32 {
        let mut result = 0.;

        let weights = self.weights.output[idx];

        for i in 0..self.hidden_layer.len() {
            result += weights[i] * self.hidden_layer[i].max(0.);
        }

        result
    }
}

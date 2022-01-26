pub const NUMBER_FEATURES: usize = 768;
const NUMBER_HIDDEN: usize = 32;

struct NNWeights {
    hidden_bias: &'static [f32],
    hidden: &'static [[f32; NUMBER_FEATURES]],
    output: &'static [[f32; NUMBER_HIDDEN]],
}

const EVAL_HIDDEN_BIAS: [f32; NUMBER_HIDDEN] = include!("model/hidden_bias");
const EVAL_HIDDEN_WEIGHTS: [[f32; NUMBER_FEATURES]; NUMBER_HIDDEN] =
    include!("model/hidden_weights");

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

    pub fn set_inputs(&mut self, inputs: &[f32; NUMBER_FEATURES]) {
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

        for i in 0..self.hidden_layer.len() {
            result += self.weights.output[idx][i] * self.hidden_layer[i].max(0.);
        }

        result
    }
}

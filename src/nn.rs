pub const NUMBER_FEATURES: usize = 768;
const NUMBER_HIDDEN: usize = 32;

struct NNWeights {
    hidden1_bias: &'static [f32; NUMBER_HIDDEN],
    hidden1: &'static [[f32; NUMBER_FEATURES]; NUMBER_HIDDEN],
    hidden2_bias: &'static [f32; NUMBER_HIDDEN],
    hidden2: &'static [[f32; NUMBER_HIDDEN]; NUMBER_HIDDEN],
    output: &'static [[f32; NUMBER_HIDDEN]],
}

#[allow(clippy::excessive_precision)]
const EVAL_HIDDEN1_BIAS: [f32; NUMBER_HIDDEN] = include!("model/hidden1_bias");

#[allow(clippy::excessive_precision)]
const EVAL_HIDDEN1_WEIGHTS: [[f32; NUMBER_FEATURES]; NUMBER_HIDDEN] =
    include!("model/hidden1_weights");

#[allow(clippy::excessive_precision)]
const EVAL_HIDDEN2_BIAS: [f32; NUMBER_HIDDEN] = include!("model/hidden2_bias");

#[allow(clippy::excessive_precision)]
const EVAL_HIDDEN2_WEIGHTS: [[f32; NUMBER_HIDDEN]; NUMBER_HIDDEN] =
    include!("model/hidden2_weights");

#[allow(clippy::excessive_precision)]
const EVAL_OUTPUT_WEIGHTS: [[f32; NUMBER_HIDDEN]; 1] = include!("model/output_weights");

const EVAL_WEIGHTS: NNWeights = NNWeights {
    hidden1_bias: &EVAL_HIDDEN1_BIAS,
    hidden1: &EVAL_HIDDEN1_WEIGHTS,
    hidden2_bias: &EVAL_HIDDEN2_BIAS,
    hidden2: &EVAL_HIDDEN2_WEIGHTS,
    output: &EVAL_OUTPUT_WEIGHTS,
};

pub struct NN {
    weights: NNWeights,
    hidden1: [f32; NUMBER_HIDDEN],
    hidden2: [f32; NUMBER_HIDDEN],
}

impl NN {
    fn new(weights: NNWeights) -> Self {
        #[allow(clippy::uninit_assumed_init)]
        let hidden1 = unsafe { std::mem::MaybeUninit::uninit().assume_init() };

        #[allow(clippy::uninit_assumed_init)]
        let hidden2 = unsafe { std::mem::MaybeUninit::uninit().assume_init() };

        Self {
            weights,
            hidden1,
            hidden2,
        }
    }

    pub fn new_eval() -> Self {
        Self::new(EVAL_WEIGHTS)
    }

    pub fn set_inputs(&mut self, inputs: &[f32; NUMBER_FEATURES]) {
        self.hidden1.copy_from_slice(self.weights.hidden1_bias);
        self.hidden2.copy_from_slice(self.weights.hidden2_bias);

        for i in 0..inputs.len() {
            if inputs[i] > 0.5 {
                for j in 0..self.hidden1.len() {
                    self.hidden1[j] += self.weights.hidden1[j][i];
                }
            }
        }

        for idx in 0..self.hidden2.len() {
            for (h1, w) in self.hidden1.iter().zip(self.weights.hidden2[idx]) {
                self.hidden2[idx] += h1.max(0.) * w;
            }
            self.hidden2[idx] = self.hidden2[idx].max(0.)
        }
    }

    pub fn get_output(&self, idx: usize) -> f32 {
        let mut result = 0.;

        for (h, w) in self.hidden2.iter().zip(self.weights.output[idx]) {
            result += h * w;
        }

        result
    }
}

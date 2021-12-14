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

const FROM_HIDDEN_BIAS: [f32; NUMBER_HIDDEN] = include!("from_model/hidden_bias");
const FROM_HIDDEN_WEIGHTS: [[f32; NUMBER_FEATURES]; NUMBER_HIDDEN] = include!("from_model/hidden_weights");
const FROM_OUTPUT_WEIGHTS: [[f32; NUMBER_HIDDEN]; 64] = include!("from_model/output_weights");

const FROM_WEIGHTS: NNWeights = NNWeights {
    hidden_bias: &FROM_HIDDEN_BIAS,
    hidden: &FROM_HIDDEN_WEIGHTS,
    output: &FROM_OUTPUT_WEIGHTS,
};

const TO_HIDDEN_BIAS: [f32; NUMBER_HIDDEN] = include!("to_model/hidden_bias");
const TO_HIDDEN_WEIGHTS: [[f32; NUMBER_FEATURES]; NUMBER_HIDDEN] = include!("to_model/hidden_weights");
const TO_OUTPUT_WEIGHTS: [[f32; NUMBER_HIDDEN]; 64] = include!("to_model/output_weights");

const TO_WEIGHTS: NNWeights = NNWeights {
    hidden_bias: &TO_HIDDEN_BIAS,
    hidden: &TO_HIDDEN_WEIGHTS,
    output: &TO_OUTPUT_WEIGHTS,
};

pub struct NN {
    weights: NNWeights,
    hidden_layer: [f32; NUMBER_HIDDEN],
}

impl NN {
    fn new(weights: NNWeights) -> Self {
        Self {
            weights,
            hidden_layer: [0f32; NUMBER_HIDDEN],
        }
    }

    pub fn new_eval() -> Self {
        Self::new(EVAL_WEIGHTS)
    }

    pub fn new_from() -> Self {
        Self::new(FROM_WEIGHTS)
    }

    pub fn new_to() -> Self {
        Self::new(TO_WEIGHTS)
    }

    pub fn set_inputs(&mut self, inputs: &[f32]) {
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

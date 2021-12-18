pub const NUMBER_FEATURES: usize = 768;

const EVAL_HIDDEN: usize = 32;
const POLICY_HIDDEN: usize = 128;
const P_HIDDEN: usize = 256;

struct NNWeights<const NH: usize> {
    hidden_bias: &'static [f32],
    hidden: &'static [[f32; NUMBER_FEATURES]],
    output: &'static [[f32; NH]],
}

#[allow(clippy::excessive_precision)]
const EVAL_HIDDEN_BIAS: [f32; NUMBER_HIDDEN] = include!("model/hidden_bias");

#[allow(clippy::excessive_precision)]
const EVAL_HIDDEN_WEIGHTS: [[f32; NUMBER_FEATURES]; NUMBER_HIDDEN] =
    include!("model/hidden_weights");

#[allow(clippy::excessive_precision)]
const EVAL_OUTPUT_WEIGHTS: [[f32; NUMBER_HIDDEN]; 1] = include!("model/output_weights");

const EVAL_WEIGHTS: NNWeights<EVAL_HIDDEN> = NNWeights {
    hidden_bias: &EVAL_HIDDEN_BIAS,
    hidden: &EVAL_HIDDEN_WEIGHTS,
    output: &EVAL_OUTPUT_WEIGHTS,
};

const POLICY_HIDDEN_BIAS: [f32; P_HIDDEN] = include!("policy/hidden_bias");
const POLICY_HIDDEN_WEIGHTS: [[f32; NUMBER_FEATURES]; P_HIDDEN] =
    include!("policy/hidden_weights");

const POLICY_OUTPUT_WEIGHTS: [[f32; P_HIDDEN]; 4096] = include!("policy/output_weights");

const POLICY_WEIGHTS: NNWeights<P_HIDDEN> = NNWeights {
    hidden_bias: &POLICY_HIDDEN_BIAS,
    hidden: &POLICY_HIDDEN_WEIGHTS,
    output: &POLICY_OUTPUT_WEIGHTS,
};


const FROM_HIDDEN_BIAS: [f32; POLICY_HIDDEN] = include!("from_model/hidden_bias");
const FROM_HIDDEN_WEIGHTS: [[f32; NUMBER_FEATURES]; POLICY_HIDDEN] = include!("from_model/hidden_weights");
const FROM_OUTPUT_WEIGHTS: [[f32; POLICY_HIDDEN]; 64] = include!("from_model/output_weights");

const FROM_WEIGHTS: NNWeights<POLICY_HIDDEN> = NNWeights {
    hidden_bias: &FROM_HIDDEN_BIAS,
    hidden: &FROM_HIDDEN_WEIGHTS,
    output: &FROM_OUTPUT_WEIGHTS,
};

const TO_HIDDEN_BIAS: [f32; POLICY_HIDDEN] = include!("to_model/hidden_bias");
const TO_HIDDEN_WEIGHTS: [[f32; NUMBER_FEATURES]; POLICY_HIDDEN] = include!("to_model/hidden_weights");
const TO_OUTPUT_WEIGHTS: [[f32; POLICY_HIDDEN]; 64] = include!("to_model/output_weights");

const TO_WEIGHTS: NNWeights<POLICY_HIDDEN> = NNWeights {
    hidden_bias: &TO_HIDDEN_BIAS,
    hidden: &TO_HIDDEN_WEIGHTS,
    output: &TO_OUTPUT_WEIGHTS,
};

pub struct NN<const NH: usize> {
    weights: NNWeights<NH>,
    hidden_layer: [f32; NH],
}

pub fn new_eval() -> NN<EVAL_HIDDEN> {
    NN::new(EVAL_WEIGHTS)
}

pub fn new_policy() -> NN<P_HIDDEN> {
    NN::new(POLICY_WEIGHTS)
}

pub fn new_from() -> NN<POLICY_HIDDEN> {
    NN::new(FROM_WEIGHTS)
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

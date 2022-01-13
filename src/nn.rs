use std::cmp;

pub const NUMBER_FEATURES: usize = 768;

const EVAL_HIDDEN: usize = 32;
const POLICY_HIDDEN: usize = 128;
const P_HIDDEN: usize = 256;

struct NNWeights<const NH: usize> {
    hidden_bias: &'static [f32],
    hidden: &'static [[f32; NUMBER_FEATURES]],
    output: &'static [[f32; NH]],
}

const EVAL_HIDDEN_BIAS: [f32; EVAL_HIDDEN] = include!("model/hidden_bias");
const EVAL_HIDDEN_WEIGHTS: [[f32; NUMBER_FEATURES]; EVAL_HIDDEN] = include!("model/hidden_weights");

const EVAL_OUTPUT_WEIGHTS: [[f32; EVAL_HIDDEN]; 1] = include!("model/output_weights");

const EVAL_WEIGHTS: NNWeights<EVAL_HIDDEN> = NNWeights {
    hidden_bias: &EVAL_HIDDEN_BIAS,
    hidden: &EVAL_HIDDEN_WEIGHTS,
    output: &EVAL_OUTPUT_WEIGHTS,
};

const POLICY_HIDDEN_BIAS: [f32; P_HIDDEN] = include!("policy/hidden_bias");
const POLICY_HIDDEN_WEIGHTS: [[f32; NUMBER_FEATURES]; P_HIDDEN] = include!("policy/hidden_weights");

const POLICY_OUTPUT_WEIGHTS: [[f32; P_HIDDEN]; 4096] = include!("policy/output_weights");

const POLICY_WEIGHTS: NNWeights<P_HIDDEN> = NNWeights {
    hidden_bias: &POLICY_HIDDEN_BIAS,
    hidden: &POLICY_HIDDEN_WEIGHTS,
    output: &POLICY_OUTPUT_WEIGHTS,
};

const FROM_HIDDEN_BIAS: [f32; POLICY_HIDDEN] = include!("from_model/hidden_bias");
const FROM_HIDDEN_WEIGHTS: [[f32; NUMBER_FEATURES]; POLICY_HIDDEN] =
    include!("from_model/hidden_weights");
const FROM_OUTPUT_WEIGHTS: [[f32; POLICY_HIDDEN]; 64] = include!("from_model/output_weights");

const FROM_WEIGHTS: NNWeights<POLICY_HIDDEN> = NNWeights {
    hidden_bias: &FROM_HIDDEN_BIAS,
    hidden: &FROM_HIDDEN_WEIGHTS,
    output: &FROM_OUTPUT_WEIGHTS,
};

const TO_HIDDEN_BIAS: [f32; POLICY_HIDDEN] = include!("to_model/hidden_bias");
const TO_HIDDEN_WEIGHTS: [[f32; NUMBER_FEATURES]; POLICY_HIDDEN] =
    include!("to_model/hidden_weights");
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

pub fn new_to() -> NN<POLICY_HIDDEN> {
    NN::new(TO_WEIGHTS)
}

impl<const NH: usize> NN<NH> {
    fn new(weights: NNWeights<NH>) -> Self {
        NN {
            weights,
            hidden_layer: [0f32; NH],
        }
    }

    pub fn set_inputs(&mut self, inputs: &[f32]) {
        self.hidden_layer.copy_from_slice(self.weights.hidden_bias);

        for i in 0..inputs.len() {
            if inputs[i] == 1.0 {
                for (h, ws) in self.hidden_layer.iter_mut().zip(self.weights.hidden) {
                    *h += ws[i]
                }
            }
        }

        for h in self.hidden_layer.iter_mut() {
            *h = h.max(0.);
        }
    }

    pub fn get_output(&self, idx: usize) -> f32 {
        unrolled_dot(&self.hidden_layer, &self.weights.output[idx])
    }
}

/// Compute the dot product.
///
/// `xs` and `ys` must be the same length
fn unrolled_dot(xs: &[f32], ys: &[f32]) -> f32 {
    debug_assert_eq!(xs.len(), ys.len());
    // eightfold unrolled so that floating point can be vectorized
    // (even with strict floating point accuracy semantics)
    let len = cmp::min(xs.len(), ys.len());
    let mut xs = &xs[..len];
    let mut ys = &ys[..len];
    let mut sum = 0.;
    let (mut p0, mut p1, mut p2, mut p3, mut p4, mut p5, mut p6, mut p7) =
        (0., 0., 0., 0., 0., 0., 0., 0.);
    while xs.len() >= 8 {
        p0 = p0 + xs[0] * ys[0];
        p1 = p1 + xs[1] * ys[1];
        p2 = p2 + xs[2] * ys[2];
        p3 = p3 + xs[3] * ys[3];
        p4 = p4 + xs[4] * ys[4];
        p5 = p5 + xs[5] * ys[5];
        p6 = p6 + xs[6] * ys[6];
        p7 = p7 + xs[7] * ys[7];

        xs = &xs[8..];
        ys = &ys[8..];
    }
    sum = sum + (p0 + p4);
    sum = sum + (p1 + p5);
    sum = sum + (p2 + p6);
    sum = sum + (p3 + p7);

    for (i, (&x, &y)) in xs.iter().zip(ys).enumerate() {
        if i >= 7 {
            break;
        }
        sum = sum + x * y;
    }
    sum
}

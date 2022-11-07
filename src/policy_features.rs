use state;
use state::{Move, State};

const NUMBER_INPUTS: usize = state::NUMBER_FEATURES;

#[allow(clippy::excessive_precision)]
const POLICY_WEIGHTS: [[f32; NUMBER_INPUTS]; 1792] = include!("policy/output_weights");

pub fn evaluate_moves(
    state: &State,
    features: &[f32; state::NUMBER_FEATURES],
    moves: &[Move],
) -> Vec<f32> {
    let mut evalns = Vec::with_capacity(moves.len());

    if moves.len() == 0 {
        return evalns;
    }

    let mut move_idxs = Vec::with_capacity(moves.len());

    for m in 0..moves.len() {
        move_idxs.push(state.move_to_index(&moves[m]));
        evalns.push(0.);
    }

    for f in 0..features.len() {
        if features[f] > 0.5 {
            for m in 0..moves.len() {
                evalns[m] += POLICY_WEIGHTS[move_idxs[m]][f];
            }
        }
    }

    softmax(&mut evalns);

    evalns
}

pub fn softmax(arr: &mut [f32]) {
    let max = max(arr);
    let mut s = 0.;

    for x in arr.iter_mut() {
        *x = fastapprox::faster::exp(*x - max);
        s += *x;
    }
    for x in arr.iter_mut() {
        *x /= s;
    }
}

fn max(arr: &[f32]) -> f32 {
    let mut max = std::f32::NEG_INFINITY;
    for x in arr.iter() {
        max = max.max(*x);
    }
    max
}

use chess::*;
use features::FeatureVec;
use nn;
use state::{Move, State};

pub fn featurize(state: &State) -> FeatureVec {
    FeatureVec {
        arr: state.features().iter().map(|v| *v as i8).collect(),
    }
}

pub fn evaluate_single(state: &State, mov: &Move) -> f32 {
    let mut p_nn = nn::new_policy();

    p_nn.set_inputs(&state.features());

    p_nn.get_output(state.move_to_index(mov))
}

pub fn evaluate_moves(state: &State, moves: &[Move]) -> Vec<f32> {
    let mut p_nn = nn::new_policy();

    p_nn.set_inputs(&state.features());

    let mut evalns: Vec<_> = moves.iter().map(|x| {
        p_nn.get_output(state.move_to_index(x)).max(0.)
    }).collect();
    //let mut evalns: Vec<_> = moves.iter().map(|x| evaluate_single(state, x)).collect();
    softmax(&mut evalns);

    evalns
}

pub fn softmax(arr: &mut [f32]) {
    for x in arr.iter_mut() {
        *x = x.exp();
    }
    let s = 1.0 / arr.iter().sum::<f32>();
    for x in arr.iter_mut() {
        *x *= s;
    }
}

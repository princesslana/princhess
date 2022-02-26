use nn::NN;
use state::{Move, State};

pub fn evaluate_moves(state: &State, moves: &[Move]) -> Vec<f32> {
    let mut nn = NN::new_policy();
    nn.set_inputs(&state.features());

    let mut evalns: Vec<_> = moves
        .iter()
        .map(|x| nn.get_output(state.move_to_index(x)))
        .collect();
    softmax(&mut evalns);
    evalns
}

pub fn softmax(arr: &mut [f32]) {
    let mut max = std::f32::NEG_INFINITY;
    for x in arr.iter() {
        if x > &max {
            max = *x
        }
    }
    for x in arr.iter_mut() {
        *x = (*x - max).exp();
    }
    let s = 1.0 / arr.iter().sum::<f32>();
    for x in arr.iter_mut() {
        *x *= s;
    }
}

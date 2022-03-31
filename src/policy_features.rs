use state::{Move, State};

pub fn evaluate_moves(state: &State, moves: &[Move]) -> Vec<f32> {
    vec![1. / moves.len() as f32; moves.len()]
}

pub fn softmax(arr: &mut [f32]) {
    for x in arr.iter_mut() {
        *x = fastapprox::faster::exp(*x);
    }
    let s = 1.0 / arr.iter().sum::<f32>();
    for x in arr.iter_mut() {
        *x *= s;
    }
}

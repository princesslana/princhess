use nn;
use nn::NN;
use options::{
    get_policy_bad_capture_factor, get_policy_good_capture_factor, get_policy_softmax_temp,
};
use shakmaty::Setup;
use state::{Move, State};

#[allow(clippy::excessive_precision)]
const POLICY_WEIGHTS: [[f32; nn::NUMBER_FEATURES]; 1792] = include!("policy/output_weights");

/*
const HIDDEN_BIAS: [f32; 32] = include!("policy/hidden_bias_1");
const HIDDEN_WEIGHTS: [[f32; nn::NUMBER_FEATURES]; 32] = include!("policy/hidden_weights_1");
*/

fn piece_eval(pc: chess::Piece) -> u16 {
    match pc {
        chess::Piece::King => 20000,
        chess::Piece::Queen => 900,
        chess::Piece::Rook => 500,
        chess::Piece::Bishop => 300,
        chess::Piece::Knight => 300,
        chess::Piece::Pawn => 100,
    }
}

pub fn evaluate_moves(
    state: &State,
    features: &[f32; nn::NUMBER_FEATURES],
    moves: &[Move],
) -> Vec<f32> {
    let mut evalns = Vec::with_capacity(moves.len());
    /*let mut nn = NN::new_policy();

    nn.set_inputs(features);

    let mut evalns: Vec<_> =
         moves.iter().map(|m| nn.get_output(state.move_to_index(m))).collect();
         */
    /*
    for m in moves.iter() {
        evalns.push(nn.get_output(state.move_to_index(m)));
    }
    */

    let mut move_idxs = Vec::with_capacity(moves.len());

    /*
    let mut hidden: [f32; nn::NUMBER_FEATURES + 32] = unsafe { std::mem::MaybeUninit::uninit().assume_init() };
    */

    for m in 0..moves.len() {
        move_idxs.push(state.move_to_index(&moves[m]));
        evalns.push(0.);
    }

    /*
    hidden[..32].copy_from_slice(&HIDDEN_BIAS);
    hidden[32..].copy_from_slice(features);

    for f in 0..features.len() {
        if features[f] > 0.5 {
            for h in 0..HIDDEN_WEIGHTS.len() {
                hidden[h] += HIDDEN_WEIGHTS[h][f];
            }
        }
    }

    for h in 0..hidden.len() {
        if hidden[h] > 0.5 {
            for m in 0..moves.len() {
                evalns[m] += POLICY_WEIGHTS[move_idxs[m]][h];
            }
        }
    }

    */

    for f in 0..features.len() {
        if features[f] > 0.5 {
            for m in 0..moves.len() {
                evalns[m] += POLICY_WEIGHTS[move_idxs[m]][f];
            }
        }
    }

    softmax(&mut evalns);

    /*
    let gcf = get_policy_good_capture_factor();
    let bcf = get_policy_bad_capture_factor();

    for m in 0..moves.len() {
        if let Some(captured) = state
            .board()
            .piece_on(moves[m].get_dest())
            .map(piece_eval)
        {
            let moved = piece_eval(state.board().piece_on(moves[m].get_source()).unwrap());
            evalns[m] *= if captured > moved { gcf } else { bcf };
        }
    }
    */

    evalns
}

pub fn softmax(arr: &mut [f32]) {
    let t = get_policy_softmax_temp();

    for x in arr.iter_mut() {
        *x = fastapprox::faster::exp(*x / t);
    }
    let s = 1.0 / arr.iter().sum::<f32>();
    for x in arr.iter_mut() {
        *x *= s;
    }
}

use chess::*;
use math;
use mcts::{Evaluator, GameState};
use search::{GooseMcts, SCALE};
use shakmaty::Position;
use shakmaty_syzygy::Wdl;
use state::{self, MoveList, Player, State};
use std::mem::{self, MaybeUninit};
use std::ptr;
use tablebase::probe_tablebase_wdl;

const MATE_FACTOR: f32 = 1.1;

pub struct GooseEval;

impl GooseEval {
    pub fn new() -> Self {
        Self
    }
}

impl Evaluator<GooseMcts> for GooseEval {
    fn evaluate_new_state(&self, state: &State, moves: &MoveList) -> (Vec<f32>, i64, bool) {
        let features = state.features();
        let move_evaluations = evaluate_policy(state, &features, moves);
        let mut tb_hit = false;
        let state_evaluation = if moves.is_empty() {
            let x = (MATE_FACTOR * SCALE) as i64;
            if state.shakmaty_board().is_check() {
                match state.current_player() {
                    chess::Color::White => -x,
                    chess::Color::Black => x,
                }
            } else {
                0
            }
        } else if let Some(wdl) = probe_tablebase_wdl(state.shakmaty_board()) {
            tb_hit = true;
            let win_score = SCALE as i64;
            match (wdl, state.board().side_to_move()) {
                (Wdl::Win, Color::White) => win_score,
                (Wdl::Loss, Color::White) => -win_score,
                (Wdl::Win, Color::Black) => -win_score,
                (Wdl::Loss, Color::Black) => win_score,
                _ => 0,
            }
        } else {
            (evaluate_state(state, &features) * SCALE as f32) as i64
        };
        (move_evaluations, state_evaluation, tb_hit)
    }

    fn interpret_evaluation_for_player(&self, evaln: &i64, player: &Player) -> i64 {
        match *player {
            Color::White => *evaln,
            Color::Black => -*evaln,
        }
    }
}

const STATE_NUMBER_INPUTS: usize = state::NUMBER_FEATURES;
const NUMBER_HIDDEN: usize = 128;
const NUMBER_OUTPUTS: usize = 1;

#[allow(clippy::excessive_precision)]
static EVAL_HIDDEN_BIAS: [f32; NUMBER_HIDDEN] = include!("model/hidden_bias_0");

#[allow(clippy::excessive_precision)]
static EVAL_HIDDEN_WEIGHTS: [[f32; STATE_NUMBER_INPUTS]; NUMBER_HIDDEN] =
    include!("model/hidden_weights_0");

#[allow(clippy::excessive_precision)]
static EVAL_OUTPUT_WEIGHTS: [[f32; NUMBER_HIDDEN]; NUMBER_OUTPUTS] =
    include!("model/output_weights");

fn evaluate_state(state: &State, features: &[f32; state::NUMBER_FEATURES]) -> f32 {
    let mut hidden_layer: [f32; NUMBER_HIDDEN] = unsafe {
        let mut out: [MaybeUninit<f32>; NUMBER_HIDDEN] = MaybeUninit::uninit().assume_init();

        ptr::copy_nonoverlapping(
            EVAL_HIDDEN_BIAS.as_ptr(),
            out.as_mut_ptr().cast::<f32>(),
            NUMBER_HIDDEN,
        );

        mem::transmute(out)
    };

    hidden_layer.copy_from_slice(&EVAL_HIDDEN_BIAS);

    for (i, f) in features.iter().enumerate() {
        if *f > 0.5 {
            for j in 0..hidden_layer.len() {
                hidden_layer[j] += EVAL_HIDDEN_WEIGHTS[j][i];
            }
        }
    }

    let mut result = 0.;

    let weights = EVAL_OUTPUT_WEIGHTS[0];

    for i in 0..hidden_layer.len() {
        result += weights[i] * hidden_layer[i].max(0.);
    }

    result = result.tanh();

    if state.board().side_to_move() == Color::Black {
        result = -result;
    }

    result
}

const POLICY_NUMBER_INPUTS: usize = state::NUMBER_FEATURES;

#[allow(clippy::excessive_precision)]
static POLICY_WEIGHTS: [[f32; POLICY_NUMBER_INPUTS]; 1792] = include!("policy/output_weights");

fn evaluate_policy(
    state: &State,
    features: &[f32; state::NUMBER_FEATURES],
    moves: &MoveList,
) -> Vec<f32> {
    let mut evalns = Vec::with_capacity(moves.len());

    if moves.is_empty() {
        return evalns;
    }

    let mut move_idxs = Vec::with_capacity(moves.len());

    for m in 0..moves.len() {
        move_idxs.push(state.move_to_index(&moves[m]));
        evalns.push(0.);
    }

    for (i, f) in features.iter().enumerate() {
        if *f > 0.5 {
            for m in 0..moves.len() {
                evalns[m] += POLICY_WEIGHTS[move_idxs[m]][i];
            }
        }
    }

    math::softmax(&mut evalns);

    evalns
}

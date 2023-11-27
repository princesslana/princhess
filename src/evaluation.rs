use shakmaty::{MoveList, Position};

use crate::math;
use crate::search::SCALE;
use crate::state::{self, State};
use crate::tablebase::{self, Wdl};

#[derive(Clone, Copy, Debug)]
pub enum Flag {
    Standard,
    #[allow(dead_code)] // Turns out we don't need it, but feels incomplete without it
    TerminalWin,
    TerminalDraw,
    TerminalLoss,
    TablebaseWin,
    TablebaseDraw,
    TablebaseLoss,
}

impl Flag {
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            Flag::TerminalWin | Flag::TerminalDraw | Flag::TerminalLoss
        )
    }

    pub fn is_tablebase(self) -> bool {
        matches!(
            self,
            Flag::TablebaseWin | Flag::TablebaseDraw | Flag::TablebaseLoss
        )
    }
}

pub fn evaluate_state(state: &State) -> i64 {
    (run_eval_net(state) * SCALE) as i64
}

pub fn evaluate_state_flag(state: &State, moves: &MoveList) -> Flag {
    if moves.is_empty() {
        if state.board().is_check() {
            Flag::TerminalLoss
        } else {
            Flag::TerminalDraw
        }
    } else if let Some(wdl) = tablebase::probe_wdl(state.board()) {
        match wdl {
            Wdl::Win => Flag::TablebaseWin,
            Wdl::Loss => Flag::TablebaseLoss,
            Wdl::Draw => Flag::TablebaseDraw,
        }
    } else {
        Flag::Standard
    }
}

pub fn evaluate_policy(state: &State, moves: &MoveList, t: f32) -> Vec<f32> {
    run_policy_net(state, moves, t)
}

const QAB: f32 = 256. * 256.;

const STATE_NUMBER_INPUTS: usize = 768 * 2;
const NUMBER_HIDDEN: usize = 128;
const NUMBER_OUTPUTS: usize = 1;

static EVAL_HIDDEN_BIAS: [i32; NUMBER_HIDDEN] = include!("model/hidden_bias");

static EVAL_HIDDEN_WEIGHTS: [[i16; NUMBER_HIDDEN]; STATE_NUMBER_INPUTS / 2] =
    include!("model/hidden_weights");

static EVAL_OUTPUT_WEIGHTS: [[i16; NUMBER_HIDDEN * 2]; NUMBER_OUTPUTS] =
    include!("model/output_weights");

static EVAL_OUTPUT_BIAS: [i16; NUMBER_OUTPUTS] = include!("model/output_bias");

const POLICY_NUMBER_INPUTS: usize = state::NUMBER_FEATURES;

#[allow(clippy::excessive_precision, clippy::unreadable_literal)]
static POLICY_WEIGHTS: [[f32; POLICY_NUMBER_INPUTS]; 384] = include!("policy/output_weights");

fn run_eval_net(state: &State) -> f32 {
    let mut hidden_layer = [0; NUMBER_HIDDEN * 2];

    hidden_layer[..NUMBER_HIDDEN].copy_from_slice(&EVAL_HIDDEN_BIAS);
    hidden_layer[NUMBER_HIDDEN..].copy_from_slice(&EVAL_HIDDEN_BIAS);

    state.state_features_map(|idx| {
        if idx < 768 {
            for (j, l) in hidden_layer[..NUMBER_HIDDEN].iter_mut().enumerate() {
                *l += i32::from(EVAL_HIDDEN_WEIGHTS[idx][j]);
            }
        } else {
            for (j, l) in hidden_layer[NUMBER_HIDDEN..].iter_mut().enumerate() {
                *l += i32::from(EVAL_HIDDEN_WEIGHTS[idx - 768][j]);
            }
        }
    });

    let mut result: i32 = i32::from(EVAL_OUTPUT_BIAS[0]);
    let weights = EVAL_OUTPUT_WEIGHTS[0];

    for i in 0..hidden_layer.len() {
        result += i32::from(weights[i]) * hidden_layer[i].max(0);
    }

    (result as f32 / QAB).tanh()
}

fn run_policy_net(state: &State, moves: &MoveList, t: f32) -> Vec<f32> {
    let mut evalns = Vec::with_capacity(moves.len());

    if moves.is_empty() {
        return evalns;
    }

    let mut move_idxs = Vec::with_capacity(moves.len());

    for m in 0..moves.len() {
        move_idxs.push(state.move_to_index(&moves[m]));
        evalns.push(0.);
    }

    state.policy_features_map(|idx| {
        for m in 0..moves.len() {
            evalns[m] += POLICY_WEIGHTS[move_idxs[m]][idx];
        }
    });

    math::softmax(&mut evalns, t);

    evalns
}

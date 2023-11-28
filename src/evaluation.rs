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

const HIDDEN: usize = 256;
const QA: i32 = 256;
const QB: i32 = 256;
const QAB: i32 = QA * QB;

const STATE_NUMBER_INPUTS: usize = state::NUMBER_FEATURES;

#[repr(C)]
struct EvalNet {
    hidden_weights: [Accumulator; STATE_NUMBER_INPUTS],
    hidden_bias: Accumulator,
    output_weights: Accumulator,
}

static EVAL_HIDDEN_WEIGHTS:  [[i16; HIDDEN]; STATE_NUMBER_INPUTS] = include!("model/hidden_weights");
static EVAL_HIDDEN_BIAS: [i16; HIDDEN] = include!("model/hidden_bias");
static EVAL_OUTPUT_WEIGHTS: [[i16; HIDDEN]; 1] = include!("model/output_weights");

static EVAL_NET: EvalNet = EvalNet {
    hidden_weights: unsafe { std::mem::transmute(EVAL_HIDDEN_WEIGHTS) },
    hidden_bias: unsafe { std::mem::transmute(EVAL_HIDDEN_BIAS) },
    output_weights: unsafe { std::mem::transmute(EVAL_OUTPUT_WEIGHTS) },
};

#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct Accumulator {
    vals: [i16; HIDDEN],
}

impl Accumulator {
    pub fn set(&mut self, idx: usize) {
        for (i, d) in self.vals.iter_mut().zip(&EVAL_NET.hidden_weights[idx].vals) {
            *i += *d;
        }
    }
}

impl Default for Accumulator {
    fn default() -> Self {
        EVAL_NET.hidden_bias
    }
}

fn activate(x: i16) -> i32 {
    i32::from(x).max(0)
}


fn run_eval_net(state: &State) -> f32 {
    let mut acc = Accumulator::default();

    state.state_features_map(|idx| {
        acc.set(idx);
    });

    let mut result: i32 = 0;

    for (&x, &w) in acc.vals.iter().zip(&EVAL_NET.output_weights.vals) {
        result += activate(x) * i32::from(w);
    }

    (result as f32 / QAB as f32).tanh()
}

const POLICY_NUMBER_INPUTS: usize = state::NUMBER_FEATURES;

#[allow(clippy::excessive_precision, clippy::unreadable_literal)]
static POLICY_WEIGHTS: [[f32; POLICY_NUMBER_INPUTS]; 384] = include!("policy/output_weights");

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

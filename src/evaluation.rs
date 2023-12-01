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
    hidden_weights: [Accumulator<HIDDEN>; STATE_NUMBER_INPUTS],
    hidden_bias: Accumulator<HIDDEN>,
    output_weights: Accumulator<HIDDEN>,
}

static EVAL_HIDDEN_WEIGHTS: [[i16; HIDDEN]; STATE_NUMBER_INPUTS] = include!("model/hidden_weights");
static EVAL_HIDDEN_BIAS: [i16; HIDDEN] = include!("model/hidden_bias");
static EVAL_OUTPUT_WEIGHTS: [[i16; HIDDEN]; 1] = include!("model/output_weights");

static EVAL_NET: EvalNet = EvalNet {
    hidden_weights: unsafe { std::mem::transmute(EVAL_HIDDEN_WEIGHTS) },
    hidden_bias: unsafe { std::mem::transmute(EVAL_HIDDEN_BIAS) },
    output_weights: unsafe { std::mem::transmute(EVAL_OUTPUT_WEIGHTS) },
};

#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct Accumulator<const H: usize> {
    vals: [i16; H],
}

impl<const H: usize> Accumulator<H> {
    pub fn eval_hidden() -> Accumulator<HIDDEN> {
        EVAL_NET.hidden_bias
    }

    pub fn policy_left() -> Accumulator<POLICY_NUMBER_OUTPUTS> {
        POLICY_NET.left_bias
    }

    pub fn policy_right() -> Accumulator<POLICY_NUMBER_OUTPUTS> {
        POLICY_NET.right_bias
    }

    pub fn set(&mut self, weights: &Accumulator<H>) {
        for (i, d) in self.vals.iter_mut().zip(&weights.vals) {
            *i += *d;
        }
    }
}

fn activate(x: i16) -> i32 {
    i32::from(x).max(0)
}

fn run_eval_net(state: &State) -> f32 {
    let mut acc = Accumulator::<HIDDEN>::eval_hidden();

    state.state_features_map(|idx| {
        acc.set(&EVAL_NET.hidden_weights[idx]);
    });

    let mut result: i32 = 0;

    for (&x, &w) in acc.vals.iter().zip(&EVAL_NET.output_weights.vals) {
        result += activate(x) * i32::from(w);
    }

    (result as f32 / QAB as f32).tanh()
}

const POLICY_NUMBER_INPUTS: usize = state::NUMBER_FEATURES;
const POLICY_NUMBER_OUTPUTS: usize = 384;

#[repr(C)]
struct PolicyNet {
    left_weights: [Accumulator<POLICY_NUMBER_OUTPUTS>; POLICY_NUMBER_INPUTS],
    left_bias: Accumulator<POLICY_NUMBER_OUTPUTS>,
    right_weights: [Accumulator<POLICY_NUMBER_OUTPUTS>; POLICY_NUMBER_INPUTS],
    right_bias: Accumulator<POLICY_NUMBER_OUTPUTS>,
}

static POLICY_LEFT_WEIGHTS: [[i16; POLICY_NUMBER_OUTPUTS]; POLICY_NUMBER_INPUTS] =
    include!("policy/left_weights");
static POLICY_LEFT_BIAS: [i16; POLICY_NUMBER_OUTPUTS] = include!("policy/left_bias");
static POLICY_RIGHT_WEIGHTS: [[i16; POLICY_NUMBER_OUTPUTS]; POLICY_NUMBER_INPUTS] =
    include!("policy/right_weights");
static POLICY_RIGHT_BIAS: [i16; POLICY_NUMBER_OUTPUTS] = include!("policy/right_bias");

static POLICY_NET: PolicyNet = PolicyNet {
    left_weights: unsafe { std::mem::transmute(POLICY_LEFT_WEIGHTS) },
    left_bias: unsafe { std::mem::transmute(POLICY_LEFT_BIAS) },
    right_weights: unsafe { std::mem::transmute(POLICY_RIGHT_WEIGHTS) },
    right_bias: unsafe { std::mem::transmute(POLICY_RIGHT_BIAS) },
};

fn run_policy_net(state: &State, moves: &MoveList, t: f32) -> Vec<f32> {
    let mut evalns = Vec::with_capacity(moves.len());

    if moves.is_empty() {
        return evalns;
    }

    let mut acc_left = Accumulator::<POLICY_NUMBER_OUTPUTS>::policy_left();
    let mut acc_right = Accumulator::<POLICY_NUMBER_OUTPUTS>::policy_right();

    state.policy_features_map(|idx| {
        acc_left.set(&POLICY_NET.left_weights[idx]);
        acc_right.set(&POLICY_NET.right_weights[idx]);
    });

    for m in moves {
        let move_idx = state.move_to_index(m);
        let logit = activate(acc_left.vals[move_idx]) * activate(acc_right.vals[move_idx]);
        evalns.push(logit as f32 / QAB as f32);
    }

    math::softmax(&mut evalns, t);

    evalns
}

use crate::chess::MoveList;
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

pub fn evaluate_value(state: &State) -> i64 {
    (run_value_net(state) * SCALE) as i64
}

pub fn evaluate_policy(state: &State, moves: &MoveList, t: f32) -> Vec<f32> {
    run_policy_net(state, moves, t)
}

pub fn evaluate_state_flag(state: &State, is_legal_moves: bool) -> Flag {
    if !is_legal_moves {
        if state.is_check() {
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

const HIDDEN: usize = 384;
const QA: i32 = 256;
const QB: i32 = 256;
const QAB: i32 = QA * QB;

const VALUE_NUMBER_INPUTS: usize = 2048;

#[repr(C)]
struct ValueNetwork {
    hidden_weights: [Accumulator<HIDDEN>; VALUE_NUMBER_INPUTS],
    hidden_bias: Accumulator<HIDDEN>,
    output_weights: Accumulator<HIDDEN>,
    output_bias: i32,
}

static VALUE_HIDDEN_WEIGHTS: [[i16; HIDDEN]; VALUE_NUMBER_INPUTS] =
    include!("value/hidden_weights");
static VALUE_HIDDEN_BIAS: [i16; HIDDEN] = include!("value/hidden_bias");
static VALUE_OUTPUT_WEIGHTS: [[i16; HIDDEN]; 1] = include!("value/output_weights");
static VALUE_OUTPUT_BIAS: i32 = include!("value/output_bias")[0];

static VALUE_NETWORK: ValueNetwork = ValueNetwork {
    hidden_weights: unsafe { std::mem::transmute(VALUE_HIDDEN_WEIGHTS) },
    hidden_bias: unsafe { std::mem::transmute(VALUE_HIDDEN_BIAS) },
    output_weights: unsafe { std::mem::transmute(VALUE_OUTPUT_WEIGHTS) },
    output_bias: VALUE_OUTPUT_BIAS,
};

#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct Accumulator<const H: usize> {
    vals: [i16; H],
}

impl<const H: usize> Accumulator<H> {
    pub fn set(&mut self, weights: &Accumulator<H>) {
        for (i, d) in self.vals.iter_mut().zip(&weights.vals) {
            *i += *d;
        }
    }
}

fn relu(x: i16) -> i32 {
    i32::from(x).max(0)
}

fn run_value_net(state: &State) -> f32 {
    let mut acc = VALUE_NETWORK.hidden_bias;

    state.value_features_map(|idx| {
        acc.set(&VALUE_NETWORK.hidden_weights[idx]);
    });

    let mut result: i32 = VALUE_NETWORK.output_bias;

    for (&x, &w) in acc.vals.iter().zip(&VALUE_NETWORK.output_weights.vals) {
        result += relu(x) * i32::from(w);
    }

    (result as f32 / QAB as f32).tanh()
}

const POLICY_NUMBER_INPUTS: usize = state::POLICY_NUMBER_FEATURES;
const POLICY_NUMBER_OUTPUTS: usize = 384;

#[repr(C)]
struct PolicyNetwork {
    left_weights: [Accumulator<POLICY_NUMBER_OUTPUTS>; POLICY_NUMBER_INPUTS],
    left_bias: Accumulator<POLICY_NUMBER_OUTPUTS>,
    right_weights: [Accumulator<POLICY_NUMBER_OUTPUTS>; POLICY_NUMBER_INPUTS],
    right_bias: Accumulator<POLICY_NUMBER_OUTPUTS>,
    constant_weights: [Accumulator<POLICY_NUMBER_OUTPUTS>; POLICY_NUMBER_INPUTS],
    constant_bias: Accumulator<POLICY_NUMBER_OUTPUTS>,
}

static POLICY_LEFT_WEIGHTS: [[i16; POLICY_NUMBER_OUTPUTS]; POLICY_NUMBER_INPUTS] =
    include!("policy/left_weights");
static POLICY_LEFT_BIAS: [i16; POLICY_NUMBER_OUTPUTS] = include!("policy/left_bias");
static POLICY_RIGHT_WEIGHTS: [[i16; POLICY_NUMBER_OUTPUTS]; POLICY_NUMBER_INPUTS] =
    include!("policy/right_weights");
static POLICY_RIGHT_BIAS: [i16; POLICY_NUMBER_OUTPUTS] = include!("policy/right_bias");
static POLICY_CONSTANT_WEIGHTS: [[i16; POLICY_NUMBER_OUTPUTS]; POLICY_NUMBER_INPUTS] =
    include!("policy/constant_weights");
static POLICY_CONSTANT_BIAS: [i16; POLICY_NUMBER_OUTPUTS] = include!("policy/constant_bias");

static POLICY_NET: PolicyNetwork = PolicyNetwork {
    left_weights: unsafe { std::mem::transmute(POLICY_LEFT_WEIGHTS) },
    left_bias: unsafe { std::mem::transmute(POLICY_LEFT_BIAS) },
    right_weights: unsafe { std::mem::transmute(POLICY_RIGHT_WEIGHTS) },
    right_bias: unsafe { std::mem::transmute(POLICY_RIGHT_BIAS) },
    constant_weights: unsafe { std::mem::transmute(POLICY_CONSTANT_WEIGHTS) },
    constant_bias: unsafe { std::mem::transmute(POLICY_CONSTANT_BIAS) },
};

fn run_policy_net(state: &State, moves: &MoveList, t: f32) -> Vec<f32> {
    let mut evalns = Vec::with_capacity(moves.len());

    if moves.is_empty() {
        return evalns;
    }

    let mut move_idxs = Vec::with_capacity(moves.len());
    let mut acc = Vec::with_capacity(moves.len());

    for m in moves {
        let move_idx = state.move_to_index(*m);
        move_idxs.push(move_idx);
        acc.push((
            POLICY_NET.constant_bias.vals[move_idx],
            POLICY_NET.left_bias.vals[move_idx],
            POLICY_NET.right_bias.vals[move_idx],
        ));
    }

    state.policy_features_map(|idx| {
        let cw = &POLICY_NET.constant_weights[idx];
        let lw = &POLICY_NET.left_weights[idx];
        let rw = &POLICY_NET.right_weights[idx];

        for (&move_idx, (c, l, r)) in move_idxs.iter().zip(acc.iter_mut()) {
            *c += cw.vals[move_idx];
            *l += lw.vals[move_idx];
            *r += rw.vals[move_idx];
        }
    });

    for (c, l, r) in &acc {
        let logit = QA * i32::from(*c) + i32::from(*l) * relu(*r);
        evalns.push(logit as f32 / QAB as f32);
    }

    math::softmax(&mut evalns, t);

    evalns
}

use shakmaty::{MoveList, Position};
use shakmaty_syzygy::Wdl;
use std::mem::{self, MaybeUninit};
use std::ptr;

use crate::math;
use crate::search::SCALE;
use crate::state::{self, State};
use crate::tablebase::probe_tablebase_wdl;

const MATE: Evaluation = Evaluation::Terminal(SCALE as i64);
const DRAW: Evaluation = Evaluation::Terminal(0);

const TB_WIN: Evaluation = Evaluation::Tablebase(SCALE as i64);
const TB_LOSS: Evaluation = Evaluation::Tablebase(-SCALE as i64);
const TB_DRAW: Evaluation = Evaluation::Tablebase(0);

#[derive(Debug, Copy, Clone)]
pub enum Evaluation {
    Scaled(i64),
    Tablebase(i64),
    Terminal(i64),
}

impl Evaluation {
    pub const fn draw() -> Self {
        DRAW
    }

    pub fn flip(&self) -> Self {
        match self {
            Evaluation::Scaled(s) => Evaluation::Scaled(-s),
            Evaluation::Tablebase(s) => Evaluation::Tablebase(-s),
            Evaluation::Terminal(s) => Evaluation::Terminal(-s),
        }
    }

    pub fn is_terminal(&self) -> bool {
        matches!(self, Evaluation::Terminal(_))
    }

    pub fn is_tablebase(&self) -> bool {
        matches!(self, Evaluation::Tablebase(_))
    }
}

impl From<f32> for Evaluation {
    fn from(f: f32) -> Self {
        Evaluation::Scaled((f * SCALE) as i64)
    }
}

impl From<Evaluation> for i64 {
    fn from(e: Evaluation) -> Self {
        match e {
            Evaluation::Scaled(s) | Evaluation::Tablebase(s) | Evaluation::Terminal(s) => s,
        }
    }
}

pub fn evaluate_state(state: &State) -> Evaluation {
    let state_evaluation = Evaluation::from(run_eval_net(state));
    state
        .side_to_move()
        .fold_wb(state_evaluation, state_evaluation.flip())
}

pub fn evaluate_policy(state: &State, moves: &MoveList) -> (Vec<f32>, Evaluation) {
    let move_evaluations = run_policy_net(state, moves);

    let state_evaluation = if moves.is_empty() {
        if state.board().is_check() {
            MATE.flip()
        } else {
            DRAW
        }
    } else if let Some(wdl) = probe_tablebase_wdl(state.board()) {
        match wdl {
            Wdl::Win => TB_WIN,
            Wdl::Loss => TB_LOSS,
            _ => TB_DRAW,
        }
    } else {
        Evaluation::Scaled(0)
    };

    (
        move_evaluations,
        state
            .side_to_move()
            .fold_wb(state_evaluation, state_evaluation.flip()),
    )
}

const STATE_NUMBER_INPUTS: usize = state::NUMBER_FEATURES;
const NUMBER_HIDDEN: usize = 192;
const NUMBER_OUTPUTS: usize = 1;

#[allow(clippy::excessive_precision, clippy::unreadable_literal)]
static EVAL_HIDDEN_BIAS: [f32; NUMBER_HIDDEN] = include!("model/hidden_bias_0");

#[allow(clippy::excessive_precision, clippy::unreadable_literal)]
static EVAL_HIDDEN_WEIGHTS: [[f32; NUMBER_HIDDEN]; STATE_NUMBER_INPUTS] =
    include!("model/hidden_weights_0");

#[allow(clippy::excessive_precision, clippy::unreadable_literal)]
static EVAL_OUTPUT_WEIGHTS: [[f32; NUMBER_HIDDEN]; NUMBER_OUTPUTS] =
    include!("model/output_weights");

const POLICY_NUMBER_INPUTS: usize = state::NUMBER_FEATURES;

#[allow(clippy::excessive_precision, clippy::unreadable_literal)]
static POLICY_WEIGHTS: [[f32; POLICY_NUMBER_INPUTS]; 384] = include!("policy/output_weights");

fn run_eval_net(state: &State) -> f32 {
    let mut hidden_layer: [f32; NUMBER_HIDDEN] = unsafe {
        let mut out: [MaybeUninit<f32>; NUMBER_HIDDEN] = MaybeUninit::uninit().assume_init();

        ptr::copy_nonoverlapping(
            EVAL_HIDDEN_BIAS.as_ptr(),
            out.as_mut_ptr().cast::<f32>(),
            NUMBER_HIDDEN,
        );

        mem::transmute(out)
    };

    state.features_map(|idx| {
        for (j, l) in hidden_layer.iter_mut().enumerate() {
            *l += EVAL_HIDDEN_WEIGHTS[idx][j];
        }
    });

    let mut result = 0.;
    let weights = EVAL_OUTPUT_WEIGHTS[0];

    for i in 0..hidden_layer.len() {
        result += weights[i] * hidden_layer[i].max(0.);
    }

    result.tanh()
}

fn run_policy_net(state: &State, moves: &MoveList) -> Vec<f32> {
    let mut evalns = Vec::with_capacity(moves.len());

    if moves.is_empty() {
        return evalns;
    }

    let mut move_idxs = Vec::with_capacity(moves.len());

    for m in 0..moves.len() {
        move_idxs.push(state.move_to_index(&moves[m]));
        evalns.push(0.);
    }

    state.features_map(|idx| {
        for m in 0..moves.len() {
            evalns[m] += POLICY_WEIGHTS[move_idxs[m]][idx];
        }
    });

    math::softmax(&mut evalns);

    evalns
}

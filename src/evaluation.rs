use shakmaty::{MoveList, Position};
use shakmaty_syzygy::Wdl;
use std::mem::{self, MaybeUninit};
use std::ptr;

use crate::math;
use crate::search::SCALE;
use crate::state::{self, State};
use crate::tablebase::probe_tablebase_wdl;

const MATE_FACTOR: f32 = 1.1;

const MATE: StateEvaluation = StateEvaluation::Scaled((SCALE * MATE_FACTOR) as i64);
const DRAW: StateEvaluation = StateEvaluation::Scaled(0);

const TB_WIN: StateEvaluation = StateEvaluation::Tablebase(SCALE as i64);
const TB_LOSS: StateEvaluation = StateEvaluation::Tablebase(-SCALE as i64);
const TB_DRAW: StateEvaluation = StateEvaluation::Tablebase(0);

#[derive(Debug, Copy, Clone)]
pub enum StateEvaluation {
    Scaled(i64),
    Tablebase(i64),
}

impl StateEvaluation {
    pub fn draw() -> Self {
        DRAW
    }

    pub fn flip(&self) -> Self {
        match self {
            StateEvaluation::Scaled(s) => StateEvaluation::Scaled(-s),
            StateEvaluation::Tablebase(s) => StateEvaluation::Tablebase(-s),
        }
    }

    pub fn is_tablebase(&self) -> bool {
        matches!(self, StateEvaluation::Tablebase(_))
    }
}

impl From<f32> for StateEvaluation {
    fn from(f: f32) -> Self {
        StateEvaluation::Scaled((f * SCALE) as i64)
    }
}

impl From<StateEvaluation> for i64 {
    fn from(e: StateEvaluation) -> Self {
        match e {
            StateEvaluation::Scaled(s) => s,
            StateEvaluation::Tablebase(s) => s,
        }
    }
}

pub fn evaluate_new_state(state: &State, moves: &MoveList) -> (Vec<f32>, StateEvaluation) {
    let (state_evaluation, move_evaluations) = run_nets(state, moves);

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
        StateEvaluation::from(state_evaluation)
    };

    (
        move_evaluations,
        state
            .side_to_move()
            .fold_wb(state_evaluation, state_evaluation.flip()),
    )
}

const STATE_NUMBER_INPUTS: usize = state::NUMBER_FEATURES;
const NUMBER_HIDDEN: usize = 128;
const NUMBER_OUTPUTS: usize = 1;

#[allow(clippy::excessive_precision)]
static EVAL_HIDDEN_BIAS: [f32; NUMBER_HIDDEN] = include!("model/hidden_bias_0");

#[allow(clippy::excessive_precision)]
static EVAL_HIDDEN_WEIGHTS: [[f32; NUMBER_HIDDEN]; STATE_NUMBER_INPUTS] =
    include!("model/hidden_weights_0");

#[allow(clippy::excessive_precision)]
static EVAL_OUTPUT_WEIGHTS: [[f32; NUMBER_HIDDEN]; NUMBER_OUTPUTS] =
    include!("model/output_weights");

const POLICY_NUMBER_INPUTS: usize = state::NUMBER_FEATURES;

#[allow(clippy::excessive_precision)]
static POLICY_WEIGHTS: [[f32; POLICY_NUMBER_INPUTS]; 384] = include!("policy/output_weights");

fn run_nets(state: &State, moves: &MoveList) -> (f32, Vec<f32>) {
    let mut evalns = Vec::with_capacity(moves.len());

    if moves.is_empty() {
        // Returning 0 is ok here, as we'll immediately check for why there's no moves
        return (0., evalns);
    }

    let mut move_idxs = Vec::with_capacity(moves.len());

    for m in 0..moves.len() {
        move_idxs.push(state.move_to_index(&moves[m]));
        evalns.push(0.);
    }

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

    state.features_map(|idx| {
        for (j, l) in hidden_layer.iter_mut().enumerate() {
            *l += EVAL_HIDDEN_WEIGHTS[idx][j]
        }

        for m in 0..moves.len() {
            evalns[m] += POLICY_WEIGHTS[move_idxs[m]][idx];
        }
    });

    math::softmax(&mut evalns);

    let mut result = 0.;
    let weights = EVAL_OUTPUT_WEIGHTS[0];

    for i in 0..hidden_layer.len() {
        result += weights[i] * hidden_layer[i].max(0.);
    }

    (result.tanh(), evalns)
}

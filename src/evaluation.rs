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

const HIDDEN: usize = 768;
const QA: i32 = 255;
const QB: i32 = 64;
const QAB: i32 = QA * QB;

const POLICY_NUMBER_INPUTS: usize = state::NUMBER_FEATURES;

#[allow(clippy::excessive_precision, clippy::unreadable_literal)]
static POLICY_WEIGHTS: [[f32; POLICY_NUMBER_INPUTS]; 384] = include!("policy/output_weights");
#[repr(C)]
struct EvalNet {
    hidden_weights: [Accumulator; 768],
    hidden_bias: Accumulator,
    output_weights: [Accumulator; 2],
    output_bias: i16,
}

/*
static EVAL_NET: Lazy<EvalNet> = Lazy::new(|| EvalNet::from_slices(
    include!("model/hidden_weights"),
    include!("model/hidden_bias"),
    include!("model/output_weights"),
    include!("model/output_bias")[0],
));
*/

static EVAL_NET: EvalNet = unsafe { std::mem::transmute(*include_bytes!("model/altair-net.bin")) };

impl EvalNet {
    pub fn from_slices(
        hw: [[i16; HIDDEN]; 768],
        hb: [i16; HIDDEN],
        ow: [[i16; HIDDEN * 2]; 1],
        ob: i16,
    ) -> Self {
        let mut hidden_weights = [Accumulator::default(); 768];
        for (i, h) in hidden_weights.iter_mut().enumerate() {
            h.vals.copy_from_slice(&hw[i]);
        }

        let mut output_weights = [Accumulator::default(); 2];
        output_weights[0].vals.copy_from_slice(&ow[0][..HIDDEN]);
        output_weights[1].vals.copy_from_slice(&ow[0][HIDDEN..]);

        Self {
            hidden_weights,
            hidden_bias: Accumulator { vals: hb },
            output_weights,
            output_bias: ob,
        }
    }
}

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
    i32::from(x).clamp(0, QA)
}

fn run_eval_net(state: &State) -> f32 {
    let mut acc_stm = Accumulator::default();
    let mut acc_nstm = Accumulator::default();

    state.state_features_map(|idx| {
        if idx < 768 {
            acc_stm.set(idx);
        } else {
            acc_nstm.set(idx - 768);
        }
    });

    let mut result: i32 = i32::from(EVAL_NET.output_bias);

    for (&x, &w) in acc_stm.vals.iter().zip(&EVAL_NET.output_weights[0].vals) {
        result += activate(x) * i32::from(w);
    }

    for (&x, &w) in acc_nstm.vals.iter().zip(&EVAL_NET.output_weights[1].vals) {
        result += activate(x) * i32::from(w);
    }

    //_(result as f32 / QAB).tanh()
    let cp = (result * 400 / QAB) as f32;

    let uniform = 1.0 / (1.0 + (-cp / 400.).exp());

    uniform * 2. - 1.
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

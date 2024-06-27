use crate::chess::MoveList;
use crate::math;
use crate::nets::{relu, Accumulator};
use crate::search::SCALE;
use crate::state::{self, State};
use crate::tablebase::{self, Wdl};
use crate::value::QuantizedValueNetwork;

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
    #[must_use]
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            Flag::TerminalWin | Flag::TerminalDraw | Flag::TerminalLoss
        )
    }

    #[must_use]
    pub fn is_tablebase(self) -> bool {
        matches!(
            self,
            Flag::TablebaseWin | Flag::TablebaseDraw | Flag::TablebaseLoss
        )
    }
}

#[cfg(not(feature = "no-net"))]
static VALUE_NETWORK: QuantizedValueNetwork =
    unsafe { std::mem::transmute(*include_bytes!("nets/value.bin")) };

#[must_use]
#[cfg(not(feature = "no-net"))]
pub fn value(state: &State) -> i64 {
    (VALUE_NETWORK.get(state) * SCALE) as i64
}

#[must_use]
#[cfg(feature = "no-net")]
pub fn value(_state: &State) -> i64 {
    0
}

#[must_use]
pub fn evaluate_policy(state: &State, moves: &MoveList, t: f32) -> Vec<f32> {
    run_policy_net(state, moves, t)
}

#[must_use]
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

const QA: i32 = 256;
const QAA: i32 = QA * QA;

const POLICY_NUMBER_INPUTS: usize = state::POLICY_NUMBER_FEATURES;
const POLICY_NUMBER_OUTPUTS: usize = 384;

#[repr(C)]
struct PolicyNet {
    left_weights: [Accumulator<POLICY_NUMBER_OUTPUTS>; POLICY_NUMBER_INPUTS],
    left_bias: Accumulator<POLICY_NUMBER_OUTPUTS>,
    right_weights: [Accumulator<POLICY_NUMBER_OUTPUTS>; POLICY_NUMBER_INPUTS],
    right_bias: Accumulator<POLICY_NUMBER_OUTPUTS>,
    add_weights: [Accumulator<POLICY_NUMBER_OUTPUTS>; POLICY_NUMBER_INPUTS],
    add_bias: Accumulator<POLICY_NUMBER_OUTPUTS>,
}

static POLICY_LEFT_WEIGHTS: [[i16; POLICY_NUMBER_OUTPUTS]; POLICY_NUMBER_INPUTS] =
    include!("policy/left_weights");
static POLICY_LEFT_BIAS: [i16; POLICY_NUMBER_OUTPUTS] = include!("policy/left_bias");
static POLICY_RIGHT_WEIGHTS: [[i16; POLICY_NUMBER_OUTPUTS]; POLICY_NUMBER_INPUTS] =
    include!("policy/right_weights");
static POLICY_RIGHT_BIAS: [i16; POLICY_NUMBER_OUTPUTS] = include!("policy/right_bias");
static POLICY_ADD_WEIGHTS: [[i16; POLICY_NUMBER_OUTPUTS]; POLICY_NUMBER_INPUTS] =
    include!("policy/add_weights");
static POLICY_ADD_BIAS: [i16; POLICY_NUMBER_OUTPUTS] = include!("policy/add_bias");

static POLICY_NET: PolicyNet = PolicyNet {
    left_weights: unsafe { std::mem::transmute(POLICY_LEFT_WEIGHTS) },
    left_bias: unsafe { std::mem::transmute(POLICY_LEFT_BIAS) },
    right_weights: unsafe { std::mem::transmute(POLICY_RIGHT_WEIGHTS) },
    right_bias: unsafe { std::mem::transmute(POLICY_RIGHT_BIAS) },
    add_weights: unsafe { std::mem::transmute(POLICY_ADD_WEIGHTS) },
    add_bias: unsafe { std::mem::transmute(POLICY_ADD_BIAS) },
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
            POLICY_NET.add_bias.vals[move_idx],
            POLICY_NET.left_bias.vals[move_idx],
            POLICY_NET.right_bias.vals[move_idx],
        ));
    }

    state.policy_features_map(|idx| {
        let aw = &POLICY_NET.add_weights[idx];
        let lw = &POLICY_NET.left_weights[idx];
        let rw = &POLICY_NET.right_weights[idx];

        for (&move_idx, (a, l, r)) in move_idxs.iter().zip(acc.iter_mut()) {
            *a += aw.vals[move_idx];
            *l += lw.vals[move_idx];
            *r += rw.vals[move_idx];
        }
    });

    for (a, l, r) in &acc {
        let logit = QA * i32::from(*a) + i32::from(*l) * relu(*r);
        evalns.push(logit as f32 / QAA as f32);
    }

    math::softmax(&mut evalns, t);

    evalns
}

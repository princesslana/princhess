use goober::SparseVector;

use crate::chess::MoveList;
use crate::math;

use crate::policy::QuantizedPolicyNetwork;
use crate::search::SCALE;
use crate::state::State;
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
pub fn policy(state: &State, moves: &MoveList, t: f32) -> Vec<f32> {
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

#[cfg(not(feature = "no-policy-net"))]
static POLICY_NETWORK: QuantizedPolicyNetwork =
    unsafe { std::mem::transmute(*include_bytes!("nets/policy.bin")) };

#[cfg(feature = "no-policy-net")]
fn run_policy_net(_state: &State, moves: &MoveList, _t: f32) -> Vec<f32> {
    let mut evalns = Vec::with_capacity(moves.len());

    for _ in moves {
        evalns.push(1.0 / moves.len() as f32);
    }

    evalns
}

#[cfg(not(feature = "no-policy-net"))]
fn run_policy_net(state: &State, moves: &MoveList, t: f32) -> Vec<f32> {
    let mut evalns = Vec::with_capacity(moves.len());

    if moves.is_empty() {
        return evalns;
    }

    let mut features = SparseVector::with_capacity(32);
    state.policy_features_map(|idx| features.push(idx));

    POLICY_NETWORK.get_all(&features, state.moves_to_indexes(moves), &mut evalns);

    math::softmax(&mut evalns, t);

    evalns
}

use crate::chess::MoveList;
use crate::engine::SCALE;
use crate::math;
use crate::quantized_policy::QuantizedPolicyNetwork;
use crate::quantized_value::QuantizedValueNetwork;
use crate::state::State;
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

    #[must_use]
    pub fn adjust_eval(self, eval: i64) -> i64 {
        match self {
            Flag::TerminalWin => 2 * SCALE as i64,
            Flag::TerminalLoss => -2 * SCALE as i64,
            Flag::TablebaseWin => SCALE as i64,
            Flag::TablebaseLoss => -SCALE as i64,
            Flag::TerminalDraw | Flag::TablebaseDraw => 0,
            Flag::Standard => eval,
        }
    }
}

#[cfg(feature = "value-net")]
static VALUE_NETWORK: QuantizedValueNetwork =
    unsafe { std::mem::transmute(*include_bytes!("nets/value.bin")) };

#[must_use]
#[cfg(feature = "value-net")]
pub fn value(state: &State) -> i64 {
    (VALUE_NETWORK.get(state) * SCALE) as i64
}

#[must_use]
#[cfg(not(feature = "value-net"))]
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

#[cfg(feature = "policy-net")]
static MG_POLICY_NETWORK: QuantizedPolicyNetwork =
    unsafe { std::mem::transmute(*include_bytes!("nets/mg-policy.bin")) };

#[cfg(feature = "policy-net")]
static EG_POLICY_NETWORK: QuantizedPolicyNetwork =
    unsafe { std::mem::transmute(*include_bytes!("nets/eg-policy.bin")) };

#[cfg(not(feature = "policy-net"))]
fn run_policy_net(_state: &State, moves: &MoveList, _t: f32) -> Vec<f32> {
    let mut evalns = Vec::with_capacity(moves.len());

    for _ in moves {
        evalns.push(1.0 / moves.len() as f32);
    }

    evalns
}

#[cfg(feature = "policy-net")]
fn run_policy_net(state: &State, moves: &MoveList, t: f32) -> Vec<f32> {
    let mut evalns = vec![0.0; moves.len()];

    if moves.is_empty() {
        return evalns;
    }

    let network = [&MG_POLICY_NETWORK, &EG_POLICY_NETWORK][usize::from(state.is_endgame())];
    network.get_all(state, state.moves_to_indexes(moves), &mut evalns);

    math::softmax(&mut evalns, t);

    evalns
}

use crate::chess::MoveList;
use crate::engine::SCALE;
#[cfg(feature = "policy-net")]
use crate::math;
use crate::options::EvaluationOptions;
#[cfg(feature = "policy-net")]
use crate::quantized_policy::QuantizedPolicyNetwork;
#[cfg(feature = "value-net")]
use crate::quantized_value::QuantizedValueNetwork;
use crate::state::State;
use crate::tablebase::{self, Wdl};

/// Bitflag representation for position evaluation state.
/// Bits 0-1: outcome (00=standard, 01=loss, 10=win, 11=draw)
/// Bit 2: source (0=terminal, 1=tablebase)
/// Bits 3-7: reserved (must be zero)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Flag(u8);

impl Flag {
    const OUTCOME_MASK: u8 = 0b0000_0011;
    const TABLEBASE_MASK: u8 = 0b0000_0100;
    const RESERVED_MASK: u8 = 0b1111_1000;

    pub const STANDARD: Self = Self(0b0000_0000);
    pub const TERMINAL_LOSS: Self = Self(0b0000_0001);
    pub const TERMINAL_WIN: Self = Self(0b0000_0010);
    pub const TERMINAL_DRAW: Self = Self(0b0000_0011);
    pub const TABLEBASE_LOSS: Self = Self(0b0000_0101);
    pub const TABLEBASE_WIN: Self = Self(0b0000_0110);
    pub const TABLEBASE_DRAW: Self = Self(0b0000_0111);

    #[must_use]
    #[inline]
    pub const fn is_standard(self) -> bool {
        self.0 == 0
    }

    #[must_use]
    #[inline]
    pub const fn is_valid(self) -> bool {
        (self.0 & Self::RESERVED_MASK) == 0
    }

    #[must_use]
    #[inline]
    pub const fn is_terminal(self) -> bool {
        (self.0 & Self::OUTCOME_MASK) != 0 && (self.0 & Self::TABLEBASE_MASK) == 0
    }

    #[must_use]
    #[inline]
    pub const fn is_tablebase(self) -> bool {
        (self.0 & Self::OUTCOME_MASK) != 0 && (self.0 & Self::TABLEBASE_MASK) != 0
    }

    #[must_use]
    pub const fn adjust_eval(self, eval: i64) -> i64 {
        match self {
            Self::TERMINAL_WIN => 2 * SCALE as i64,
            Self::TERMINAL_LOSS => -2 * SCALE as i64,
            Self::TABLEBASE_WIN => SCALE as i64,
            Self::TABLEBASE_LOSS => -SCALE as i64,
            Self::TERMINAL_DRAW | Self::TABLEBASE_DRAW => 0,
            _ => eval,
        }
    }
}

#[cfg(feature = "value-net")]
static VALUE_NETWORK: QuantizedValueNetwork =
    unsafe { std::mem::transmute(*include_bytes!("nets/value.bin")) };

#[must_use]
#[cfg(feature = "value-net")]
pub fn value(state: &State, eval_options: EvaluationOptions) -> i64 {
    (VALUE_NETWORK.get(state, eval_options) * SCALE) as i64
}

#[must_use]
#[cfg(not(feature = "value-net"))]
pub fn value(_state: &State, _eval_options: EvaluationOptions) -> i64 {
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
            Flag::TERMINAL_LOSS
        } else {
            Flag::TERMINAL_DRAW
        }
    } else if let Some(wdl) = tablebase::probe_wdl(state.board()) {
        match wdl {
            Wdl::Win => Flag::TABLEBASE_WIN,
            Wdl::Loss => Flag::TABLEBASE_LOSS,
            Wdl::Draw => Flag::TABLEBASE_DRAW,
        }
    } else {
        Flag::STANDARD
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

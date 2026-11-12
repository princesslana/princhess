use crate::chess::MoveList;
use crate::engine::{KNOWN_WIN, MATE_SCORE};
use crate::options::EvaluationOptions;
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
            Self::TERMINAL_WIN => MATE_SCORE,
            Self::TERMINAL_LOSS => -MATE_SCORE,
            Self::TABLEBASE_WIN => KNOWN_WIN,
            Self::TABLEBASE_LOSS => -KNOWN_WIN,
            Self::TERMINAL_DRAW | Self::TABLEBASE_DRAW => 0,
            _ => eval,
        }
    }
}

/// Loads the nets now, so the first search doesn't pay for it.
pub fn init_nets() {
    value_net::init();
    policy_net::init();
}

#[must_use]
pub fn value(state: &State, eval_options: EvaluationOptions, correction: i32) -> i64 {
    value_net::value(state, eval_options, correction)
}

#[must_use]
pub fn policy(state: &State, moves: &MoveList, t: f32) -> Vec<f32> {
    policy_net::policy(state, moves, t)
}

#[must_use]
pub fn evaluate_state_flag(state: &State, is_legal_moves: bool) -> Flag {
    if !is_legal_moves {
        if state.is_check() {
            Flag::TERMINAL_LOSS
        } else {
            Flag::TERMINAL_DRAW
        }
    } else if state.halfmove_clock() != 0 {
        Flag::STANDARD
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

#[cfg(feature = "value-net")]
mod value_net {
    use std::sync::LazyLock;

    use crate::engine::SCALE;
    use crate::nets;
    use crate::options::EvaluationOptions;
    use crate::quantized_value::QuantizedValueNetwork;
    use crate::shm;
    use crate::state::State;

    static EMBEDDED: QuantizedValueNetwork = bytemuck::must_cast(*include_bytes!("nets/value.bin"));

    static NETWORK: LazyLock<&QuantizedValueNetwork> =
        LazyLock::new(|| shm::share(nets::NET_MD5_VALUE, &EMBEDDED));

    pub fn init() {
        LazyLock::force(&NETWORK);
    }

    pub fn value(state: &State, eval_options: EvaluationOptions, correction: i32) -> i64 {
        (NETWORK.get(state, eval_options, correction) * SCALE) as i64
    }
}

#[cfg(not(feature = "value-net"))]
mod value_net {
    use crate::options::EvaluationOptions;
    use crate::state::State;

    pub fn init() {}

    pub fn value(_state: &State, _eval_options: EvaluationOptions, _correction: i32) -> i64 {
        0
    }
}

#[cfg(feature = "policy-net")]
mod policy_net {
    use std::sync::LazyLock;

    use crate::chess::MoveList;
    use crate::math;
    use crate::nets;
    use crate::quantized_eg_policy::QuantizedEgPolicyNetwork;
    use crate::quantized_mg_policy::QuantizedMgPolicyNetwork;
    use crate::shm;
    use crate::state::State;

    static MG_EMBEDDED: QuantizedMgPolicyNetwork =
        bytemuck::must_cast(*include_bytes!("nets/mg-policy.bin"));

    static MG_NETWORK: LazyLock<&QuantizedMgPolicyNetwork> =
        LazyLock::new(|| shm::share(nets::NET_MD5_MG_POLICY, &MG_EMBEDDED));

    static EG_EMBEDDED: QuantizedEgPolicyNetwork =
        bytemuck::must_cast(*include_bytes!("nets/eg-policy.bin"));

    static EG_NETWORK: LazyLock<&QuantizedEgPolicyNetwork> =
        LazyLock::new(|| shm::share(nets::NET_MD5_EG_POLICY, &EG_EMBEDDED));

    pub fn init() {
        LazyLock::force(&MG_NETWORK);
        LazyLock::force(&EG_NETWORK);
    }

    pub fn policy(state: &State, moves: &MoveList, t: f32) -> Vec<f32> {
        let mut evalns = vec![0.0; moves.len()];

        if moves.is_empty() {
            return evalns;
        }

        if state.is_endgame() {
            EG_NETWORK.get_all(state, state.moves_to_indexes(moves), &mut evalns);
        } else {
            MG_NETWORK.get_all(state, state.moves_to_indexes(moves), &mut evalns);
        }

        math::softmax(&mut evalns, t);

        evalns
    }
}

#[cfg(not(feature = "policy-net"))]
mod policy_net {
    use crate::chess::MoveList;
    use crate::state::State;

    pub fn init() {}

    pub fn policy(_state: &State, moves: &MoveList, _t: f32) -> Vec<f32> {
        vec![1.0 / moves.len() as f32; moves.len()]
    }
}

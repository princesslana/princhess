use crate::chess::MoveList;
use crate::math;
use crate::mem::boxed_and_zeroed;
use crate::search::SCALE;
use crate::state::{self, State};
use crate::tablebase::{self, Wdl};

use std::fs;
use std::io::Write;
use std::mem;
use std::path::Path;
use std::slice;

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

#[must_use]
pub fn evaluate_value(state: &State) -> i64 {
    (run_value_net(state) * SCALE) as i64
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

const HIDDEN: usize = 512;
const QA: i32 = 256;
const QB: i32 = 256;
const QAB: i32 = QA * QB;

const VALUE_NUMBER_INPUTS: usize = state::VALUE_NUMBER_FEATURES;

#[repr(C)]
pub struct ValueNetwork {
    hidden_weights: [Accumulator<HIDDEN>; VALUE_NUMBER_INPUTS],
    hidden_bias: Accumulator<HIDDEN>,
    output_weights: Accumulator<HIDDEN>,
    output_bias: i32,
}

#[cfg(not(feature = "no-net"))]
static VALUE_NETWORK: ValueNetwork =
    unsafe { std::mem::transmute(*include_bytes!("nets/value.bin")) };

impl ValueNetwork {
    fn boxed_and_zeroed() -> Box<Self> {
        boxed_and_zeroed()
    }

    #[must_use]
    pub fn from_slices(
        hidden_weights: &[[i16; HIDDEN]; VALUE_NUMBER_INPUTS],
        hidden_bias: &[i16; HIDDEN],
        output_weights: &[[i16; HIDDEN]; 1],
        output_bias: i32,
    ) -> Box<Self> {
        let mut network = Self::boxed_and_zeroed();

        network.hidden_weights = unsafe { std::mem::transmute(*hidden_weights) };
        network.hidden_bias = unsafe { std::mem::transmute(*hidden_bias) };
        network.output_weights = unsafe { std::mem::transmute(*output_weights) };
        network.output_bias = output_bias;

        network
    }

    pub fn save_to_bin(&self, dir: &Path) {
        let mut file = fs::File::create(dir.join("value.bin")).expect("Failed to create file");

        let size_of = mem::size_of::<Self>();

        unsafe {
            let ptr: *const Self = self;
            let slice_ptr: *const u8 = ptr.cast::<u8>();
            let slice = slice::from_raw_parts(slice_ptr, size_of);
            file.write_all(slice).unwrap();
        }
    }
}

#[derive(Clone, Copy)]
#[repr(C, align(64))]
pub struct Accumulator<const H: usize> {
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

#[cfg(feature = "no-net")]
fn run_value_net(_state: &State) -> f32 {
    0.0
}

#[cfg(not(feature = "no-net"))]
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
        evalns.push(logit as f32 / QAB as f32);
    }

    math::softmax(&mut evalns, t);

    evalns
}

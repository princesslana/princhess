use std::path::Path;

use bytemuck::{allocation, Pod, Zeroable};

use crate::chess::Piece;
use crate::mem::Align64;
use crate::nets::{self, Accumulator};
use crate::options::EvaluationOptions;
use crate::state::{self, State};

pub const INPUT_SIZE: usize = state::VALUE_NUMBER_FEATURES;
pub const HIDDEN_SIZE: usize = 320;
pub const QA: i32 = 256;
pub const QB: i32 = 256;
pub const QAB: i32 = QA * QB;

const PHASE_DIVISOR: i32 = 1024;
const PHASE_OFFSET: i32 = 752;

pub type RawFeatureWeights = Align64<[[i16; HIDDEN_SIZE]; INPUT_SIZE]>;
pub type RawFeatureBias = Align64<[i16; HIDDEN_SIZE]>;
pub type RawOutputWeights = Align64<[i16; HIDDEN_SIZE * 2]>;

pub type QuantizedFeatureWeights = [Align64<Accumulator<i16, HIDDEN_SIZE>>; INPUT_SIZE];
pub type QuantizedFeatureBias = Align64<Accumulator<i16, HIDDEN_SIZE>>;
pub type QuantizedOutputWeights = [Align64<Accumulator<i16, HIDDEN_SIZE>>; 2];

#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(C)]
pub struct QuantizedValueNetwork {
    stm_weights: QuantizedFeatureWeights,
    stm_bias: QuantizedFeatureBias,
    nstm_weights: QuantizedFeatureWeights,
    nstm_bias: QuantizedFeatureBias,
    output_weights: QuantizedOutputWeights,
    output_bias: i32,
    _padding: [u8; 60],
}

impl QuantizedValueNetwork {
    #[must_use]
    pub fn zeroed() -> Box<Self> {
        allocation::zeroed_box()
    }

    #[must_use]
    pub fn from_slices(
        stm_weights: &RawFeatureWeights,
        stm_bias: &RawFeatureBias,
        nstm_weights: &RawFeatureWeights,
        nstm_bias: &RawFeatureBias,
        output_weights: &RawOutputWeights,
        output_bias: i32,
    ) -> Box<Self> {
        let mut network = Self::zeroed();

        network.stm_weights = *bytemuck::must_cast_ref(stm_weights);
        network.stm_bias = *bytemuck::must_cast_ref(stm_bias);

        network.nstm_weights = *bytemuck::must_cast_ref(nstm_weights);
        network.nstm_bias = *bytemuck::must_cast_ref(nstm_bias);

        network.output_weights = *bytemuck::must_cast_ref(output_weights);
        network.output_bias = output_bias;

        network
    }

    pub fn save_to_bin(&self, dir: &Path) {
        nets::save_to_bin(dir, "value.bin", self);
    }

    #[must_use]
    #[allow(clippy::cast_possible_wrap)]
    pub fn get(&self, state: &State, eval_options: EvaluationOptions) -> f32 {
        let mut stm = self.stm_bias;
        let mut nstm = self.nstm_bias;

        state.value_features_map(|idx| {
            if idx < INPUT_SIZE {
                stm.set(&self.stm_weights[idx]);
            } else {
                nstm.set(&self.nstm_weights[idx % INPUT_SIZE]);
            }
        });

        let mut result: i32 = 0;

        for (&x, w) in stm.vals.iter().zip(self.output_weights[0].vals) {
            result += nets::screlu(x, QA) * i32::from(w);
        }

        for (&x, w) in nstm.vals.iter().zip(self.output_weights[1].vals) {
            result += nets::screlu(x, QA) * i32::from(w);
        }

        result = result / QA + self.output_bias;

        // Material scaling
        let board = state.board();
        let material_phase = (board.knights().count() as i32) * Piece::KNIGHT.see_value()
            + (board.bishops().count() as i32) * Piece::BISHOP.see_value()
            + (board.rooks().count() as i32) * Piece::ROOK.see_value()
            + (board.queens().count() as i32) * Piece::QUEEN.see_value();
        let scale_factor = PHASE_OFFSET + material_phase / 32;
        result = [result, result * scale_factor / PHASE_DIVISOR]
            [usize::from(eval_options.enable_material_scaling)];

        // Fifty move rule dampening
        let hmc = state.halfmove_clock();
        let dampen_factor = 256 - i32::from(hmc.saturating_sub(20) + hmc.saturating_sub(52));
        result =
            [result, result * dampen_factor / 256][usize::from(eval_options.enable_50mr_scaling)];

        (result as f32 / QAB as f32).tanh()
    }

    #[must_use]
    pub fn stm_weight(&self, feat_idx: usize) -> &Accumulator<i16, HIDDEN_SIZE> {
        &self.stm_weights[feat_idx]
    }

    #[must_use]
    pub fn stm_bias(&self) -> &Accumulator<i16, HIDDEN_SIZE> {
        &self.stm_bias
    }

    #[must_use]
    pub fn nstm_weight(&self, feat_idx: usize) -> &Accumulator<i16, HIDDEN_SIZE> {
        &self.nstm_weights[feat_idx]
    }

    #[must_use]
    pub fn nstm_bias(&self) -> &Accumulator<i16, HIDDEN_SIZE> {
        &self.nstm_bias
    }

    #[must_use]
    pub fn output_weight(&self, idx: usize) -> &Accumulator<i16, HIDDEN_SIZE> {
        &self.output_weights[idx]
    }

    #[must_use]
    pub fn output_bias(&self) -> i32 {
        self.output_bias
    }
}

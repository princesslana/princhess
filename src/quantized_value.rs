use bytemuck::{allocation, Pod, Zeroable};
use std::path::Path;

use crate::mem::Align64;
use crate::nets::{save_to_bin, screlu, Accumulator};
use crate::state::{self, State};

pub const INPUT_SIZE: usize = state::VALUE_NUMBER_FEATURES;
pub const HIDDEN_SIZE: usize = 320;
pub const QA: i32 = 256;
pub const QB: i32 = 256;
pub const QAB: i32 = QA * QB;

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
        save_to_bin(dir, "value.bin", self);
    }

    #[must_use]
    #[allow(clippy::cast_possible_wrap)]
    pub fn get(&self, state: &State) -> f32 {
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
            result += screlu(x, QA) * i32::from(w);
        }

        for (&x, w) in nstm.vals.iter().zip(self.output_weights[1].vals) {
            result += screlu(x, QA) * i32::from(w);
        }

        result = result / QA + self.output_bias;

        // Fifty move rule dampening
        // Constants are chosen to make the max effect more significant at higher levels and max 50%
        let hmc = state.halfmove_clock();
        result = result * (256 - i32::from(hmc.saturating_sub(20) + hmc.saturating_sub(52))) / 256;

        (result as f32 / QAB as f32).tanh()
    }
}

use crate::math::Rng;
use crate::state;
use crate::train::{boxed_and_zeroed, randomize, write_layer, write_vector};

use goober::activation::{ReLU, Tanh};
use goober::layer::{DenseConnected, SparseConnected};
use goober::FeedForwardNetwork;
use std::boxed::Box;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

const INPUT_SIZE: usize = state::VALUE_NUMBER_FEATURES;
const HIDDEN_SIZE: usize = 384;
const OUTPUT_SIZE: usize = 1;

const QA: f32 = 256.;
const QB: f32 = 256.;
const QAB: f32 = QA * QB;

#[allow(clippy::module_name_repetitions)]
#[derive(FeedForwardNetwork)]
pub struct ValueNetwork {
    hidden: SparseConnected<ReLU, INPUT_SIZE, HIDDEN_SIZE>,
    output: DenseConnected<Tanh, HIDDEN_SIZE, OUTPUT_SIZE>,
}

impl ValueNetwork {
    #[must_use]
    pub fn zeroed() -> Box<Self> {
        boxed_and_zeroed()
    }

    #[must_use]
    pub fn random() -> Box<Self> {
        let mut rng = Rng::default();

        let mut network = Self::zeroed();

        randomize(&mut network.hidden, &mut rng);

        // cos output layer is dense, we can't use randomize
        let limit = (6. / (HIDDEN_SIZE + OUTPUT_SIZE) as f32).sqrt();

        for row_idx in 0..OUTPUT_SIZE {
            let row = network.output.weights_row_mut(row_idx);
            for weight_idx in 0..HIDDEN_SIZE {
                row[weight_idx] = rng.next_f32_range(-limit, limit);
            }
        }

        network
    }

    pub fn decay_weights(&mut self, decay: f32) {
        for row_idx in 0..INPUT_SIZE {
            let row = self.hidden.weights_row_mut(row_idx);
            for weight_idx in 0..HIDDEN_SIZE {
                row[weight_idx] *= decay;
            }
        }

        for weight_idx in 0..HIDDEN_SIZE {
            self.hidden.bias_mut()[weight_idx] *= decay;
        }

        for row_idx in 0..OUTPUT_SIZE {
            let row = self.output.weights_row_mut(row_idx);
            for weight_idx in 0..HIDDEN_SIZE {
                row[weight_idx] *= decay;
            }
        }

        for weight_idx in 0..OUTPUT_SIZE {
            self.output.bias_mut()[weight_idx] *= decay;
        }
    }

    pub fn save(&self, path: &str) {
        fs::create_dir(path).expect("Failed to create directory");

        let dir = Path::new(path);

        write_layer(dir, "hidden", &self.hidden, QA);

        let output_weights_file =
            File::create(dir.join("output_weights")).expect("Failed to create file");
        let mut w = BufWriter::new(output_weights_file);

        writeln!(w, "[").unwrap();
        for row_idx in 0..OUTPUT_SIZE {
            let row = self.output.weights_row(row_idx);
            write_vector(&mut w, &row, QB);
            write!(w, ",").unwrap();
        }
        writeln!(w, "]").unwrap();

        let output_bias_file =
            File::create(dir.join("output_bias")).expect("Failed to create file");
        let mut w = BufWriter::new(output_bias_file);

        write_vector(&mut w, &self.output.bias(), QAB);
    }
}

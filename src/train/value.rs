use crate::evaluation;
use crate::math::Rng;
use crate::state;

use goober::activation::{ReLU, Tanh};
use goober::layer::{DenseConnected, SparseConnected};
use goober::{FeedForwardNetwork, Vector};
use std::alloc::{self, Layout};
use std::boxed::Box;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

const INPUT_SIZE: usize = state::VALUE_NUMBER_FEATURES;
const HIDDEN_SIZE: usize = 448;
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
        unsafe {
            let layout = Layout::new::<Self>();
            let ptr = alloc::alloc_zeroed(layout);
            if ptr.is_null() {
                alloc::handle_alloc_error(layout);
            }
            Box::from_raw(ptr.cast())
        }
    }

    #[must_use]
    pub fn random() -> Box<Self> {
        let mut rng = Rng::default();

        let mut network = Self::zeroed();

        let hidden_limit = (6. / (INPUT_SIZE + HIDDEN_SIZE) as f32).sqrt() * 2f32.sqrt();
        let output_limit = (6. / (HIDDEN_SIZE + OUTPUT_SIZE) as f32).sqrt();

        for row_idx in 0..INPUT_SIZE {
            let row = network.hidden.weights_row_mut(row_idx);
            for weight_idx in 0..HIDDEN_SIZE {
                row[weight_idx] = rng.next_f32_range(-hidden_limit, hidden_limit);
            }
        }

        for row_idx in 0..OUTPUT_SIZE {
            let row = network.output.weights_row_mut(row_idx);
            for weight_idx in 0..HIDDEN_SIZE {
                row[weight_idx] = rng.next_f32_range(-output_limit, output_limit);
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

    pub fn save(&self, dir: &Path) {
        let hidden_weights_file =
            File::create(dir.join("hidden_weights")).expect("Failed to create file");
        let mut w = BufWriter::new(hidden_weights_file);

        writeln!(w, "[").unwrap();
        for row_idx in 0..INPUT_SIZE {
            let row = self.hidden.weights_row(row_idx);
            write_vector(&mut w, &row, QA);
            write!(w, ",").unwrap();
        }
        writeln!(w, "]").unwrap();

        let hidden_bias_file =
            File::create(dir.join("hidden_bias")).expect("Failed to create file");
        let mut w = BufWriter::new(hidden_bias_file);

        write_vector(&mut w, &self.hidden.bias(), QA);

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

fn write_vector<const N: usize>(w: &mut BufWriter<File>, v: &Vector<N>, q: f32) {
    writeln!(w, "    [").unwrap();
    write!(w, "    ").unwrap();
    for weight_idx in 0..N {
        write!(w, "{:>6}, ", (v[weight_idx] * q) as i16).unwrap();

        if weight_idx % 8 == 7 {
            writeln!(w).unwrap();
            write!(w, "    ").unwrap();
        }
        if weight_idx % 64 == 63 {
            writeln!(w).unwrap();
            write!(w, "    ").unwrap();
        }
    }
    writeln!(w, "]").unwrap();
}

#[allow(clippy::large_stack_arrays)]
impl From<&Box<ValueNetwork>> for evaluation::ValueNetwork {
    fn from(network: &Box<ValueNetwork>) -> Self {
        let mut hidden_weights = [[0; HIDDEN_SIZE]; INPUT_SIZE];
        let mut hidden_bias = [0; HIDDEN_SIZE];
        let mut output_weights = [[0; HIDDEN_SIZE]; 1];

        for (row_idx, weights) in hidden_weights.iter_mut().enumerate() {
            let row = network.hidden.weights_row(row_idx);
            for weight_idx in 0..HIDDEN_SIZE {
                weights[weight_idx] = q_i16(row[weight_idx], QA);
            }
        }

        for (weight_idx, bias) in hidden_bias.iter_mut().enumerate() {
            *bias = q_i16(network.hidden.bias()[weight_idx], QA);
        }

        for (row_idx, weights) in output_weights.iter_mut().enumerate() {
            let row = network.output.weights_row(row_idx);
            for weight_idx in 0..HIDDEN_SIZE {
                weights[weight_idx] = q_i16(row[weight_idx], QB);
            }
        }

        let output_bias = q_i32(network.output.bias()[0], QAB);

        evaluation::ValueNetwork::from_slices(
            &hidden_weights,
            &hidden_bias,
            &output_weights,
            output_bias,
        )
    }
}

fn q_i16(x: f32, q: f32) -> i16 {
    let quantized = x * q;
    assert!(f32::from(i16::MIN) < quantized && quantized < f32::from(i16::MAX),);
    quantized as i16
}

fn q_i32(x: f32, q: f32) -> i32 {
    let quantized = x * q;
    assert!((i32::MIN as f32) < quantized && quantized < i32::MAX as f32);
    quantized as i32
}

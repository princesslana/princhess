use crate::math::Rng;
use crate::state;

use goober::activation::{ReLU, Tanh};
use goober::layer::{DenseConnected, SparseConnected};
use goober::{FeedForwardNetwork, Vector};
use std::alloc::{self, Layout};
use std::boxed::Box;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

const INPUT_SIZE: usize = state::NUMBER_FEATURES;
const HIDDEN_SIZE: usize = 384;
const OUTPUT_SIZE: usize = 1;

const QA: f32 = 256.;
const QB: f32 = 256.;
const QAB: f32 = QA * QB;

static HIDDEN_WEIGHTS: [[i16; HIDDEN_SIZE]; INPUT_SIZE] = include!("../model/hidden_weights");
static HIDDEN_BIAS: [i16; HIDDEN_SIZE] = include!("../model/hidden_bias");
static OUTPUT_WEIGHTS: [[i16; HIDDEN_SIZE]; OUTPUT_SIZE] = include!("../model/output_weights");
static OUTPUT_BIAS: [i32; OUTPUT_SIZE] = include!("../model/output_bias");

#[allow(clippy::module_name_repetitions)]
#[derive(FeedForwardNetwork)]
pub struct StateNetwork {
    hidden: SparseConnected<ReLU, INPUT_SIZE, HIDDEN_SIZE>,
    output: DenseConnected<Tanh, HIDDEN_SIZE, OUTPUT_SIZE>,
}

impl StateNetwork {
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

        let hidden_limit = (6. / (INPUT_SIZE + HIDDEN_SIZE) as f32).sqrt() * 2f32.sqrt();
        let output_limit = (6. / (HIDDEN_SIZE + OUTPUT_SIZE) as f32).sqrt();

        let mut zerof = |_| 0.;

        Box::new(Self {
            hidden: SparseConnected::from_fn(
                |_, _| rng.next_f32_range(-hidden_limit, hidden_limit),
                &mut zerof,
            ),
            output: DenseConnected::from_fn(
                |_, _| rng.next_f32_range(-output_limit, output_limit),
                &mut zerof,
            ),
        })
    }

    #[must_use]
    pub fn from_current() -> Box<Self> {
        let mut hidden_weights_f = |i: usize, j: usize| f32::from(HIDDEN_WEIGHTS[i][j]) / QA;
        let mut hidden_bias_f = |i: usize| f32::from(HIDDEN_BIAS[i]) / QA;
        let mut output_weights_f = |i: usize, j: usize| f32::from(OUTPUT_WEIGHTS[i][j]) / QB;
        let mut output_bias_f = |i: usize| OUTPUT_BIAS[i] as f32 / QAB;

        Box::new(Self {
            hidden: SparseConnected::from_fn(&mut hidden_weights_f, &mut hidden_bias_f),
            output: DenseConnected::from_fn(&mut output_weights_f, &mut output_bias_f),
        })
    }

    pub fn save(&self, path: &str) {
        fs::create_dir(path).expect("Failed to create directory");

        let dir = Path::new(path);

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

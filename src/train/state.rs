use crate::math::Rng;
use crate::state;
use crate::train::activation::TanH;

use goober::activation::ReLU;
use goober::layer::{DenseConnected, SparseConnected};
use goober::{FeedForwardNetwork, Matrix, Vector};
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

#[allow(clippy::module_name_repetitions)]
#[derive(FeedForwardNetwork)]
pub struct StateNetwork {
    hidden: SparseConnected<ReLU, INPUT_SIZE, HIDDEN_SIZE>,
    output: DenseConnected<TanH, HIDDEN_SIZE, OUTPUT_SIZE>,
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

        let mut rw = |_| f32::from(rng.next_i8()) / 127.;

        let mut random_weights = Matrix::zeroed();

        for row in random_weights.iter_mut() {
            *row = Vector::from_fn(&mut rw);
        }

        let hidden_weights = Matrix::from_raw(*random_weights);
        let output_weights = Matrix::from_raw([Vector::from_fn(&mut rw)]);

        let mut network = Self::zeroed();

        network.hidden = SparseConnected::from_raw(hidden_weights, Vector::zeroed());
        network.output = DenseConnected::from_raw(output_weights, Vector::zeroed());

        network
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
        }
        writeln!(w, "]").unwrap();

        let hidden_bias_file =
            File::create(dir.join("hidden_bias")).expect("Failed to create file");
        let mut w = BufWriter::new(hidden_bias_file);

        write_vector(&mut w, &self.hidden.bias(), QA);

        let output_weights_file =
            File::create(dir.join("output_weights")).expect("Failed to create file");
        let mut w = BufWriter::new(output_weights_file);
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
    writeln!(w, "],").unwrap();
}

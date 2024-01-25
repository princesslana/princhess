mod data;
mod policy;
mod value;

pub use crate::train::data::TrainingPosition;
pub use crate::train::policy::PolicyNetwork;
pub use crate::train::value::ValueNetwork;

use goober::activation::Activation;
use goober::layer::SparseConnected;
use goober::Vector;
use std::alloc::{self, Layout};
use std::boxed::Box;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::math::Rng;

fn boxed_and_zeroed<T>() -> Box<T> {
    unsafe {
        let layout = Layout::new::<T>();
        let ptr = alloc::alloc_zeroed(layout);
        if ptr.is_null() {
            alloc::handle_alloc_error(layout);
        }
        Box::from_raw(ptr.cast())
    }
}

fn randomize<A: Activation, const I: usize, const O: usize>(
    layer: &mut SparseConnected<A, I, O>,
    rng: &mut Rng,
) {
    let limit = (6. / (I + O) as f32).sqrt();

    for row_idx in 0..I {
        let row = layer.weights_row_mut(row_idx);
        for weight_idx in 0..O {
            row[weight_idx] = rng.next_f32_range(-limit, limit);
        }
    }
}

fn write_layer<A: Activation, const M: usize, const N: usize>(
    dir: &Path,
    prefix: &str,
    layer: &SparseConnected<A, M, N>,
    q: f32,
) {
    let weights_file = File::create(dir.join(format!("{prefix}_weights"))).unwrap();
    let mut w = BufWriter::new(weights_file);
    write_weights(&mut w, layer, q);

    let bias_file = File::create(dir.join(format!("{prefix}_bias"))).unwrap();
    let mut w = BufWriter::new(bias_file);
    write_vector(&mut w, &layer.bias(), q);
}

fn write_weights<A: Activation, const M: usize, const N: usize>(
    w: &mut BufWriter<File>,
    layer: &SparseConnected<A, M, N>,
    q: f32,
) {
    writeln!(w, "[").unwrap();
    for row_idx in 0..M {
        let row = layer.weights_row(row_idx);
        write_vector(w, &row, q);
        write!(w, ",").unwrap();
    }
    writeln!(w, "]").unwrap();
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

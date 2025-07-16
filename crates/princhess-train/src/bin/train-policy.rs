use arrayvec::ArrayVec;
use bytemuck::Zeroable;
use princhess::math;
use princhess::state::State;
use princhess_train::neural::SparseVector;
use std::env;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, Write};
use std::ops::AddAssign;
use std::path::Path;
use std::thread;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use princhess_train::data::TrainingPosition;
use princhess_train::policy::{Phase, PolicyCount, PolicyNetwork};

const TARGET_BATCH_COUNT: usize = 300_000;
const BATCH_SIZE: usize = 32768;
const THREADS: usize = 6;

const LR: f32 = 0.001;
const LR_DROP_AT: f32 = 0.35;
const LR_DROP_FACTOR: f32 = 0.5;

const SOFT_TARGET_WEIGHT: f32 = 0.1;
const SOFT_TARGET_TEMPERATURE: f32 = 4.0;

const EPSILON: f32 = 1e-9;

const _BUFFER_SIZE_CHECK: () = assert!(TrainingPosition::BUFFER_SIZE % BATCH_SIZE == 0);

#[derive(Debug, Default, Clone, Copy)]
struct BatchMetrics {
    loss: f32,
    accuracy: f32,
    processed_count: usize,
}

impl AddAssign for BatchMetrics {
    fn add_assign(&mut self, rhs: Self) {
        self.loss += rhs.loss;
        self.accuracy += rhs.accuracy;
        self.processed_count += rhs.processed_count;
    }
}

fn main() {
    println!("Running...");
    let mut args = env::args();
    args.next();

    let input = args.next().expect("Missing input file");
    let phase_arg = args
        .next()
        .expect("Missing phase argument (--phase <mg|eg>)");

    let phase = Phase::from_arg(&phase_arg)
        .unwrap_or_else(|| panic!("Invalid phase argument: {phase_arg}. Use 'mg' or 'eg'."));

    let file = File::open(input.clone()).unwrap();
    let count = file.metadata().unwrap().len() as usize / TrainingPosition::SIZE;

    let mut network = PolicyNetwork::random();

    let mut lr = LR;
    let mut momentum = PolicyNetwork::zeroed();
    let mut velocity = PolicyNetwork::zeroed();

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    println!("Beginning training...");
    println!("Network: {network}");
    println!("Positions: {count}");
    println!("File: {input}");
    println!("Training Phase: {phase}");

    let epochs = TARGET_BATCH_COUNT.div_ceil(count / BATCH_SIZE);
    let lr_drop_at = (epochs as f32 * LR_DROP_AT).ceil() as usize;
    let net_save_period = epochs.div_ceil(10);

    for epoch in 1..=epochs {
        println!("\nEpoch {epoch}/{epochs} (LR: {lr})...");
        let start = Instant::now();

        train(
            &mut network,
            lr,
            &mut momentum,
            &mut velocity,
            input.as_str(),
            phase,
        );

        let seconds = start.elapsed().as_secs();

        println!(
            "Epoch {} complete in {}m {}s.",
            epoch,
            (seconds / 60),
            (seconds % 60)
        );

        if epoch % net_save_period == 0 || epoch == epochs {
            let dir_name = format!("nets/policy-{phase}-{timestamp}-e{epoch:03}");

            fs::create_dir(&dir_name).unwrap();

            let dir = Path::new(&dir_name);

            network
                .to_boxed_and_quantized()
                .save_to_bin(dir, format!("{phase}-policy.bin").as_str());

            println!("Saved to {dir_name}");
        }

        if epoch % lr_drop_at == 0 {
            lr *= LR_DROP_FACTOR;
        }
    }
}

fn train(
    network: &mut PolicyNetwork,
    lr: f32,
    momentum: &mut PolicyNetwork,
    velocity: &mut PolicyNetwork,
    input: &str,
    phase: Phase,
) {
    let file = File::open(input).unwrap();
    let _positions = file.metadata().unwrap().len() as usize / TrainingPosition::SIZE;

    let mut buffer = BufReader::with_capacity(TrainingPosition::BUFFER_SIZE, file);

    let mut running_metrics = BatchMetrics::default();
    let mut batch_n = 0;
    let batches = _positions.div_ceil(BATCH_SIZE);

    while let Ok(bytes) = buffer.fill_buf() {
        if bytes.is_empty() {
            break;
        }

        let data = TrainingPosition::read_buffer(bytes);

        for batch in data.chunks(BATCH_SIZE) {
            let mut gradients = PolicyNetwork::zeroed();
            let mut count = PolicyCount::zeroed();

            let batch_metrics = gradients_batch(network, &mut gradients, &mut count, batch, phase);
            running_metrics += batch_metrics;

            network.adam(&gradients, momentum, velocity, &count, lr);

            batch_n += 1;
            print!("Batch {batch_n}/{batches}\r");
            io::stdout().flush().unwrap();
        }

        let consumed = bytes.len();
        buffer.consume(consumed);
    }

    if running_metrics.processed_count > 0 {
        println!(
            "Running loss: {}",
            running_metrics.loss / running_metrics.processed_count as f32
        );
        println!(
            "Running accuracy: {}",
            running_metrics.accuracy / running_metrics.processed_count as f32
        );
    } else {
        println!("No positions processed for phase: {phase}");
    }
}

fn gradients_batch(
    network: &PolicyNetwork,
    gradients: &mut PolicyNetwork,
    count: &mut PolicyCount,
    batch: &[TrainingPosition],
    phase: Phase,
) -> BatchMetrics {
    let size = (batch.len() / THREADS) + 1;
    let mut total_metrics = BatchMetrics::default();

    thread::scope(|s| {
        batch
            .chunks(size)
            .map(|chunk| {
                s.spawn(move || {
                    let mut inner_gradients = PolicyNetwork::zeroed();
                    let mut inner_count = PolicyCount::zeroed();
                    let mut inner_metrics = BatchMetrics::default();

                    for position in chunk {
                        update_gradient(
                            position,
                            network,
                            &mut inner_gradients,
                            &mut inner_count,
                            &mut inner_metrics,
                            phase,
                        );
                    }
                    (inner_gradients, inner_count, inner_metrics)
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .for_each(|(inner_gradients, inner_count, inner_metrics)| {
                *gradients += &inner_gradients;
                *count += &inner_count;
                total_metrics += inner_metrics;
            });
    });

    total_metrics
}

fn update_gradient(
    position: &TrainingPosition,
    network: &PolicyNetwork,
    gradients: &mut PolicyNetwork,
    count: &mut PolicyCount,
    metrics: &mut BatchMetrics,
    phase: Phase,
) {
    let state = State::from(position);

    if !phase.matches(&state) {
        return;
    }

    let moves = position.moves();

    let mut features = SparseVector::with_capacity(64);
    state.policy_features_map(|feature| features.push(feature));

    let only_moves = moves.iter().map(|(mv, _)| *mv).collect();
    let move_idxes = state.moves_to_indexes(&only_moves).collect::<Vec<_>>();

    let mut raw_outputs = vec![0.0; moves.len()];
    network.get_all(&features, move_idxes.iter().copied(), &mut raw_outputs);

    let mut actual_policy = raw_outputs;
    math::softmax(&mut actual_policy, 1.0);

    let raw_counts: ArrayVec<f32, { TrainingPosition::MAX_MOVES }> =
        moves.iter().map(|(_, v)| f32::from(*v)).collect();

    let expected_primary = calculate_target(&raw_counts, 1.0);
    let expected_secondary = calculate_target(&raw_counts, SOFT_TARGET_TEMPERATURE);

    for idx in 0..moves.len() {
        let move_idx = move_idxes[idx];
        let actual_val = actual_policy[idx];
        let log_actual_val = actual_val.max(EPSILON).ln();

        let expected_primary_val = expected_primary[idx];
        metrics.loss -= expected_primary_val * log_actual_val;

        let expected_secondary_val = expected_secondary[idx];
        metrics.loss -= expected_secondary_val * log_actual_val * SOFT_TARGET_WEIGHT;

        let combined_error = (actual_val - expected_primary_val)
            + (actual_val - expected_secondary_val) * SOFT_TARGET_WEIGHT;
        network.backprop(&features, gradients, move_idx, combined_error);
        count.increment(move_idx);
    }

    if argmax(&expected_primary) == argmax(&actual_policy) {
        metrics.accuracy += 1.;
    }
    metrics.processed_count += 1;
}

fn create_uniform_distribution(len: usize) -> ArrayVec<f32, { TrainingPosition::MAX_MOVES }> {
    let mut target = ArrayVec::new();
    if len == 0 {
        return target;
    }
    let uniform_val = 1.0 / len as f32;
    for _ in 0..len {
        target.push(uniform_val);
    }
    target
}

fn calculate_target(
    values: &[f32],
    temperature: f32,
) -> ArrayVec<f32, { TrainingPosition::MAX_MOVES }> {
    let mut target: ArrayVec<f32, { TrainingPosition::MAX_MOVES }> =
        ArrayVec::from_iter(values.iter().copied());

    if target.is_empty() {
        return target;
    }

    // If all values are zero, return a uniform distribution to avoid NaN from log(0) and division by zero.
    let all_zeros = target.iter().all(|&x| x == 0.0);
    if all_zeros {
        return create_uniform_distribution(target.len());
    }

    // `x^(1/T) = exp(ln(x)/T)`. So, we pass `ln(x)` as the logit to `softmax` with temperature `T`.
    // Zero values are mapped to negative infinity, which correctly results in 0 after exp.
    for val in target.iter_mut() {
        *val = if *val > 0.0 {
            val.ln()
        } else {
            f32::NEG_INFINITY
        };
    }

    math::softmax(&mut target, temperature);

    target
}

fn argmax(arr: &[f32]) -> usize {
    let mut max_idx = 0;
    let mut max_val = f32::NEG_INFINITY;

    for (idx, &val) in arr.iter().enumerate() {
        if val > max_val {
            max_val = val;
            max_idx = idx;
        }
    }

    max_idx
}

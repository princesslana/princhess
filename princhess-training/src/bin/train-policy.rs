use bytemuck::Zeroable;
use std::env;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, Write};
use std::path::Path;
use std::thread;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use arrayvec::ArrayVec;
use princhess::math;
use princhess::policy::{PolicyCount, PolicyNetwork};
use princhess::state::State;
use princhess_training::train::TrainingPosition;

const TARGET_BATCH_COUNT: usize = 300_000;
const BATCH_SIZE: usize = 16384;
const THREADS: usize = 6;

const LR: f32 = 0.001;
const LR_DROP_AT: f32 = 0.4;
const LR_DROP_FACTOR: f32 = 0.5;

const SOFT_TARGET_WEIGHT: f32 = 0.1;
const SOFT_TARGET_TEMPERATURE: f32 = 4.0;

const EPSILON: f32 = 1e-9;

const _BUFFER_SIZE_CHECK: () = assert!(TrainingPosition::BUFFER_SIZE % BATCH_SIZE == 0);

fn main() {
    println!("Running...");
    let mut args = env::args();
    args.next();

    let input = args.next().expect("Missing input file");

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
    println!("Network: {}", network);
    println!("Positions: {}", count);
    println!("File: {}", input);

    let epochs = TARGET_BATCH_COUNT.div_ceil(count / BATCH_SIZE);
    let lr_drop_at = (epochs as f32 * LR_DROP_AT) as usize;
    let net_save_period = epochs.div_ceil(10);

    for epoch in 1..=epochs {
        println!("\nEpoch {}/{} (LR: {})...", epoch, epochs, lr);
        let start = Instant::now();

        train(
            &mut network,
            lr,
            &mut momentum,
            &mut velocity,
            input.as_str(),
        );

        let seconds = start.elapsed().as_secs();

        println!(
            "Epoch {} complete in {}m {}s.",
            epoch,
            (seconds / 60),
            (seconds % 60)
        );

        if epoch % net_save_period == 0 || epoch == epochs {
            let dir_name = format!("nets/policy-{}-e{:03}", timestamp, epoch);

            fs::create_dir(&dir_name).unwrap();

            let dir = Path::new(&dir_name);

            network.to_boxed_and_quantized().save_to_bin(dir);

            println!("Saved to {}", dir_name);
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
) {
    let file = File::open(input).unwrap();
    let positions = file.metadata().unwrap().len() as usize / TrainingPosition::SIZE;

    let mut buffer = BufReader::with_capacity(TrainingPosition::BUFFER_SIZE, file);

    let mut running_loss = 0.0;
    let mut running_acc = 0.;
    let mut batch_n = 0;
    let batches = positions.div_ceil(BATCH_SIZE);

    while let Ok(bytes) = buffer.fill_buf() {
        if bytes.is_empty() {
            break;
        }

        let data = TrainingPosition::read_buffer(bytes);

        for batch in data.chunks(BATCH_SIZE) {
            let mut gradients = PolicyNetwork::zeroed();
            let mut count = PolicyCount::zeroed();

            let (loss, acc) = gradients_batch(network, &mut gradients, &mut count, batch);
            running_loss += loss;
            running_acc += acc;

            network.adam(&gradients, momentum, velocity, &count, lr);

            batch_n += 1;
            print!("Batch {}/{}\r", batch_n, batches,);
            io::stdout().flush().unwrap();
        }

        let consumed = bytes.len();
        buffer.consume(consumed);
    }

    println!("Running loss: {}", running_loss / positions as f32);
    println!("Running accuracy: {}", running_acc / positions as f32);
}

fn gradients_batch(
    network: &PolicyNetwork,
    gradients: &mut PolicyNetwork,
    count: &mut PolicyCount,
    batch: &[TrainingPosition],
) -> (f32, f32) {
    let size = (batch.len() / THREADS) + 1;
    let mut loss = [0.0; THREADS];
    let mut acc = [0.0; THREADS];

    thread::scope(|s| {
        batch
            .chunks(size)
            .zip(loss.iter_mut().zip(acc.iter_mut()))
            .map(|(chunk, (loss, acc))| {
                s.spawn(move || {
                    let mut inner_gradients = PolicyNetwork::zeroed();
                    let mut inner_count = PolicyCount::zeroed();

                    for position in chunk {
                        update_gradient(
                            position,
                            network,
                            &mut inner_gradients,
                            &mut inner_count,
                            loss,
                            acc,
                        );
                    }
                    (inner_gradients, inner_count)
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .for_each(|(inner_gradients, inner_count)| {
                *gradients += &inner_gradients;
                *count += &inner_count;
            });
    });

    (loss.iter().sum::<f32>(), acc.iter().sum::<f32>())
}

fn update_gradient(
    position: &TrainingPosition,
    network: &PolicyNetwork,
    gradients: &mut PolicyNetwork,
    count: &mut PolicyCount,
    loss: &mut f32,
    acc: &mut f32,
) {
    let state = State::from(position);
    let features = position.get_policy_features();
    let moves = position.moves();

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

        let expected_primary_val = expected_primary[idx];
        let log_actual_val = actual_val.max(EPSILON).ln();
        let error_primary = actual_val - expected_primary_val;
        *loss -= expected_primary_val * log_actual_val;

        let expected_secondary_val = expected_secondary[idx];
        let error_secondary = actual_val - expected_secondary_val;
        *loss -= expected_secondary_val * log_actual_val * SOFT_TARGET_WEIGHT;

        let combined_error = error_primary + error_secondary * SOFT_TARGET_WEIGHT;
        network.backprop(&features, gradients, move_idx, combined_error);
        count.increment(move_idx);
    }

    if argmax(&expected_primary) == argmax(&actual_policy) {
        *acc += 1.;
    }
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

use goober::{FeedForwardNetwork, OutputLayer, Vector};
use std::env;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, Write};
use std::path::Path;
use std::thread;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use princhess::math;
use princhess::state::State;
use princhess::train::{PolicyNetwork, TrainingPosition};

const EPOCHS: usize = 15;
const BATCH_SIZE: usize = 16384;
const THREADS: usize = 6;
const BUFFER_COUNT: usize = 1 << 16;

const LR: f32 = 0.001;
const LR_DROP_AT: usize = EPOCHS * 2 / 3;
const LR_DROP_FACTOR: f32 = 0.1;

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

    for epoch in 1..=EPOCHS {
        println!("\nEpoch {}/{} (LR: {})...", epoch, EPOCHS, lr);
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

        let dir_name = format!("nets/policy-{}-e{:03}", timestamp, epoch);

        fs::create_dir(&dir_name).unwrap();

        let dir = Path::new(&dir_name);

        network.to_boxed_evaluation_network().save_to_bin(dir);

        println!("Saved to {}", dir_name);

        if epoch % LR_DROP_AT == 0 {
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

    let buffer_size = BUFFER_COUNT * TrainingPosition::SIZE;
    let mut buffer = BufReader::with_capacity(buffer_size, file);

    let mut running_loss = 0.0;
    let mut running_acc = 0.;
    let mut batch_n = 0;
    let batches = (positions + BATCH_SIZE - 1) / BATCH_SIZE;

    while let Ok(bytes) = buffer.fill_buf() {
        if bytes.is_empty() {
            break;
        }

        let data = TrainingPosition::read_batch(bytes);

        for batch in data.chunks(BATCH_SIZE) {
            let mut gradients = PolicyNetwork::zeroed();

            let (loss, acc) = gradients_batch(network, &mut gradients, batch);
            running_loss += loss;
            running_acc += acc;

            let adj = 2.0 / batch.len() as f32;

            //network.decay_weights(1.0 - WEIGHT_DECAY * lr);
            network.adam(&gradients, momentum, velocity, adj, lr);

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
                    for position in chunk {
                        update_gradient(position, network, &mut inner_gradients, loss, acc);
                    }
                    inner_gradients
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .for_each(|inner_gradients| *gradients += &inner_gradients);
    });

    (loss.iter().sum::<f32>(), acc.iter().sum::<f32>())
}

fn update_gradient(
    position: &TrainingPosition,
    network: &PolicyNetwork,
    gradients: &mut PolicyNetwork,
    loss: &mut f32,
    acc: &mut f32,
) {
    let state = State::from(position);
    let features = position.get_policy_features();
    let moves = position.moves();

    let mut move_idxes = [0; TrainingPosition::MAX_MOVES];
    let mut actual = [0.; TrainingPosition::MAX_MOVES];
    let mut expected = [0.; TrainingPosition::MAX_MOVES];

    for idx in 0..moves.len() {
        let (m, v) = moves[idx];
        let move_idx = state.move_to_index(m);

        let c = &network.constant[move_idx]
            .out_with_layers(&features)
            .output_layer()[0];
        let l = &network.left[move_idx]
            .out_with_layers(&features)
            .output_layer()[0];
        let r = &network.right[move_idx]
            .out_with_layers(&features)
            .output_layer()[0];

        move_idxes[idx] = move_idx;
        actual[idx] = c + l * r;

        /*
        expected[idx] = if m == position.best_move() {
            1.0
        } else {
            0.0
        };*/
        expected[idx] = f32::from(v);
    }

    math::softmax(&mut actual[..moves.len()], 1.);

    let sum_expected = expected.iter().sum::<f32>();

    for e in expected.iter_mut() {
        *e /= sum_expected;
    }

    for idx in 0..moves.len() {
        let move_idx = move_idxes[idx];
        let expected = expected[idx];
        let actual = actual[idx];

        let error = actual - expected;
        *loss -= expected * actual.ln();

        let c = &network.constant[move_idx];
        let l = &network.left[move_idx];
        let r = &network.right[move_idx];

        let c_layers = c.out_with_layers(&features);
        let l_layers = l.out_with_layers(&features);
        let r_layers = r.out_with_layers(&features);

        let error_v = Vector::from_raw([error]);

        c.backprop(
            &features,
            &mut gradients.constant[move_idx],
            error_v,
            &c_layers,
        );

        l.backprop(
            &features,
            &mut gradients.left[move_idx],
            error_v * r_layers.output_layer(),
            &l_layers,
        );

        r.backprop(
            &features,
            &mut gradients.right[move_idx],
            error_v * l_layers.output_layer(),
            &r_layers,
        );
    }

    if move_idxes[argmax(&expected)] == move_idxes[argmax(&actual)] {
        *acc += 1.;
    }
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

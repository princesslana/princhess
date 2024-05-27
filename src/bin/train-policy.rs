use std::env;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, Write};
use std::path::Path;
use std::thread;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use princhess::math;
use princhess::policy::PolicyNetwork;
use princhess::state::State;
use princhess::train::TrainingPosition;

const EPOCHS: usize = 10;
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

        network.save_to_bin(dir);

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

    let mut actual = Vec::with_capacity(moves.len());

    let move_idxes = moves
        .iter()
        .map(|(m, _)| state.move_to_index(*m))
        .collect::<Vec<_>>();

    network.get_all(&features, move_idxes.iter().copied(), &mut actual);

    let mut expected = moves.iter().map(|(_, v)| f32::from(*v)).collect::<Vec<_>>();

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

        network.backprop(&features, gradients, move_idx, error);
    }

    if argmax(&expected) == argmax(&actual) {
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

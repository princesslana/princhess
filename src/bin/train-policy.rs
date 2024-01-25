use goober::{FeedForwardNetwork, OutputLayer, Vector};
use std::env;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};
use std::thread;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use princhess::math;
use princhess::state::State;
use princhess::train::{PolicyNetwork, TrainingPosition};

const EPOCHS: usize = 20;
const BATCH_SIZE: usize = 16384;
const THREADS: usize = 6;

const LR: f32 = 0.0001;

fn main() {
    println!("Running...");
    let mut args = env::args();
    args.next();

    let input = args.next().expect("Missing input file");

    let file = File::open(input.clone()).unwrap();
    let count = file.metadata().unwrap().len() as usize / TrainingPosition::SIZE;

    let mut network = PolicyNetwork::random();

    let lr = LR;
    let mut momentum = PolicyNetwork::zeroed();
    let mut velocity = PolicyNetwork::zeroed();

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    println!(
        "Beginning training on {} positions from {}...",
        count, input
    );

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

        let file_name = format!("nets/policy-{}-e{:03}", timestamp, epoch);
        network.save(file_name.as_str());
        println!("Saved to {}", file_name);
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

    let buffer_size = (1 << (THREADS * 2)) * BATCH_SIZE * TrainingPosition::SIZE;
    let mut buffer = BufReader::with_capacity(buffer_size, file);

    let mut running_loss = 0.0;
    let mut batch_n = 0;
    let batches = (positions + BATCH_SIZE - 1) / BATCH_SIZE;

    while let Ok(bytes) = buffer.fill_buf() {
        if bytes.is_empty() {
            break;
        }

        let data = TrainingPosition::read_batch(bytes);

        for batch in data.chunks(BATCH_SIZE) {
            let mut gradients = PolicyNetwork::zeroed();
            running_loss += gradients_batch(network, &mut gradients, batch);
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
}

fn gradients_batch(
    network: &PolicyNetwork,
    gradients: &mut PolicyNetwork,
    batch: &[TrainingPosition],
) -> f32 {
    let size = (batch.len() / THREADS).max(1);
    let mut loss = [0.0; THREADS];

    thread::scope(|s| {
        batch
            .chunks(size)
            .zip(loss.iter_mut())
            .map(|(chunk, loss)| {
                s.spawn(move || {
                    let mut inner_gradients = PolicyNetwork::zeroed();
                    for position in chunk {
                        update_gradient(position, network, &mut inner_gradients, loss);
                    }
                    inner_gradients
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .for_each(|inner_gradients| *gradients += &inner_gradients);
    });

    loss.iter().sum::<f32>()
}

fn update_gradient(
    position: &TrainingPosition,
    network: &PolicyNetwork,
    gradients: &mut PolicyNetwork,
    loss: &mut f32,
) {
    let state = State::from(position);
    let features = position.get_policy_features();
    let moves = position.moves();

    let output_layers = network.out_with_layers(&features);
    let output = output_layers.output_layer();

    let mut move_idxs = Vec::with_capacity(moves.len());
    let mut expected = Vec::with_capacity(moves.len());
    let mut actual = Vec::with_capacity(moves.len());

    for (m, v) in &moves {
        let move_idx = state.move_to_index(*m);
        move_idxs.push(move_idx);
        expected.push(*v);
        actual.push(output[move_idx]);
    }

    math::softmax(&mut actual, 1.);

    let mut errors = Vector::zeroed();

    for idx in 0..moves.len() {
        let move_idx = move_idxs[idx];

        let expected = expected[idx];
        let actual = actual[idx];

        errors[move_idx] = actual - expected;
        *loss -= expected * actual.ln();
    }

    network.backprop(&features, gradients, errors, &output_layers);
}

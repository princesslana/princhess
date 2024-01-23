use princhess::train::{StateNetwork, TrainingPosition};

use goober::{FeedForwardNetwork, OutputLayer, Vector};
use std::env;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};
use std::thread;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const EPOCHS: usize = 20;
const BATCH_SIZE: usize = 16384;
const THREADS: usize = 6;

const LR: f32 = 0.001;
const LR_DROP_AT: usize = EPOCHS * 2 / 3;
const LR_DROP_FACTOR: f32 = 0.01;

const WEIGHT_DECAY: f32 = 0.99;

fn main() {
    println!("Running...");
    let mut args = env::args();
    args.next();

    let input = args.next().expect("Missing input file");

    let file = File::open(input.clone()).unwrap();
    let count = file.metadata().unwrap().len() as usize / TrainingPosition::SIZE;

    let mut network = StateNetwork::random();

    let mut lr = LR;
    let mut momentum = StateNetwork::zeroed();
    let mut velocity = StateNetwork::zeroed();

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

        network.decay_weights(WEIGHT_DECAY);

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

        let file_name = format!("nets/state-{}-e{:03}", timestamp, epoch);
        network.save(file_name.as_str());
        println!("Saved to {}", file_name);

        if epoch % LR_DROP_AT == 0 {
            lr *= LR_DROP_FACTOR;
        }
    }
}

fn train(
    network: &mut StateNetwork,
    lr: f32,
    momentum: &mut StateNetwork,
    velocity: &mut StateNetwork,
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
            let mut gradients = StateNetwork::zeroed();
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
    network: &StateNetwork,
    gradients: &mut StateNetwork,
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
                    let mut inner_gradients = StateNetwork::zeroed();
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
    network: &StateNetwork,
    gradients: &mut StateNetwork,
    loss: &mut f32,
) {
    let features = position.get_features();

    let net_out = network.out_with_layers(&features);

    let expected = position.stm_relative_result() as f32;
    let actual = net_out.output_layer()[0];

    let error = actual - expected;
    *loss += error * error;

    network.backprop(&features, gradients, Vector::from_raw([error]), &net_out);
}

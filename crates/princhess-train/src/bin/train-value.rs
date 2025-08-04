use princhess::state::State;
use princhess_train::neural::{
    AdamWConfig, AdamWOptimizer, FeedForwardNetwork, OutputLayer, SparseVector, Vector,
};
use std::env;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, Write};
use std::path::Path;
use std::thread;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use princhess_train::data::TrainingPosition;
use princhess_train::value::ValueNetwork;

const TARGET_BATCH_COUNT: usize = 400_000;
const BATCH_SIZE: usize = 16384;
const THREADS: usize = 6;

const LR: f32 = 0.001;
const LR_DROP_AT: f32 = 0.7;
const LR_DROP_FACTOR: f32 = 0.1;

const WDL_WEIGHT: f32 = 0.3;

const _BUFFER_SIZE_CHECK: () = assert!(TrainingPosition::BUFFER_SIZE % BATCH_SIZE == 0);

fn main() {
    println!("Running...");
    let mut args = env::args();
    args.next();

    let input = args.next().expect("Missing input file");

    let file = File::open(input.clone()).unwrap();
    let count = file.metadata().unwrap().len() as usize / TrainingPosition::SIZE;

    let mut network = ValueNetwork::random();

    let mut lr = LR;
    let mut momentum = ValueNetwork::zeroed();
    let mut velocity = ValueNetwork::zeroed();

    let mut optimizer = AdamWOptimizer::new(AdamWConfig {
        learning_rate: lr,
        ..Default::default()
    });

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    println!("Beginning training...");
    println!("Network: {network}");
    println!("Positions: {count}");
    println!("File: {input}");

    let epochs = TARGET_BATCH_COUNT.div_ceil(count / BATCH_SIZE);
    let lr_drop_at = (epochs as f32 * LR_DROP_AT) as usize;

    for epoch in 1..=epochs {
        println!("\nEpoch {epoch}/{epochs} (LR: {lr})...");
        let start = Instant::now();

        train(
            &mut network,
            &mut momentum,
            &mut velocity,
            &mut optimizer,
            input.as_str(),
            start,
        );

        let seconds = start.elapsed().as_secs();

        println!(
            "Epoch {} complete in {}m {}s.",
            epoch,
            (seconds / 60),
            (seconds % 60)
        );

        let dir_name = format!("nets/value-{timestamp}-e{epoch:03}");

        fs::create_dir(dir_name.clone()).expect("Failed to create directory");

        let dir = Path::new(&dir_name);

        network.to_boxed_and_quantized().save_to_bin(dir);

        println!("Saved to {dir_name}");

        if epoch % lr_drop_at == 0 {
            lr *= LR_DROP_FACTOR;
            optimizer.set_learning_rate(lr);
        }
    }
}

fn train(
    network: &mut ValueNetwork,
    momentum: &mut ValueNetwork,
    velocity: &mut ValueNetwork,
    optimizer: &mut AdamWOptimizer,
    input: &str,
    start_time: Instant,
) {
    let file = File::open(input).unwrap();
    let positions = file.metadata().unwrap().len() as usize / TrainingPosition::SIZE;

    let mut buffer = BufReader::with_capacity(TrainingPosition::BUFFER_SIZE, file);

    let mut running_loss = 0.0;
    let mut batch_n = 0;
    let batches = positions.div_ceil(BATCH_SIZE);

    while let Ok(bytes) = buffer.fill_buf() {
        if bytes.is_empty() {
            break;
        }

        let data = TrainingPosition::read_buffer(bytes);

        for batch in data.chunks(BATCH_SIZE) {
            let mut gradients = ValueNetwork::zeroed();
            running_loss += gradients_batch(network, &mut gradients, batch);
            let adj = 2.0 / batch.len() as f32;

            optimizer.step();
            network.adamw(&gradients, momentum, velocity, optimizer, adj);

            batch_n += 1;

            // Calculate time remaining estimate
            let elapsed = start_time.elapsed().as_secs();
            let estimated_total = if batch_n > 0 {
                (elapsed * batches as u64) / batch_n as u64
            } else {
                0
            };
            let remaining = estimated_total.saturating_sub(elapsed);
            let remaining_min = remaining / 60;

            print!("Batch {batch_n:5}/{batches:5} (ETA: {remaining_min:3}m)\r");
            io::stdout().flush().unwrap();
        }

        let consumed = bytes.len();
        buffer.consume(consumed);
    }

    println!();
    println!("Running loss: {}", running_loss / positions as f32);
}

fn gradients_batch(
    network: &ValueNetwork,
    gradients: &mut ValueNetwork,
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
                    let mut inner_gradients = ValueNetwork::zeroed();
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
    network: &ValueNetwork,
    gradients: &mut ValueNetwork,
    loss: &mut f32,
) {
    let mut features = SparseVector::with_capacity(64);
    State::from(position).value_features_map(|feature| features.push(feature));

    let net_out = network.out_with_layers(&features);

    let expected = position.stm_relative_result() as f32 * WDL_WEIGHT
        + position.stm_relative_evaluation() * (1.0 - WDL_WEIGHT);
    let actual = net_out.output_layer()[0];

    let error = actual - expected;
    *loss += error * error;

    network.backprop(&features, gradients, Vector::from_raw([error]), &net_out);
}

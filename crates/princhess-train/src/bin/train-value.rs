use princhess::state::State;
use princhess_train::neural::{
    AdamWOptimizer, CosineAnnealingLRScheduler, FeedForwardNetwork, LRScheduler, OutputLayer,
    SparseVector, Vector,
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

const FEATURE_LEARNING_RATE: f32 = 0.001;
const OUTPUT_LEARNING_RATE: f32 = 0.001;
const MIN_LEARNING_RATE: f32 = 0.0001;

const WDL_WEIGHT: f32 = 0.3;
const FEATURE_WEIGHT_DECAY: f32 = 0.01;
const OUTPUT_WEIGHT_DECAY: f32 = 0.01;

const _BUFFER_SIZE_CHECK: () = assert!(TrainingPosition::BUFFER_SIZE.is_multiple_of(BATCH_SIZE));

struct Optimizers<S: LRScheduler> {
    feature: AdamWOptimizer<S>,
    output: AdamWOptimizer<S>,
}

impl<S: LRScheduler> Optimizers<S> {
    fn step(&mut self) {
        self.feature.step();
        self.output.step();
    }
}

#[derive(Default)]
struct GradientStats {
    samples: usize,
    sum: f32,
    min: f32,
    max: f32,
    last: f32,
}

impl GradientStats {
    fn new() -> Self {
        Self {
            samples: 0,
            sum: 0.0,
            min: f32::INFINITY,
            max: 0.0,
            last: 0.0,
        }
    }

    fn update(&mut self, value: f32) {
        self.samples += 1;
        self.sum += value;
        self.min = self.min.min(value);
        self.max = self.max.max(value);
        self.last = value;
    }

    fn avg(&self) -> f32 {
        if self.samples > 0 {
            self.sum / self.samples as f32
        } else {
            0.0
        }
    }
}

fn main() {
    let mut args = env::args();
    args.next();

    let input = args.next().expect("Missing input file");

    let file = File::open(input.clone()).unwrap();
    let count = file.metadata().unwrap().len() as usize / TrainingPosition::SIZE;

    let mut network = ValueNetwork::random();

    let mut momentum = ValueNetwork::zeroed();
    let mut velocity = ValueNetwork::zeroed();

    let batches = count.div_ceil(BATCH_SIZE);
    assert!(batches > 0, "No training positions found in {input}");
    let epochs = TARGET_BATCH_COUNT.div_ceil(batches);
    let total_steps = (epochs * batches) as u32;
    let feature_scheduler =
        CosineAnnealingLRScheduler::new(FEATURE_LEARNING_RATE, MIN_LEARNING_RATE, total_steps);
    let output_scheduler =
        CosineAnnealingLRScheduler::new(OUTPUT_LEARNING_RATE, MIN_LEARNING_RATE, total_steps);

    let mut optimizers = Optimizers {
        feature: AdamWOptimizer::with_scheduler(feature_scheduler)
            .weight_decay(FEATURE_WEIGHT_DECAY),
        output: AdamWOptimizer::with_scheduler(output_scheduler).weight_decay(OUTPUT_WEIGHT_DECAY),
    };

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    println!("Network: {network}");
    println!("Positions: {count}");
    println!("File: {input}");

    for epoch in 1..=epochs {
        println!("\nEpoch {epoch}/{epochs}...");
        let start = Instant::now();

        train(
            &mut network,
            &mut momentum,
            &mut velocity,
            &mut optimizers,
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
    }
}

fn train<S: LRScheduler>(
    network: &mut ValueNetwork,
    momentum: &mut ValueNetwork,
    velocity: &mut ValueNetwork,
    optimizers: &mut Optimizers<S>,
    input: &str,
    start_time: Instant,
) {
    let file = File::open(input).unwrap();
    let positions = file.metadata().unwrap().len() as usize / TrainingPosition::SIZE;

    let mut buffer = BufReader::with_capacity(TrainingPosition::BUFFER_SIZE, file);

    let mut running_loss = 0.0;
    let mut batch_n = 0;
    let batches = positions.div_ceil(BATCH_SIZE);

    // Gradient tracking stats
    let mut total_grad_stats = GradientStats::new();
    let mut stm_grad_stats = GradientStats::new();
    let mut nstm_grad_stats = GradientStats::new();
    let mut output_grad_stats = GradientStats::new();

    while let Ok(bytes) = buffer.fill_buf() {
        if bytes.is_empty() {
            break;
        }

        let data = TrainingPosition::read_buffer(bytes);

        for batch in data.chunks(BATCH_SIZE) {
            let mut gradients = ValueNetwork::zeroed();
            running_loss += gradients_batch(network, &mut gradients, batch);

            *gradients /= batch.len() as f32;

            // Sample gradient norms every 1000 batches
            if batch_n % 1000 == 0 {
                total_grad_stats.update(gradients.gradient_norm());
                stm_grad_stats.update(gradients.stm_weights_norm());
                nstm_grad_stats.update(gradients.nstm_weights_norm());
                output_grad_stats.update(gradients.output_weights_norm());
            }

            optimizers.step();

            network.train_step(
                &gradients,
                momentum,
                velocity,
                &optimizers.feature,
                &optimizers.output,
            );

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

            print!(
                "B {:>5}/{:>5} | ETA {:>3}m | LR {:.6}/{:.6} | GRAD A {:.4} L {:.4}\r",
                batch_n,
                batches,
                remaining_min,
                optimizers.feature.get_learning_rate(),
                optimizers.output.get_learning_rate(),
                total_grad_stats.avg(),
                total_grad_stats.last
            );
            io::stdout().flush().unwrap();
        }

        let consumed = bytes.len();
        buffer.consume(consumed);
    }

    println!();
    println!("Running loss: {}", running_loss / positions as f32);

    // Log gradient statistics for vanishing gradient detection
    if total_grad_stats.samples > 0 {
        println!(
            "Total gradient stats - Min: {:.6}, Max: {:.6}, Avg: {:.6}, Last: {:.6} (n={})",
            total_grad_stats.min,
            total_grad_stats.max,
            total_grad_stats.avg(),
            total_grad_stats.last,
            total_grad_stats.samples
        );
        println!(
            "STM gradient stats   - Min: {:.6}, Max: {:.6}, Avg: {:.6}, Last: {:.6}",
            stm_grad_stats.min,
            stm_grad_stats.max,
            stm_grad_stats.avg(),
            stm_grad_stats.last
        );
        println!(
            "NSTM gradient stats  - Min: {:.6}, Max: {:.6}, Avg: {:.6}, Last: {:.6}",
            nstm_grad_stats.min,
            nstm_grad_stats.max,
            nstm_grad_stats.avg(),
            nstm_grad_stats.last
        );
        println!(
            "Output gradient stats- Min: {:.6}, Max: {:.6}, Avg: {:.6}, Last: {:.6}",
            output_grad_stats.min,
            output_grad_stats.max,
            output_grad_stats.avg(),
            output_grad_stats.last
        );
    }
}

fn gradients_batch(
    network: &ValueNetwork,
    gradients: &mut ValueNetwork,
    batch: &[TrainingPosition],
) -> f32 {
    let size = (batch.len() / THREADS).max(1);
    let mut loss = [0.0; THREADS];

    let results: Vec<_> = thread::scope(|s| {
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
            .collect()
    });

    for inner_gradients in results {
        *gradients += &inner_gradients;
    }

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

    network.backprop(
        &features,
        gradients,
        Vector::from_raw([2.0 * error]),
        &net_out,
    );
}

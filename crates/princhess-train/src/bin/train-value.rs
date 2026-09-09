use std::fs::{self, File};
use std::io::{self, Write};
use std::ops::AddAssign;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use chrono::Utc;
use toml::{Table, Value};

use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::Color;
use ratatui::widgets::{Block, Borders, Paragraph};
use ratatui::Frame;
use scc::{Guard, Queue};

use princhess::engine::SCALE;
use princhess::state::State;
use princhess_train::args::Args;
use princhess_train::data::{TrainingData, TrainingPosition};
use princhess_train::neural::{
    AdamWOptimizer, FeedForwardNetwork, LRScheduler, OutputLayer, PolynomialWarmupDecayLRScheduler,
    SparseVector, Vector,
};
use princhess_train::system;
use princhess_train::tui;
use princhess_train::value::ValueNetwork;

const BATCHES_PER_SUPER_BATCH: usize = 6_104;
const TOTAL_SUPER_BATCHES: usize = 75;
const BATCH_SIZE: usize = 16384;

const TUI_TOTAL_HEIGHT: u16 = 28;
const SAMPLE_INTERVAL_SECS: u64 = 5;
const MAX_RATE_SAMPLES: usize = 512;
const MAX_LR_SAMPLES: usize = 512;
const LR_SAMPLES_PER_SUPER_BATCH: usize = 5;

const LEARNING_RATE: f32 = 0.001;
const WEIGHT_DECAY: f32 = 0.01;
const WDL_WEIGHT: f32 = 0.3;

const SAVE_EVERY_N_SUPER_BATCHES: usize = 10;

const _BUFFER_SIZE_CHECK: () = assert!(TrainingPosition::BUFFER_SIZE.is_multiple_of(BATCH_SIZE));

#[derive(Debug, Default, Clone, Copy)]
struct BatchMetrics {
    loss: f32,
    processed_count: usize,
}

impl AddAssign for BatchMetrics {
    fn add_assign(&mut self, rhs: Self) {
        self.loss += rhs.loss;
        self.processed_count += rhs.processed_count;
    }
}

#[derive(Clone)]
struct TrainingConfig {
    input_file: String,
    network_info: String,
    data_positions: usize,
    threads: usize,
}

struct TrainingStats {
    start_time: Instant,
    current_super_batch: AtomicUsize,
    current_batch_in_super: AtomicUsize,
    total_positions_processed: AtomicU64,

    // Current super batch running metrics (quantized by SCALE for atomic accumulation)
    current_loss_sum: AtomicI64, // loss * SCALE
    current_count: AtomicUsize,

    // History queues for charting (updated at end of each super batch, single-threaded)
    loss_history: Queue<f32>,

    // Best metrics (updated at end of super batch, single-threaded, stored as quantized i64)
    best_loss: AtomicI64, // loss * SCALE
    best_loss_sb: AtomicUsize,
    prev_loss: AtomicI64, // loss * SCALE

    // Rate tracking
    recent_rates: Queue<u64>,
    last_sample_positions: AtomicU64,

    // Learning rate tracking
    current_lr: AtomicU32, // stored as f32.to_bits()
    lr_history: Queue<f32>,

    // File read progress (positions consumed from current pass through the file)
    positions_consumed: AtomicU64,

    last_saved_net: Mutex<Option<String>>,
}

impl TrainingStats {
    fn new(start_time: Instant) -> Self {
        Self {
            start_time,
            current_super_batch: AtomicUsize::new(0),
            current_batch_in_super: AtomicUsize::new(0),
            total_positions_processed: AtomicU64::new(0),
            current_loss_sum: AtomicI64::new(0),
            current_count: AtomicUsize::new(0),
            loss_history: Queue::default(),
            best_loss: AtomicI64::new(i64::MAX),
            best_loss_sb: AtomicUsize::new(0),
            prev_loss: AtomicI64::new(0),
            recent_rates: Queue::default(),
            last_sample_positions: AtomicU64::new(0),
            current_lr: AtomicU32::new(0),
            lr_history: Queue::default(),
            positions_consumed: AtomicU64::new(0),
            last_saved_net: Mutex::new(None),
        }
    }

    fn record_batch(&self, metrics: BatchMetrics) {
        let loss_scaled = (metrics.loss * SCALE) as i64;

        self.current_loss_sum
            .fetch_add(loss_scaled, Ordering::Relaxed);
        self.current_count
            .fetch_add(metrics.processed_count, Ordering::Relaxed);
        self.total_positions_processed
            .fetch_add(metrics.processed_count as u64, Ordering::Relaxed);
        self.current_batch_in_super.fetch_add(1, Ordering::Relaxed);
    }

    fn finish_super_batch(&self) {
        let sb_num = self.current_super_batch.load(Ordering::Relaxed);
        let is_final_batch = sb_num + 1 >= TOTAL_SUPER_BATCHES;

        let (loss_sum_scaled, count) = if is_final_batch {
            (
                self.current_loss_sum.load(Ordering::Relaxed),
                self.current_count.load(Ordering::Relaxed),
            )
        } else {
            (
                self.current_loss_sum.swap(0, Ordering::Relaxed),
                self.current_count.swap(0, Ordering::Relaxed),
            )
        };

        if count > 0 {
            let avg_loss = loss_sum_scaled as f32 / SCALE / count as f32;

            self.loss_history.push(avg_loss);

            let avg_loss_scaled = (avg_loss * SCALE) as i64;
            self.prev_loss.store(avg_loss_scaled, Ordering::Relaxed);

            let best_loss_scaled = self.best_loss.load(Ordering::Relaxed);
            if avg_loss_scaled < best_loss_scaled {
                self.best_loss.store(avg_loss_scaled, Ordering::Relaxed);
                self.best_loss_sb.store(sb_num, Ordering::Relaxed);
            }
        }

        self.current_super_batch.fetch_add(1, Ordering::Relaxed);
        if !is_final_batch {
            self.current_batch_in_super.store(0, Ordering::Relaxed);
        }
    }

    fn get_prev_loss(&self) -> f32 {
        self.prev_loss.load(Ordering::Relaxed) as f32 / SCALE
    }

    fn get_current_avg_loss(&self) -> f32 {
        let loss_sum_scaled = self.current_loss_sum.load(Ordering::Relaxed);
        let count = self.current_count.load(Ordering::Relaxed);

        if count > 0 {
            loss_sum_scaled as f32 / SCALE / count as f32
        } else {
            0.0
        }
    }
}

fn main() {
    let mut args = Args::from_env();
    let threads = args
        .flag("-t", "--threads")
        .unwrap_or_else(system::default_thread_count) as usize;

    assert!(
        threads > 0,
        "Thread count must be at least 1, got {threads}"
    );

    let input = args.expect("input file");

    let file = File::open(&input).unwrap();
    let data_positions = file.metadata().unwrap().len() as usize / TrainingPosition::SIZE;

    assert!(
        data_positions >= TrainingPosition::BUFFER_COUNT,
        "Input file has {data_positions} positions, need at least {} (BUFFER_COUNT)",
        TrainingPosition::BUFFER_COUNT
    );

    let network = ValueNetwork::random();
    let momentum = ValueNetwork::zeroed();
    let velocity = ValueNetwork::zeroed();

    let config = TrainingConfig {
        input_file: input.clone(),
        network_info: format!("{network}"),
        data_positions,
        threads,
    };

    let total_steps = (TOTAL_SUPER_BATCHES * BATCHES_PER_SUPER_BATCH) as u32;

    let scheduler = PolynomialWarmupDecayLRScheduler::linear(LEARNING_RATE, 0.05, total_steps);
    let optimizer = AdamWOptimizer::with_scheduler(scheduler).weight_decay(WEIGHT_DECAY);

    run_training_loop(network, momentum, velocity, optimizer, config);
}

fn run_training_loop<S: LRScheduler>(
    mut network: Box<ValueNetwork>,
    mut momentum: Box<ValueNetwork>,
    mut velocity: Box<ValueNetwork>,
    mut optimizer: AdamWOptimizer<S>,
    config: TrainingConfig,
) {
    let timestamp = Utc::now().format("%Y%m%d-%H%M").to_string();

    let start_time = Instant::now();
    let stats = Arc::new(TrainingStats::new(start_time));
    let stop_signal = Arc::new(AtomicBool::new(false));

    // Spawn TUI update thread
    let stats_clone = Arc::clone(&stats);
    let stop_clone = Arc::clone(&stop_signal);
    let config_clone = config.clone();
    let tui_thread = thread::spawn(move || {
        if let Err(e) = run_tui(&stats_clone, stop_clone.clone(), config_clone) {
            eprintln!("TUI failed: {e}");
            stop_clone.store(true, Ordering::Relaxed);
        }
    });

    let mut data = TrainingData::new(&config.input_file);

    // Training loop
    for sb in 0..TOTAL_SUPER_BATCHES {
        if stop_signal.load(Ordering::Relaxed) {
            break;
        }

        train_super_batch(
            &mut network,
            &mut momentum,
            &mut velocity,
            &mut optimizer,
            &config,
            &stats,
            &mut data,
        );

        stats.finish_super_batch();

        // Save network periodically
        if (sb + 1) % SAVE_EVERY_N_SUPER_BATCHES == 0 || sb + 1 == TOTAL_SUPER_BATCHES {
            let dir_name = format!("nets/value-{timestamp}-sb{:03}", sb + 1);
            fs::create_dir(&dir_name).expect("Failed to create network save directory");
            let dir = Path::new(&dir_name);
            network.to_boxed_and_quantized().save_to_bin(dir);
            write_training_toml(dir, sb + 1, &stats, &config);

            *stats.last_saved_net.lock().unwrap() = Some(dir_name);
        }
    }

    // Give TUI time to render final state before stopping
    thread::sleep(Duration::from_millis(200));

    // Cleanup TUI
    stop_signal.store(true, Ordering::Relaxed);
    tui_thread.join().unwrap();
}

fn write_training_toml(dir: &Path, sb: usize, stats: &TrainingStats, config: &TrainingConfig) {
    let loss = stats.get_prev_loss();

    let mut doc = Table::new();
    doc.insert(
        "super_batch".into(),
        Value::Integer(i64::try_from(sb).unwrap_or(i64::MAX)),
    );
    doc.insert(
        "super_batches_total".into(),
        Value::Integer(i64::try_from(TOTAL_SUPER_BATCHES).unwrap_or(i64::MAX)),
    );
    doc.insert("network_info".into(), config.network_info.clone().into());
    doc.insert("input_file".into(), config.input_file.clone().into());
    doc.insert("loss".into(), loss.into());

    let path = dir.join("value.toml");
    let mut file = File::create(path).expect("Failed to create training TOML");
    write!(file, "{doc}").expect("Failed to write training TOML");
}

fn run_tui(
    stats: &TrainingStats,
    stop_signal: Arc<AtomicBool>,
    config: TrainingConfig,
) -> io::Result<()> {
    let stop_clone = Arc::clone(&stop_signal);
    let mut last_sample_time = Instant::now();

    tui::run_inline_tui(
        TUI_TOTAL_HEIGHT,
        || stop_signal.load(Ordering::Relaxed),
        || stop_clone.store(true, Ordering::Relaxed),
        || {
            let now = Instant::now();
            let elapsed = now.duration_since(last_sample_time).as_secs();
            if elapsed >= SAMPLE_INTERVAL_SECS {
                let current_positions = stats.total_positions_processed.load(Ordering::Relaxed);
                let last_positions = stats.last_sample_positions.load(Ordering::Relaxed);
                if last_positions > 0 {
                    let positions_diff = current_positions.saturating_sub(last_positions);
                    let rate_per_hour = (positions_diff * 3600) / elapsed;
                    let _ = stats.recent_rates.push(rate_per_hour);
                    while stats.recent_rates.len() > MAX_RATE_SAMPLES {
                        let _ = stats.recent_rates.pop();
                    }
                }
                stats
                    .last_sample_positions
                    .store(current_positions, Ordering::Relaxed);
                last_sample_time = now;
            }
        },
        |f| render_tui(f, stats, &config),
    )
}

fn render_tui(frame: &mut Frame, stats: &TrainingStats, config: &TrainingConfig) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .margin(1)
        .constraints([
            Constraint::Length(8),  // Progress
            Constraint::Length(4),  // Dataset / Network
            Constraint::Length(4),  // Metrics
            Constraint::Length(10), // Loss chart
        ])
        .split(frame.area());

    let elapsed = stats.start_time.elapsed();
    let super_batch = stats.current_super_batch.load(Ordering::Relaxed);
    let batch_in_super = stats.current_batch_in_super.load(Ordering::Relaxed);
    let total_batches_done = super_batch * BATCHES_PER_SUPER_BATCH + batch_in_super;
    let batches_per_sec = if elapsed.as_secs() > 0 {
        total_batches_done as f64 / elapsed.as_secs_f64()
    } else {
        0.0
    };
    let batches_remaining = ((TOTAL_SUPER_BATCHES * BATCHES_PER_SUPER_BATCH) as f32
        - total_batches_done as f32)
        .max(0.0);
    let eta_secs = if batches_per_sec > 0.0 {
        (batches_remaining / batches_per_sec as f32) as u64
    } else {
        0
    };

    let guard = Guard::new();
    tui::render_training_progress(
        frame,
        chunks[0],
        &tui::TrainingProgressView {
            elapsed_secs: elapsed.as_secs(),
            samples_per_sec: batches_per_sec * BATCH_SIZE as f64,
            eta_secs,
            super_batch,
            total_super_batches: TOTAL_SUPER_BATCHES,
            batch_in_super,
            batches_per_super_batch: BATCHES_PER_SUPER_BATCH,
            positions_consumed: stats.positions_consumed.load(Ordering::Relaxed),
            data_positions: config.data_positions as u64,
            recent_rates: stats.recent_rates.iter(&guard).copied().collect(),
            lr_history: stats.lr_history.iter(&guard).copied().collect(),
            lr_samples_per_super_batch: LR_SAMPLES_PER_SUPER_BATCH,
        },
    );

    let last_saved = stats.last_saved_net.lock().unwrap().clone();
    tui::render_dataset_boxes(
        frame,
        chunks[1],
        &tui::DatasetBoxesView {
            input_file: &config.input_file,
            data_positions: config.data_positions,
            phase: None,
            network_info: &config.network_info,
            last_saved: last_saved.as_deref(),
        },
    );

    render_info(frame, chunks[2], stats);

    let loss_history: Vec<f32> = stats.loss_history.iter(&guard).copied().collect();
    tui::render_history_chart(
        frame,
        chunks[3],
        &tui::HistoryChartView {
            title: "Loss",
            data: &loss_history,
            x_bound: TOTAL_SUPER_BATCHES as f64,
            y_range_fallback: (0.0, 1.0),
            y_max_clamp: None,
            y_label_precision: 3,
            color: Color::Red,
        },
    );
}

fn render_info(frame: &mut Frame, area: ratatui::layout::Rect, stats: &TrainingStats) {
    let metrics_block = Block::default().borders(Borders::ALL).title("Metrics");
    let metrics_inner = metrics_block.inner(area);
    frame.render_widget(metrics_block, area);

    let metrics_columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(metrics_inner);

    let current_loss = stats.get_current_avg_loss();
    let lr = f32::from_bits(stats.current_lr.load(Ordering::Relaxed));
    frame.render_widget(
        Paragraph::new(format!("LR:    {:.6}\nLoss:  {:.4}", lr, current_loss)),
        metrics_columns[0],
    );

    let prev_loss = stats.get_prev_loss();
    frame.render_widget(
        Paragraph::new(format!("Prev SB\n{:.4}", prev_loss)),
        metrics_columns[1],
    );
}

fn train_super_batch<S: LRScheduler>(
    network: &mut ValueNetwork,
    momentum: &mut ValueNetwork,
    velocity: &mut ValueNetwork,
    optimizer: &mut AdamWOptimizer<S>,
    config: &TrainingConfig,
    stats: &TrainingStats,
    data: &mut TrainingData,
) {
    let mut batches_processed = 0;

    while batches_processed < BATCHES_PER_SUPER_BATCH {
        let buffer = data.next_buffer();

        for batch in buffer.chunks(BATCH_SIZE) {
            if batches_processed >= BATCHES_PER_SUPER_BATCH {
                break;
            }

            let mut gradients = ValueNetwork::zeroed();

            let batch_metrics = gradients_batch(network, &mut gradients, batch, config.threads);

            *gradients /= batch.len() as f32;

            optimizer.step();

            network.train_step(&gradients, momentum, velocity, optimizer);

            stats.record_batch(batch_metrics);

            // Update current LR
            let lr = optimizer.get_learning_rate();
            stats.current_lr.store(lr.to_bits(), Ordering::Relaxed);

            // Sample LR periodically
            let sample_interval = BATCHES_PER_SUPER_BATCH / LR_SAMPLES_PER_SUPER_BATCH;
            if batches_processed % sample_interval == 0 {
                let _ = stats.lr_history.push(lr);
                while stats.lr_history.len() > MAX_LR_SAMPLES {
                    let _ = stats.lr_history.pop();
                }
            }

            batches_processed += 1;
        }

        stats
            .positions_consumed
            .store(data.positions_consumed(), Ordering::Relaxed);
    }
}

fn gradients_batch(
    network: &ValueNetwork,
    gradients: &mut ValueNetwork,
    batch: &[TrainingPosition],
    threads: usize,
) -> BatchMetrics {
    let size = (batch.len() / threads) + 1;

    let mut total_metrics = BatchMetrics::default();

    thread::scope(|s| {
        batch
            .chunks(size)
            .map(|chunk| {
                s.spawn(move || {
                    let mut inner_gradients = ValueNetwork::zeroed();
                    let mut loss = 0.0;
                    for position in chunk {
                        update_gradient(position, network, &mut inner_gradients, &mut loss);
                    }
                    (
                        inner_gradients,
                        BatchMetrics {
                            loss,
                            processed_count: chunk.len(),
                        },
                    )
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .for_each(|(inner_gradients, inner_metrics)| {
                *gradients += &inner_gradients;
                total_metrics += inner_metrics;
            });
    });

    total_metrics
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

use std::array;
use std::fs::{self, File};
use std::io::{self, Write};
use std::ops::AddAssign;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

use arrayvec::ArrayVec;
use chrono::Utc;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::Color;
use ratatui::widgets::{Block, Borders, Paragraph};
use ratatui::Frame;
use scc::{Guard, Queue};
use toml::{Table, Value};

use princhess::chess::Piece;
use princhess::engine::SCALE;
use princhess::math;
use princhess::state::State;

use princhess_train::analysis::{write_lr_analysis_toml, LrAnalysisConfig};
use princhess_train::args::Args;
use princhess_train::data::TrainingData;
use princhess_train::data::TrainingPosition;
use princhess_train::mg_policy::{is_training_position, MgPolicyNetwork};
use princhess_train::neural::{
    AdamWOptimizer, LRScheduler, PolynomialWarmupDecayLRScheduler, SparseVector,
};
use princhess_train::system;
use princhess_train::tui;

const BATCHES_PER_SUPER_BATCH: usize = 6_104;
const TOTAL_SUPER_BATCHES: usize = 35;
const BATCH_SIZE: usize = 32768;

const TUI_TOTAL_HEIGHT: u16 = 40;
const SAMPLE_INTERVAL_SECS: u64 = 5;
const MAX_RATE_SAMPLES: usize = 512;
const MAX_LR_SAMPLES: usize = 512;
const LR_SAMPLES_PER_SUPER_BATCH: usize = 5;

const LR: f32 = 1e-3;

const SOFT_TARGET_WEIGHT: f32 = 0.1;
const SOFT_TARGET_TEMPERATURE: f32 = 4.0;

const EPSILON: f32 = 1e-9;
const SAVE_EVERY_N_SUPER_BATCHES: usize = 10;

const _BUFFER_SIZE_CHECK: () = assert!(TrainingPosition::BUFFER_COUNT.is_multiple_of(BATCH_SIZE));

#[derive(Debug, Default, Clone, Copy)]
struct BatchMetrics {
    loss: f32,
    accuracy: f32,
    baseline_loss: f32,
    processed_count: usize,
    piece_correct: [usize; Piece::COUNT],
    piece_total: [usize; Piece::COUNT],
    piece_loss: [f32; Piece::COUNT],
    piece_baseline_loss: [f32; Piece::COUNT],
    wrong_piece: [usize; Piece::COUNT],
    wrong_square: [usize; Piece::COUNT],
}

impl AddAssign for BatchMetrics {
    fn add_assign(&mut self, rhs: Self) {
        self.loss += rhs.loss;
        self.accuracy += rhs.accuracy;
        self.baseline_loss += rhs.baseline_loss;
        self.processed_count += rhs.processed_count;
        for i in 0..Piece::COUNT {
            self.piece_correct[i] += rhs.piece_correct[i];
            self.piece_total[i] += rhs.piece_total[i];
            self.piece_loss[i] += rhs.piece_loss[i];
            self.piece_baseline_loss[i] += rhs.piece_baseline_loss[i];
            self.wrong_piece[i] += rhs.wrong_piece[i];
            self.wrong_square[i] += rhs.wrong_square[i];
        }
    }
}

#[derive(Clone)]
struct TrainingConfig {
    input_file: String,
    network_info: String,
    data_positions: usize,
    threads: usize,
    scheduler: String,
}

struct TrainingStats {
    start_time: Instant,
    current_super_batch: AtomicUsize,
    current_batch_in_super: AtomicUsize,
    total_positions_processed: AtomicU64,

    // Current super batch running metrics (quantized by SCALE for atomic accumulation)
    current_loss_sum: AtomicI64,     // loss * SCALE
    current_accuracy_sum: AtomicI64, // accuracy * SCALE
    current_count: AtomicUsize,

    // History queues for charting (updated at end of each super batch, single-threaded)
    loss_history: Queue<f32>,
    accuracy_history: Queue<f32>,

    // Per-batch histories for LR analysis (collected at end of training)
    grad_l1_history: Queue<f32>,
    lr_full_history: Queue<f32>,

    // Rate tracking
    recent_rates: Queue<u64>,
    last_sample_positions: AtomicU64,

    // Learning rate tracking
    current_lr: AtomicU32, // stored as f32.to_bits() (updated each batch)
    lr_history: Queue<f32>,

    // File read progress (bytes consumed from current pass through the file)
    positions_consumed: AtomicU64,

    // Previous super batch metrics (stored as quantized i64)
    prev_loss: AtomicI64,     // loss * SCALE
    prev_accuracy: AtomicI64, // accuracy * SCALE
    prev_baseline: AtomicI64, // baseline * SCALE

    current_baseline_sum: AtomicI64, // sum of ln(num_moves) * SCALE

    piece_correct: [AtomicU64; Piece::COUNT],
    piece_total: [AtomicU64; Piece::COUNT],
    piece_loss_sum: [AtomicI64; Piece::COUNT],
    piece_baseline_sum: [AtomicI64; Piece::COUNT],
    wrong_piece: [AtomicU64; Piece::COUNT],
    wrong_square: [AtomicU64; Piece::COUNT],

    // Snapshots of piece metrics from the last completed super batch (scaled by SCALE)
    prev_piece_accuracy: [AtomicI64; Piece::COUNT],
    prev_piece_info_gain: [AtomicI64; Piece::COUNT],
    prev_wrong_piece: [AtomicI64; Piece::COUNT],
    prev_wrong_square: [AtomicI64; Piece::COUNT],

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
            current_accuracy_sum: AtomicI64::new(0),
            current_count: AtomicUsize::new(0),
            loss_history: Queue::default(),
            accuracy_history: Queue::default(),
            grad_l1_history: Queue::default(),
            lr_full_history: Queue::default(),
            recent_rates: Queue::default(),
            last_sample_positions: AtomicU64::new(0),
            current_lr: AtomicU32::new(0),
            lr_history: Queue::default(),
            positions_consumed: AtomicU64::new(0),
            prev_loss: AtomicI64::new(0),
            prev_accuracy: AtomicI64::new(0),
            prev_baseline: AtomicI64::new(0),
            current_baseline_sum: AtomicI64::new(0),
            piece_correct: array::from_fn(|_| AtomicU64::new(0)),
            piece_total: array::from_fn(|_| AtomicU64::new(0)),
            piece_loss_sum: array::from_fn(|_| AtomicI64::new(0)),
            piece_baseline_sum: array::from_fn(|_| AtomicI64::new(0)),
            wrong_piece: array::from_fn(|_| AtomicU64::new(0)),
            wrong_square: array::from_fn(|_| AtomicU64::new(0)),
            prev_piece_accuracy: array::from_fn(|_| AtomicI64::new(0)),
            prev_piece_info_gain: array::from_fn(|_| AtomicI64::new(0)),
            prev_wrong_piece: array::from_fn(|_| AtomicI64::new(0)),
            prev_wrong_square: array::from_fn(|_| AtomicI64::new(0)),
            last_saved_net: Mutex::new(None),
        }
    }

    fn record_batch(&self, metrics: BatchMetrics) {
        let loss_scaled = (metrics.loss * SCALE) as i64;
        let acc_scaled = (metrics.accuracy * SCALE) as i64;
        let baseline_scaled = (metrics.baseline_loss * SCALE) as i64;

        self.current_loss_sum
            .fetch_add(loss_scaled, Ordering::Relaxed);
        self.current_accuracy_sum
            .fetch_add(acc_scaled, Ordering::Relaxed);
        self.current_baseline_sum
            .fetch_add(baseline_scaled, Ordering::Relaxed);
        self.current_count
            .fetch_add(metrics.processed_count, Ordering::Relaxed);
        self.total_positions_processed
            .fetch_add(metrics.processed_count as u64, Ordering::Relaxed);
        self.current_batch_in_super.fetch_add(1, Ordering::Relaxed);
        for i in 0..Piece::COUNT {
            self.piece_correct[i].fetch_add(metrics.piece_correct[i] as u64, Ordering::Relaxed);
            self.piece_total[i].fetch_add(metrics.piece_total[i] as u64, Ordering::Relaxed);
            self.piece_loss_sum[i]
                .fetch_add((metrics.piece_loss[i] * SCALE) as i64, Ordering::Relaxed);
            self.piece_baseline_sum[i].fetch_add(
                (metrics.piece_baseline_loss[i] * SCALE) as i64,
                Ordering::Relaxed,
            );
            self.wrong_piece[i].fetch_add(metrics.wrong_piece[i] as u64, Ordering::Relaxed);
            self.wrong_square[i].fetch_add(metrics.wrong_square[i] as u64, Ordering::Relaxed);
        }
    }

    fn finish_super_batch(&self) {
        let sb_num = self.current_super_batch.load(Ordering::Relaxed);
        let is_final_batch = sb_num + 1 >= TOTAL_SUPER_BATCHES;

        // Read metrics (use swap for non-final batches, load for final batch to preserve values)
        let (loss_sum_scaled, acc_sum_scaled, baseline_sum_scaled, count) = if is_final_batch {
            (
                self.current_loss_sum.load(Ordering::Relaxed),
                self.current_accuracy_sum.load(Ordering::Relaxed),
                self.current_baseline_sum.load(Ordering::Relaxed),
                self.current_count.load(Ordering::Relaxed),
            )
        } else {
            (
                self.current_loss_sum.swap(0, Ordering::Relaxed),
                self.current_accuracy_sum.swap(0, Ordering::Relaxed),
                self.current_baseline_sum.swap(0, Ordering::Relaxed),
                self.current_count.swap(0, Ordering::Relaxed),
            )
        };

        if count > 0 {
            let avg_loss = loss_sum_scaled as f32 / SCALE / count as f32;
            let avg_accuracy = acc_sum_scaled as f32 / SCALE / count as f32;
            let avg_baseline = baseline_sum_scaled as f32 / SCALE / count as f32;

            self.loss_history.push(avg_loss);
            self.accuracy_history.push(avg_accuracy);

            // Store as previous super batch metrics
            self.prev_loss
                .store((avg_loss * SCALE) as i64, Ordering::Relaxed);
            self.prev_accuracy
                .store((avg_accuracy * SCALE) as i64, Ordering::Relaxed);
            self.prev_baseline
                .store((avg_baseline * SCALE) as i64, Ordering::Relaxed);
        }

        self.current_super_batch.fetch_add(1, Ordering::Relaxed);

        // Snapshot piece metrics before reset
        let piece_acc = self.get_piece_accuracy();
        let piece_ig = self.get_piece_info_gain();
        let (wp, ws) = self.get_piece_error_breakdown();
        for i in 0..Piece::COUNT {
            self.prev_piece_accuracy[i].store((piece_acc[i] * SCALE) as i64, Ordering::Relaxed);
            self.prev_piece_info_gain[i].store((piece_ig[i] * SCALE) as i64, Ordering::Relaxed);
            self.prev_wrong_piece[i].store((wp[i] * SCALE) as i64, Ordering::Relaxed);
            self.prev_wrong_square[i].store((ws[i] * SCALE) as i64, Ordering::Relaxed);
        }

        // Don't reset progress counters if we've completed all super batches
        // This preserves the final display state
        if !is_final_batch {
            self.current_batch_in_super.store(0, Ordering::Relaxed);
            for i in 0..Piece::COUNT {
                self.piece_correct[i].store(0, Ordering::Relaxed);
                self.piece_total[i].store(0, Ordering::Relaxed);
                self.piece_loss_sum[i].store(0, Ordering::Relaxed);
                self.piece_baseline_sum[i].store(0, Ordering::Relaxed);
                self.wrong_piece[i].store(0, Ordering::Relaxed);
                self.wrong_square[i].store(0, Ordering::Relaxed);
            }
        }
    }

    fn piece_ratio(&self, num: &AtomicU64, i: usize) -> f32 {
        let total = self.piece_total[i].load(Ordering::Relaxed);
        if total > 0 {
            num.load(Ordering::Relaxed) as f32 / total as f32
        } else {
            0.0
        }
    }

    fn get_piece_accuracy(&self) -> [f32; Piece::COUNT] {
        array::from_fn(|i| self.piece_ratio(&self.piece_correct[i], i))
    }

    fn get_piece_error_breakdown(&self) -> ([f32; Piece::COUNT], [f32; Piece::COUNT]) {
        let mut wrong_piece = [0.0f32; Piece::COUNT];
        let mut wrong_square = [0.0f32; Piece::COUNT];
        for i in 0..Piece::COUNT {
            wrong_piece[i] = self.piece_ratio(&self.wrong_piece[i], i);
            wrong_square[i] = self.piece_ratio(&self.wrong_square[i], i);
        }
        (wrong_piece, wrong_square)
    }

    fn get_current_avg_metrics(&self) -> (f32, f32) {
        let loss_sum_scaled = self.current_loss_sum.load(Ordering::Relaxed);
        let acc_sum_scaled = self.current_accuracy_sum.load(Ordering::Relaxed);
        let count = self.current_count.load(Ordering::Relaxed);

        if count > 0 {
            let avg_loss = loss_sum_scaled as f32 / SCALE / count as f32;
            let avg_accuracy = acc_sum_scaled as f32 / SCALE / count as f32;
            (avg_loss, avg_accuracy)
        } else {
            (0.0, 0.0)
        }
    }

    fn get_prev_avg_metrics(&self) -> (f32, f32) {
        let loss_scaled = self.prev_loss.load(Ordering::Relaxed);
        let acc_scaled = self.prev_accuracy.load(Ordering::Relaxed);
        (loss_scaled as f32 / SCALE, acc_scaled as f32 / SCALE)
    }

    fn get_current_info_gain(&self) -> f32 {
        let baseline = self.current_baseline_sum.load(Ordering::Relaxed);
        let loss = self.current_loss_sum.load(Ordering::Relaxed);
        let count = self.current_count.load(Ordering::Relaxed);
        if count > 0 {
            (baseline - loss) as f32 / SCALE / count as f32
        } else {
            0.0
        }
    }

    fn get_prev_info_gain(&self) -> f32 {
        let baseline = self.prev_baseline.load(Ordering::Relaxed) as f32 / SCALE;
        let loss = self.prev_loss.load(Ordering::Relaxed) as f32 / SCALE;
        baseline - loss
    }

    fn get_prev_piece_data(
        &self,
    ) -> (
        [f32; Piece::COUNT],
        [f32; Piece::COUNT],
        [f32; Piece::COUNT],
        [f32; Piece::COUNT],
    ) {
        let load = |arr: &[AtomicI64; Piece::COUNT]| -> [f32; Piece::COUNT] {
            array::from_fn(|i| arr[i].load(Ordering::Relaxed) as f32 / SCALE)
        };
        (
            load(&self.prev_piece_accuracy),
            load(&self.prev_piece_info_gain),
            load(&self.prev_wrong_piece),
            load(&self.prev_wrong_square),
        )
    }

    fn get_piece_info_gain(&self) -> [f32; Piece::COUNT] {
        let total: u64 = (0..Piece::COUNT)
            .map(|i| self.piece_total[i].load(Ordering::Relaxed))
            .sum();
        if total == 0 {
            return [0.0; Piece::COUNT];
        }
        array::from_fn(|i| {
            let baseline = self.piece_baseline_sum[i].load(Ordering::Relaxed) as f32 / SCALE;
            let loss = self.piece_loss_sum[i].load(Ordering::Relaxed) as f32 / SCALE;
            (baseline - loss) / total as f32
        })
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
    let data = TrainingData::new(&input);
    let data_positions = data.positions();

    let network = MgPolicyNetwork::random();
    let momentum = MgPolicyNetwork::zeroed();
    let velocity = MgPolicyNetwork::zeroed();

    let total_steps = (TOTAL_SUPER_BATCHES * BATCHES_PER_SUPER_BATCH) as u32;
    let scheduler = PolynomialWarmupDecayLRScheduler::new(LR, 0.0, total_steps, 1.1);
    let config = TrainingConfig {
        input_file: input.clone(),
        network_info: format!("{network}"),
        data_positions,
        threads,
        scheduler: format!("{scheduler}"),
    };
    let optimizer = AdamWOptimizer::with_scheduler(scheduler).weight_decay(0.01);
    run_training_loop(network, momentum, velocity, optimizer, data, config);
}

fn run_training_loop<S: LRScheduler + Sync>(
    mut network: Box<MgPolicyNetwork>,
    mut momentum: Box<MgPolicyNetwork>,
    mut velocity: Box<MgPolicyNetwork>,
    mut optimizer: AdamWOptimizer<S>,
    mut data: TrainingData,
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

        // Save network periodically (always save after first super batch for sanity checks)
        if (sb + 1) % SAVE_EVERY_N_SUPER_BATCHES == 0 || sb + 1 == TOTAL_SUPER_BATCHES || sb == 0 {
            let dir_name = format!("nets/mg-policy-{timestamp}-sb{:03}", sb + 1);
            fs::create_dir_all(&dir_name).expect("Failed to create network save directory");
            let dir = Path::new(&dir_name);
            network
                .to_boxed_and_quantized()
                .save_to_bin(dir, "mg-policy.bin");
            write_training_toml(dir, sb + 1, &stats, &config);

            *stats.last_saved_net.lock().unwrap() = Some(dir_name);
        }
    }

    // Cleanup TUI
    stop_signal.store(true, Ordering::Relaxed);
    tui_thread.join().unwrap();

    let last_dir = stats.last_saved_net.lock().unwrap().clone();
    if let Some(dir) = last_dir {
        let guard = scc::Guard::new();
        let grad_l1: Vec<f32> = stats.grad_l1_history.iter(&guard).copied().collect();
        let lr_full: Vec<f32> = stats.lr_full_history.iter(&guard).copied().collect();
        let analysis_config = LrAnalysisConfig {
            name: "mg-policy",
            input_file: &config.input_file,
            network_info: &config.network_info,
            data_positions: config.data_positions,
            scheduler: &config.scheduler,
            total_super_batches: TOTAL_SUPER_BATCHES,
        };
        write_lr_analysis_toml(Path::new(&dir), &analysis_config, &grad_l1, &lr_full);
    }
}

fn write_training_toml(dir: &Path, sb: usize, stats: &TrainingStats, config: &TrainingConfig) {
    let (loss, accuracy) = stats.get_prev_avg_metrics();
    let info_gain = stats.get_prev_info_gain();
    let (piece_accuracy, piece_info_gain, wrong_piece, wrong_square) = stats.get_prev_piece_data();
    let piece_names = ["p", "n", "b", "r", "q", "k"];

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
    doc.insert("accuracy".into(), accuracy.into());
    doc.insert("info_gain".into(), info_gain.into());

    let mut pieces = Table::new();
    for (i, name) in piece_names.iter().enumerate() {
        let mut p = Table::new();
        p.insert("accuracy".into(), piece_accuracy[i].into());
        p.insert("info_gain".into(), piece_info_gain[i].into());
        p.insert("wrong_piece".into(), wrong_piece[i].into());
        p.insert("wrong_square".into(), wrong_square[i].into());
        pieces.insert((*name).into(), Value::Table(p));
    }
    doc.insert("piece".into(), Value::Table(pieces));

    let path = dir.join("mg-policy.toml");
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
            Constraint::Length(8),  // Progress section (with 2 sparklines)
            Constraint::Length(4),  // Dataset / Network info boxes
            Constraint::Length(6),  // Metrics / Piece Accuracy boxes
            Constraint::Length(10), // Loss chart
            Constraint::Length(10), // Accuracy chart
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
            phase: Some("mg"),
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

    let acc_history: Vec<f32> = stats
        .accuracy_history
        .iter(&guard)
        .map(|v| v * 100.0)
        .collect();
    tui::render_history_chart(
        frame,
        chunks[4],
        &tui::HistoryChartView {
            title: "Accuracy",
            data: &acc_history,
            x_bound: TOTAL_SUPER_BATCHES as f64,
            y_range_fallback: (0.0, 100.0),
            y_max_clamp: Some(100.0),
            y_label_precision: 1,
            color: Color::Green,
        },
    );
}

fn render_info(frame: &mut Frame, area: ratatui::layout::Rect, stats: &TrainingStats) {
    let columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area);

    let left_block = Block::default().borders(Borders::ALL).title("Metrics");
    frame.render_widget(left_block, columns[0]);

    let left_inner = columns[0].inner(ratatui::layout::Margin {
        horizontal: 1,
        vertical: 1,
    });

    let metrics_columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(left_inner);

    let (current_loss, current_acc) = stats.get_current_avg_metrics();
    let (prev_loss, prev_acc) = stats.get_prev_avg_metrics();
    let current_info_gain = stats.get_current_info_gain();
    let prev_info_gain = stats.get_prev_info_gain();
    let current_lr = f32::from_bits(stats.current_lr.load(Ordering::Relaxed));
    let current_content = format!(
        "LR:        {:9.6}\nLoss:      {:7.4}\nAccuracy:  {:5.2}%\nInfo gain: {:7.4}",
        current_lr,
        current_loss,
        current_acc * 100.0,
        current_info_gain,
    );
    frame.render_widget(Paragraph::new(current_content), metrics_columns[0]);

    let prev_content = format!(
        "Prev SB\n{:7.4}\n{:5.2}%\n{:7.4}",
        prev_loss,
        prev_acc * 100.0,
        prev_info_gain
    );
    frame.render_widget(Paragraph::new(prev_content), metrics_columns[1]);

    let right_block = Block::default()
        .borders(Borders::ALL)
        .title("Piece Accuracy");
    frame.render_widget(right_block, columns[1]);

    let right_inner = columns[1].inner(ratatui::layout::Margin {
        horizontal: 1,
        vertical: 1,
    });

    let piece_rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
        ])
        .split(right_inner);

    let piece_names = ["P", "N", "B", "R", "Q", "K"];
    let fmt_labeled = |vals: &[f32; Piece::COUNT]| -> String {
        piece_names
            .iter()
            .zip(vals.iter())
            .map(|(name, v)| format!("{name} {:5.2}%", v * 100.0))
            .collect::<Vec<_>>()
            .join("  ")
    };
    let fmt_values = |vals: &[f32; Piece::COUNT]| -> String {
        vals.iter()
            .map(|v| format!("  {:5.2}%", v * 100.0))
            .collect::<Vec<_>>()
            .join("  ")
    };
    let fmt_info_gain = |vals: &[f32; Piece::COUNT]| -> String {
        vals.iter()
            .map(|v| format!("  {:6.3}", v))
            .collect::<Vec<_>>()
            .join("  ")
    };

    let piece_acc = stats.get_piece_accuracy();
    let (wrong_piece, wrong_square) = stats.get_piece_error_breakdown();
    let piece_info_gain = stats.get_piece_info_gain();

    frame.render_widget(
        Paragraph::new(format!("Correct:    {}", fmt_labeled(&piece_acc))),
        piece_rows[0],
    );
    frame.render_widget(
        Paragraph::new(format!("Wrong piece:{}", fmt_values(&wrong_piece))),
        piece_rows[1],
    );
    frame.render_widget(
        Paragraph::new(format!("Wrong sq:   {}", fmt_values(&wrong_square))),
        piece_rows[2],
    );
    frame.render_widget(
        Paragraph::new(format!("Info gain:  {}", fmt_info_gain(&piece_info_gain))),
        piece_rows[3],
    );
}

#[allow(clippy::too_many_arguments)]
fn train_super_batch<S: LRScheduler + Sync>(
    network: &mut MgPolicyNetwork,
    momentum: &mut MgPolicyNetwork,
    velocity: &mut MgPolicyNetwork,
    optimizer: &mut AdamWOptimizer<S>,
    config: &TrainingConfig,
    stats: &TrainingStats,
    data: &mut TrainingData,
) {
    let mut thread_buffers: Vec<Box<MgPolicyNetwork>> = (0..config.threads)
        .map(|_| MgPolicyNetwork::zeroed())
        .collect();
    let mut gradients = MgPolicyNetwork::zeroed();

    let mut batches_processed = 0;

    while batches_processed < BATCHES_PER_SUPER_BATCH {
        let buffer = data.next_buffer();

        for batch in buffer.chunks(BATCH_SIZE) {
            if batches_processed >= BATCHES_PER_SUPER_BATCH {
                break;
            }

            gradients.zero_out();

            let batch_metrics =
                gradients_batch(network, &mut gradients, batch, &mut thread_buffers, config);

            stats.record_batch(batch_metrics);
            let _ = stats.grad_l1_history.push(gradients.l1_norm());

            optimizer.step();
            network.adamw(&gradients, momentum, velocity, optimizer);

            // Update current LR
            let current_lr = optimizer.get_learning_rate();
            stats
                .current_lr
                .store(current_lr.to_bits(), Ordering::Relaxed);
            let _ = stats.lr_full_history.push(current_lr);

            // Sample LR periodically
            let sample_interval = BATCHES_PER_SUPER_BATCH / LR_SAMPLES_PER_SUPER_BATCH;
            if batches_processed % sample_interval == 0 {
                let _ = stats.lr_history.push(current_lr);
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
    network: &MgPolicyNetwork,
    gradients: &mut MgPolicyNetwork,
    batch: &[TrainingPosition],
    thread_buffers: &mut [Box<MgPolicyNetwork>],
    config: &TrainingConfig,
) -> BatchMetrics {
    let size = (batch.len() / config.threads) + 1;
    let num_chunks = batch.chunks(size).count();
    let mut thread_metrics = vec![BatchMetrics::default(); num_chunks];

    for g in thread_buffers.iter_mut() {
        g.zero_out();
    }

    thread::scope(|s| {
        batch
            .chunks(size)
            .zip(thread_buffers.iter_mut())
            .zip(thread_metrics.iter_mut())
            .for_each(|((chunk, inner_gradients), inner_metrics)| {
                s.spawn(move || {
                    for position in chunk {
                        update_gradient(position, network, inner_gradients, inner_metrics);
                    }
                });
            });
    });

    let mut total_metrics = BatchMetrics::default();
    for (inner_gradients, inner_metrics) in thread_buffers.iter().zip(thread_metrics) {
        *gradients += inner_gradients;
        total_metrics += inner_metrics;
    }
    if total_metrics.processed_count > 0 {
        *gradients /= total_metrics.processed_count as f32;
    }
    total_metrics
}

fn update_gradient(
    position: &TrainingPosition,
    network: &MgPolicyNetwork,
    gradients: &mut MgPolicyNetwork,
    metrics: &mut BatchMetrics,
) {
    let state = State::from(position);

    if !is_training_position(&state) {
        return;
    }

    let moves = position.moves();

    let mut features = SparseVector::with_capacity(64);
    state.policy_features_map(|feature| features.push(feature));

    let only_moves = moves.iter().map(|(mv, _)| *mv).collect();
    let move_idxes = state.moves_to_indexes(&only_moves).collect::<Vec<_>>();

    let mut raw_outputs = vec![0.0; moves.len()];
    let cache = network.get_all_with_layers(&features, &move_idxes, &mut raw_outputs);

    let mut actual_policy = raw_outputs;
    math::softmax(&mut actual_policy, 1.0);

    let raw_counts: ArrayVec<f32, { TrainingPosition::MAX_MOVES }> =
        moves.iter().map(|(_, v)| f32::from(*v)).collect();

    let expected_primary = calculate_target(&raw_counts, 1.0);
    let expected_secondary = calculate_target(&raw_counts, SOFT_TARGET_TEMPERATURE);

    let mut position_loss = 0.0f32;
    let mut errors = ArrayVec::<f32, { TrainingPosition::MAX_MOVES }>::new();
    for idx in 0..moves.len() {
        let actual_val = actual_policy[idx];
        let log_actual_val = actual_val.max(EPSILON).ln();

        let expected_primary_val = expected_primary[idx];
        let expected_secondary_val = expected_secondary[idx];

        position_loss -= expected_primary_val * log_actual_val;
        position_loss -= expected_secondary_val * log_actual_val * SOFT_TARGET_WEIGHT;

        errors.push(
            (actual_val - expected_primary_val)
                + (actual_val - expected_secondary_val) * SOFT_TARGET_WEIGHT,
        );
    }

    network.backprop_position(&features, gradients, &move_idxes, &errors, &cache);

    let baseline = (moves.len() as f32).ln() * (1.0 + SOFT_TARGET_WEIGHT);
    metrics.loss += position_loss;
    metrics.baseline_loss += baseline;

    for idx in 0..moves.len() {
        let piece = move_idxes[idx].piece();
        let t_i = expected_primary[idx];
        let log_p_i = actual_policy[idx].max(EPSILON).ln();
        // Only primary target — soft-target contribution is excluded, so
        // sum(piece_info_gains) will slightly exceed aggregate info_gain.
        metrics.piece_loss[piece] += t_i * (-log_p_i);
        metrics.piece_baseline_loss[piece] += t_i * (moves.len() as f32).ln();
    }

    let expected_best = argmax(&expected_primary);
    let predicted_best = argmax(&actual_policy);
    let piece = move_idxes[expected_best].piece();
    metrics.piece_total[piece] += 1;
    if predicted_best == expected_best {
        metrics.accuracy += 1.;
        metrics.piece_correct[piece] += 1;
    } else if move_idxes[predicted_best].piece() == piece {
        metrics.wrong_square[piece] += 1;
    } else {
        metrics.wrong_piece[piece] += 1;
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

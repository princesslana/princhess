use std::fs::{self, File};
use std::io::{self, Read, Seek, SeekFrom};
use std::ops::AddAssign;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use arrayvec::ArrayVec;
use chrono::Utc;
use crossterm::cursor;
use crossterm::event::{poll, read, Event, KeyCode};
use crossterm::ExecutableCommand;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Alignment, Constraint, Direction, Layout};
use ratatui::style::{Color, Style};
use ratatui::text::Span;
use ratatui::widgets::{
    Axis, Block, Borders, Chart, Dataset, Gauge, GraphType, Paragraph, Sparkline,
};
use ratatui::{Frame, Terminal, TerminalOptions, Viewport};
use scc::{Guard, Queue};

use princhess::chess::Piece;
use princhess::engine::SCALE;
use princhess::math;
use princhess::state::State;

use princhess_train::args::Args;
use princhess_train::data::TrainingPosition;
use princhess_train::neural::{
    AdamWOptimizer, LRScheduler, LinearWarmupDecayLRScheduler, SparseVector,
};
use princhess_train::policy::{Phase, PolicyNetwork};
use princhess_train::system;
use princhess_train::tui::{self, RawModeGuard};

const BATCHES_PER_SUPER_BATCH: usize = 6_104;
const TOTAL_SUPER_BATCHES: usize = 50;
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

const _BUFFER_SIZE_CHECK: () = assert!(TrainingPosition::BUFFER_SIZE.is_multiple_of(BATCH_SIZE));

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
    phase: Phase,
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

    // Rate tracking
    recent_rates: Queue<u64>,
    last_sample_positions: AtomicU64,

    // Learning rate tracking
    current_lr: AtomicU32, // stored as f32.to_bits() (updated each batch)
    lr_history: Queue<f32>,

    // File read progress (bytes consumed from current pass through the file)
    file_bytes_consumed: AtomicU64,

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
            recent_rates: Queue::default(),
            last_sample_positions: AtomicU64::new(0),
            current_lr: AtomicU32::new(0),
            lr_history: Queue::default(),
            file_bytes_consumed: AtomicU64::new(0),
            prev_loss: AtomicI64::new(0),
            prev_accuracy: AtomicI64::new(0),
            prev_baseline: AtomicI64::new(0),
            current_baseline_sum: AtomicI64::new(0),
            piece_correct: std::array::from_fn(|_| AtomicU64::new(0)),
            piece_total: std::array::from_fn(|_| AtomicU64::new(0)),
            piece_loss_sum: std::array::from_fn(|_| AtomicI64::new(0)),
            piece_baseline_sum: std::array::from_fn(|_| AtomicI64::new(0)),
            wrong_piece: std::array::from_fn(|_| AtomicU64::new(0)),
            wrong_square: std::array::from_fn(|_| AtomicU64::new(0)),
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
            self.piece_loss_sum[i].fetch_add((metrics.piece_loss[i] * SCALE) as i64, Ordering::Relaxed);
            self.piece_baseline_sum[i].fetch_add((metrics.piece_baseline_loss[i] * SCALE) as i64, Ordering::Relaxed);
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
            self.prev_loss.store((avg_loss * SCALE) as i64, Ordering::Relaxed);
            self.prev_accuracy.store((avg_accuracy * SCALE) as i64, Ordering::Relaxed);
            self.prev_baseline.store((avg_baseline * SCALE) as i64, Ordering::Relaxed);
        }

        self.current_super_batch.fetch_add(1, Ordering::Relaxed);

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

    fn get_piece_accuracy(&self) -> [f32; Piece::COUNT] {
        std::array::from_fn(|i| {
            let total = self.piece_total[i].load(Ordering::Relaxed);
            if total > 0 {
                self.piece_correct[i].load(Ordering::Relaxed) as f32 / total as f32
            } else {
                0.0
            }
        })
    }

    fn get_piece_error_breakdown(&self) -> ([f32; Piece::COUNT], [f32; Piece::COUNT]) {
        let wrong_piece = std::array::from_fn(|i| {
            let total = self.piece_total[i].load(Ordering::Relaxed);
            if total > 0 {
                self.wrong_piece[i].load(Ordering::Relaxed) as f32 / total as f32
            } else {
                0.0
            }
        });
        let wrong_square = std::array::from_fn(|i| {
            let total = self.piece_total[i].load(Ordering::Relaxed);
            if total > 0 {
                self.wrong_square[i].load(Ordering::Relaxed) as f32 / total as f32
            } else {
                0.0
            }
        });
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

    fn get_piece_info_gain(&self) -> [f32; Piece::COUNT] {
        std::array::from_fn(|i| {
            let total = self.piece_total[i].load(Ordering::Relaxed);
            if total > 0 {
                let baseline = self.piece_baseline_sum[i].load(Ordering::Relaxed) as f32 / SCALE;
                let loss = self.piece_loss_sum[i].load(Ordering::Relaxed) as f32 / SCALE;
                (baseline - loss) / total as f32
            } else {
                0.0
            }
        })
    }
}

fn main() {
    let mut args = Args::from_env();
    let threads = args
        .flag("-t", "--threads")
        .unwrap_or_else(system::default_thread_count) as usize;

    let phase_arg: String = args
        .flag("-p", "--phase")
        .unwrap_or_else(|| panic!("Missing required flag: -p <mg|eg>"));
    let phase = Phase::from_arg(&phase_arg)
        .unwrap_or_else(|| panic!("Invalid phase: {phase_arg}. Use 'mg' or 'eg'."));

    let input = args.expect("input file");

    let file = File::open(&input).unwrap();
    let data_positions = file.metadata().unwrap().len() as usize / TrainingPosition::SIZE;

    let network = PolicyNetwork::random();
    let momentum = PolicyNetwork::zeroed();
    let velocity = PolicyNetwork::zeroed();

    let config = TrainingConfig {
        input_file: input.clone(),
        network_info: format!("{network}"),
        data_positions,
        threads,
        phase,
    };

    let total_steps = (TOTAL_SUPER_BATCHES * BATCHES_PER_SUPER_BATCH) as u32;
    let scheduler = LinearWarmupDecayLRScheduler::new(LR, 0.05, total_steps);
    let weight_decay = if phase == Phase::MiddleGame { 0.01 } else { 0.0 };
    let optimizer = AdamWOptimizer::with_scheduler(scheduler).weight_decay(weight_decay);
    run_training_loop(network, momentum, velocity, optimizer, config);
}

fn run_training_loop<S: LRScheduler + Sync>(
    mut network: Box<PolicyNetwork>,
    mut momentum: Box<PolicyNetwork>,
    mut velocity: Box<PolicyNetwork>,
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

    let mut file = File::open(&config.input_file).unwrap();

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
            &mut file,
            config.phase,
        );

        stats.finish_super_batch();

        // Save network periodically (always save after first super batch for sanity checks)
        if (sb + 1) % SAVE_EVERY_N_SUPER_BATCHES == 0 || sb + 1 == TOTAL_SUPER_BATCHES || sb == 0 {
            let phase = config.phase;
            let dir_name = format!("nets/{phase}-policy-{timestamp}-sb{:03}", sb + 1);
            fs::create_dir_all(&dir_name).expect("Failed to create network save directory");
            let dir = Path::new(&dir_name);
            network
                .to_boxed_and_quantized()
                .save_to_bin(dir, &format!("{phase}-policy.bin"));

            *stats.last_saved_net.lock().unwrap() = Some(dir_name);
        }
    }

    // Cleanup TUI
    stop_signal.store(true, Ordering::Relaxed);
    tui_thread.join().unwrap();
}

fn run_tui(
    stats: &TrainingStats,
    stop_signal: Arc<AtomicBool>,
    config: TrainingConfig,
) -> io::Result<()> {
    let stdout = io::stdout();
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::with_options(
        backend,
        TerminalOptions {
            viewport: Viewport::Inline(TUI_TOTAL_HEIGHT),
        },
    )?;

    let _guard = RawModeGuard::enable()?;

    let result = (|| -> io::Result<()> {
        let mut last_sample_time = Instant::now();

        loop {
            if stop_signal.load(Ordering::Relaxed) {
                break;
            }

            terminal.draw(|f| render_tui(f, stats, &config))?;

            // Update rate samples periodically
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

            // Check for key presses (Ctrl-C to quit)
            if poll(Duration::from_millis(100))? {
                if let Event::Key(key) = read()? {
                    if key.code == KeyCode::Char('c')
                        && key
                            .modifiers
                            .contains(crossterm::event::KeyModifiers::CONTROL)
                    {
                        stop_signal.store(true, Ordering::Relaxed);
                        break;
                    }
                }
            }
        }

        // Final draw to show completed state
        terminal.draw(|f| render_tui(f, stats, &config))?;

        Ok(())
    })();

    // Position cursor at the end of the viewport
    let viewport_area = terminal.get_frame().area();
    io::stdout().execute(cursor::MoveTo(0, viewport_area.bottom()))?;
    io::stdout().execute(cursor::Show)?;
    result
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

    render_progress(frame, chunks[0], stats, config);
    render_dataset_boxes(frame, chunks[1], stats, config);
    render_info(frame, chunks[2], stats);
    render_loss_chart(frame, chunks[3], stats);
    render_accuracy_chart(frame, chunks[4], stats);
}

fn render_dataset_boxes(
    frame: &mut Frame,
    area: ratatui::layout::Rect,
    stats: &TrainingStats,
    config: &TrainingConfig,
) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area);

    let dataset_block = Block::default().borders(Borders::ALL).title("Dataset");
    let dataset_inner = dataset_block.inner(cols[0]);
    frame.render_widget(dataset_block, cols[0]);
    frame.render_widget(
        Paragraph::new(format!(
            "Input: {}\nPositions: {:.1}M  Phase: {}",
            config.input_file,
            config.data_positions as f64 / 1_000_000.0,
            config.phase,
        )),
        dataset_inner,
    );

    let network_block = Block::default().borders(Borders::ALL).title("Network");
    let network_inner = network_block.inner(cols[1]);
    frame.render_widget(network_block, cols[1]);
    let last_saved = stats
        .last_saved_net
        .lock()
        .unwrap()
        .clone()
        .unwrap_or_else(|| "None".to_string());
    frame.render_widget(
        Paragraph::new(format!(
            "{}\nLast saved: {}",
            config.network_info, last_saved
        )),
        network_inner,
    );
}

fn render_progress(
    frame: &mut Frame,
    area: ratatui::layout::Rect,
    stats: &TrainingStats,
    config: &TrainingConfig,
) {
    let block = Block::default().borders(Borders::ALL).title("Progress");
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // Time/rate/ETA info
            Constraint::Length(1), // Overall progress bar
            Constraint::Length(1), // Current super batch progress bar
            Constraint::Length(1), // File read progress bar
            Constraint::Length(1), // Sparkline for processing rate
            Constraint::Length(1), // Sparkline for learning rate
        ])
        .split(inner);

    // Time info
    let elapsed = stats.start_time.elapsed();
    let total_batches_done = stats.current_super_batch.load(Ordering::Relaxed)
        * BATCHES_PER_SUPER_BATCH
        + stats.current_batch_in_super.load(Ordering::Relaxed);
    let total_batches = TOTAL_SUPER_BATCHES * BATCHES_PER_SUPER_BATCH;

    let batches_per_sec = if elapsed.as_secs() > 0 {
        total_batches_done as f64 / elapsed.as_secs_f64()
    } else {
        0.0
    };

    let samples_per_sec = batches_per_sec * BATCH_SIZE as f64;

    let batches_remaining = (total_batches as f32 - total_batches_done as f32).max(0.0);
    let eta_secs = if batches_per_sec > 0.0 {
        (batches_remaining / batches_per_sec as f32) as u64
    } else {
        0
    };

    let time_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(33),
            Constraint::Percentage(34),
            Constraint::Percentage(33),
        ])
        .split(layout[0]);

    frame.render_widget(
        Paragraph::new(tui::format_elapsed(elapsed.as_secs())).alignment(Alignment::Left),
        time_chunks[0],
    );

    frame.render_widget(
        Paragraph::new(format!("{:.1}K pos/sec", samples_per_sec / 1000.0))
            .alignment(Alignment::Center),
        time_chunks[1],
    );

    frame.render_widget(
        Paragraph::new(tui::format_eta(eta_secs)).alignment(Alignment::Right),
        time_chunks[2],
    );

    // Overall progress
    let current_sb = stats.current_super_batch.load(Ordering::Relaxed);
    let overall_ratio = current_sb as f64 / TOTAL_SUPER_BATCHES as f64;
    let overall_label = Span::styled(
        format!(
            "{:>6} / {:>6}  ({:>5.1}%)",
            current_sb,
            TOTAL_SUPER_BATCHES,
            overall_ratio * 100.0,
        ),
        Style::default().fg(Color::White),
    );
    frame.render_widget(
        Gauge::default()
            .gauge_style(Style::default().fg(Color::Green))
            .ratio(overall_ratio.min(1.0))
            .label(overall_label),
        layout[1],
    );

    // Current super batch progress
    let batch_in_sb = stats.current_batch_in_super.load(Ordering::Relaxed);
    let sb_ratio = batch_in_sb as f64 / BATCHES_PER_SUPER_BATCH as f64;
    let sb_label = Span::styled(
        format!(
            "{:>6} / {:>6}  ({:>5.1}%)",
            batch_in_sb,
            BATCHES_PER_SUPER_BATCH,
            sb_ratio * 100.0,
        ),
        Style::default().fg(Color::White),
    );
    frame.render_widget(
        Gauge::default()
            .gauge_style(Style::default().fg(Color::Blue))
            .ratio(sb_ratio.min(1.0))
            .label(sb_label),
        layout[2],
    );

    // File read progress
    let file_total_bytes = config.data_positions as u64 * TrainingPosition::SIZE as u64;
    let file_bytes = stats.file_bytes_consumed.load(Ordering::Relaxed);
    let file_ratio = if file_total_bytes > 0 {
        (file_bytes as f64 / file_total_bytes as f64).min(1.0)
    } else {
        0.0
    };
    let file_label = Span::styled(
        format!(
            "{:>6.1} / {:>6.1}M ({:>5.1}%)",
            file_bytes as f64 / TrainingPosition::SIZE as f64 / 1_000_000.0,
            config.data_positions as f64 / 1_000_000.0,
            file_ratio * 100.0,
        ),
        Style::default().fg(Color::White),
    );
    frame.render_widget(
        Gauge::default()
            .gauge_style(Style::default().fg(Color::Yellow))
            .ratio(file_ratio)
            .label(file_label),
        layout[3],
    );

    // Sparkline for processing rate
    let guard = Guard::new();
    let recent_rates: Vec<u64> = stats.recent_rates.iter(&guard).copied().collect();
    if !recent_rates.is_empty() {
        let max_bars = layout[4].width as usize;
        let data: Vec<u64> = if recent_rates.len() <= max_bars {
            recent_rates
        } else {
            recent_rates
                .iter()
                .rev()
                .take(max_bars)
                .rev()
                .copied()
                .collect()
        };
        let sparkline = Sparkline::default()
            .data(&data)
            .style(Style::default().fg(Color::Cyan));
        frame.render_widget(sparkline, layout[4]);
    }

    // Sparkline for learning rate
    let lr_samples: Vec<f32> = stats.lr_history.iter(&guard).copied().collect();
    if !lr_samples.is_empty() {
        let sparkline_width = layout[5].width as usize;
        let total_expected_samples = TOTAL_SUPER_BATCHES * LR_SAMPLES_PER_SUPER_BATCH;

        // Map full training span to widget width
        let data: Vec<u64> = (0..sparkline_width)
            .map(|i| {
                let sample_idx = (i * total_expected_samples) / sparkline_width;
                if sample_idx < lr_samples.len() {
                    (lr_samples[sample_idx] * 1_000_000.0) as u64
                } else {
                    0 // Haven't reached this point in training yet
                }
            })
            .collect();

        let lr_sparkline = Sparkline::default()
            .data(&data)
            .style(Style::default().fg(Color::Magenta));
        frame.render_widget(lr_sparkline, layout[5]);
    }
}

fn render_info(frame: &mut Frame, area: ratatui::layout::Rect, stats: &TrainingStats) {
    let columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area);

    // Left box: Metrics
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
        prev_loss, prev_acc * 100.0, prev_info_gain
    );
    frame.render_widget(Paragraph::new(prev_content), metrics_columns[1]);

    // Right box: Piece Accuracy
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

fn render_loss_chart(frame: &mut Frame, area: ratatui::layout::Rect, stats: &TrainingStats) {
    let guard = Guard::new();
    let mut loss_data: Vec<(f64, f64)> = Vec::new();

    for (idx, loss_entry) in stats.loss_history.iter(&guard).enumerate() {
        loss_data.push(((idx + 1) as f64, *loss_entry as f64));
    }

    let (min_loss, max_loss) = if loss_data.is_empty() {
        (0.0, 1.0)
    } else {
        let max = loss_data.iter().map(|(_, y)| *y).fold(0.0f64, f64::max);
        let min = loss_data.iter().map(|(_, y)| *y).fold(f64::MAX, f64::min);
        let range = max - min;
        if range < 1e-6 {
            // Single data point or very small range
            (0.0, max * 2.0)
        } else {
            let buffer = range * 0.1;
            ((min - buffer).max(0.0), max + buffer)
        }
    };

    let datasets = vec![Dataset::default()
        .name("Loss")
        .marker(ratatui::symbols::Marker::Braille)
        .graph_type(GraphType::Line)
        .style(Style::default().fg(Color::Red))
        .data(&loss_data)];

    let chart = Chart::new(datasets)
        .block(Block::default().borders(Borders::ALL).title("Loss"))
        .x_axis(Axis::default().bounds([0.0, TOTAL_SUPER_BATCHES as f64]))
        .y_axis(Axis::default().bounds([min_loss, max_loss]).labels(vec![
            Span::raw(format!("{:.3}", min_loss)),
            Span::raw(format!("{:.3}", max_loss)),
        ]));

    frame.render_widget(chart, area);
}

fn render_accuracy_chart(frame: &mut Frame, area: ratatui::layout::Rect, stats: &TrainingStats) {
    let guard = Guard::new();
    let mut acc_data: Vec<(f64, f64)> = Vec::new();

    for (idx, acc_entry) in stats.accuracy_history.iter(&guard).enumerate() {
        acc_data.push(((idx + 1) as f64, (*acc_entry * 100.0) as f64));
    }

    let (min_acc, max_acc) = if acc_data.is_empty() {
        (0.0, 100.0)
    } else {
        let max = acc_data.iter().map(|(_, y)| *y).fold(0.0f64, f64::max);
        let min = acc_data.iter().map(|(_, y)| *y).fold(f64::MAX, f64::min);
        let range = max - min;
        if range < 1e-6 {
            // Single data point - keep full range
            (0.0, 100.0)
        } else {
            let buffer = range * 0.1;
            ((min - buffer).max(0.0), (max + buffer).min(100.0))
        }
    };

    let datasets = vec![Dataset::default()
        .name("Accuracy")
        .marker(ratatui::symbols::Marker::Braille)
        .graph_type(GraphType::Line)
        .style(Style::default().fg(Color::Green))
        .data(&acc_data)];

    let chart = Chart::new(datasets)
        .block(Block::default().borders(Borders::ALL).title("Accuracy"))
        .x_axis(Axis::default().bounds([0.0, TOTAL_SUPER_BATCHES as f64]))
        .y_axis(Axis::default().bounds([min_acc, max_acc]).labels(vec![
            Span::raw(format!("{:.1}", min_acc)),
            Span::raw(format!("{:.1}", max_acc)),
        ]));

    frame.render_widget(chart, area);
}

fn train_super_batch<S: LRScheduler + Sync>(
    network: &mut PolicyNetwork,
    momentum: &mut PolicyNetwork,
    velocity: &mut PolicyNetwork,
    optimizer: &mut AdamWOptimizer<S>,
    config: &TrainingConfig,
    stats: &TrainingStats,
    file: &mut File,
    phase: Phase,
) {
    let mut thread_buffers: Vec<Box<PolicyNetwork>> = (0..config.threads)
        .map(|_| PolicyNetwork::zeroed())
        .collect();
    let mut gradients = PolicyNetwork::zeroed();
    let mut raw_buf = vec![0u8; TrainingPosition::BUFFER_SIZE];

    let mut batches_processed = 0;

    while batches_processed < BATCHES_PER_SUPER_BATCH {
        loop {
            match file.read_exact(&mut raw_buf) {
                Ok(()) => break,
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => {
                    file.seek(SeekFrom::Start(0)).unwrap();
                    stats.file_bytes_consumed.store(0, Ordering::Relaxed);
                }
                Err(e) => panic!("Data read error: {e}"),
            }
        }

        let data = TrainingPosition::read_buffer(&raw_buf);

        for batch in data.chunks(BATCH_SIZE) {
            if batches_processed >= BATCHES_PER_SUPER_BATCH {
                break;
            }

            gradients.zero_out();

            let batch_metrics =
                gradients_batch(network, &mut gradients, batch, &mut thread_buffers, config, phase);

            stats.record_batch(batch_metrics);

            optimizer.step();
            network.adamw(&gradients, momentum, velocity, optimizer);

            // Update current LR
            let current_lr = optimizer.get_learning_rate();
            stats
                .current_lr
                .store(current_lr.to_bits(), Ordering::Relaxed);

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
            .file_bytes_consumed
            .fetch_add(raw_buf.len() as u64, Ordering::Relaxed);
    }
}

fn gradients_batch(
    network: &PolicyNetwork,
    gradients: &mut PolicyNetwork,
    batch: &[TrainingPosition],
    thread_buffers: &mut [Box<PolicyNetwork>],
    config: &TrainingConfig,
    phase: Phase,
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
                        update_gradient(position, network, inner_gradients, inner_metrics, phase);
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
    network: &PolicyNetwork,
    gradients: &mut PolicyNetwork,
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

    let mut position_loss = 0.0f32;
    for idx in 0..moves.len() {
        let actual_val = actual_policy[idx];
        let log_actual_val = actual_val.max(EPSILON).ln();

        let expected_primary_val = expected_primary[idx];
        let expected_secondary_val = expected_secondary[idx];

        position_loss -= expected_primary_val * log_actual_val;
        position_loss -= expected_secondary_val * log_actual_val * SOFT_TARGET_WEIGHT;

        let error = (actual_val - expected_primary_val)
            + (actual_val - expected_secondary_val) * SOFT_TARGET_WEIGHT;

        network.backprop(&features, gradients, move_idxes[idx], error);
    }

    let baseline = (moves.len() as f32).ln();
    metrics.loss += position_loss;
    metrics.baseline_loss += baseline;

    let expected_best = argmax(&expected_primary);
    let predicted_best = argmax(&actual_policy);
    let piece = move_idxes[expected_best].piece();
    metrics.piece_total[piece] += 1;
    metrics.piece_loss[piece] += position_loss;
    metrics.piece_baseline_loss[piece] += baseline;
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

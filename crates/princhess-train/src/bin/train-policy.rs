use std::env;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader};
use std::ops::AddAssign;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use arrayvec::ArrayVec;
use bytemuck::Zeroable;
use crossterm::cursor;
use crossterm::event::{poll, read, Event, KeyCode};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode};
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

use princhess::engine::SCALE;
use princhess::math;
use princhess::state::State;
use princhess_train::data::TrainingPosition;
use princhess_train::neural::{AdamWOptimizer, LRScheduler, SparseVector, StepLRScheduler};
use princhess_train::policy::{Phase, PolicyCount, PolicyNetwork};
use princhess_train::tui;

const BATCHES_PER_SUPER_BATCH: usize = 3_052;
const TOTAL_SUPER_BATCHES: usize = 100;
const BATCH_SIZE: usize = 32768;
const THREADS: usize = 6;

const TUI_TOTAL_HEIGHT: u16 = 35;
const SAMPLE_INTERVAL_SECS: u64 = 5;
const MAX_RATE_SAMPLES: usize = 512;
const MAX_LR_SAMPLES: usize = 512;
const LR_SAMPLES_PER_SUPER_BATCH: usize = 5;

const MG_LR: f32 = 0.001;
const MG_LR_DROP_AT: f32 = 0.35;
const MG_LR_DROP_FACTOR: f32 = 0.5;

const EG_LR: f32 = 0.001;
const EG_LR_DROP_AT: f32 = 0.35;
const EG_LR_DROP_FACTOR: f32 = 0.5;

const MG_SOFT_TARGET_WEIGHT: f32 = 0.1;
const EG_SOFT_TARGET_WEIGHT: f32 = 0.1;
const SOFT_TARGET_TEMPERATURE: f32 = 4.0;

const EPSILON: f32 = 1e-9;
const SAVE_EVERY_N_SUPER_BATCHES: usize = 10;

const _BUFFER_SIZE_CHECK: () = assert!(TrainingPosition::BUFFER_SIZE.is_multiple_of(BATCH_SIZE));

#[derive(Debug, Default, Clone, Copy)]
struct BatchMetrics {
    loss: f32,
    accuracy: f32,
    processed_count: usize,
}

impl AddAssign for BatchMetrics {
    fn add_assign(&mut self, rhs: Self) {
        self.loss += rhs.loss;
        self.accuracy += rhs.accuracy;
        self.processed_count += rhs.processed_count;
    }
}

struct TrainingConfig {
    input_file: String,
    phase: Phase,
    network_info: String,
    data_positions: usize,
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

    // Best metrics (updated at end of super batch, single-threaded, stored as quantized i64)
    best_loss: AtomicI64, // loss * SCALE
    best_loss_sb: AtomicUsize,
    best_accuracy: AtomicI64, // accuracy * SCALE
    best_accuracy_sb: AtomicUsize,

    // Rate tracking
    recent_rates: Queue<u64>,
    last_sample_positions: AtomicU64,

    // Learning rate tracking
    current_lr: AtomicU32, // stored as f32.to_bits() (updated each batch)
    lr_history: Queue<f32>,

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
            best_loss: AtomicI64::new(i64::MAX),
            best_loss_sb: AtomicUsize::new(0),
            best_accuracy: AtomicI64::new(0),
            best_accuracy_sb: AtomicUsize::new(0),
            recent_rates: Queue::default(),
            last_sample_positions: AtomicU64::new(0),
            current_lr: AtomicU32::new(0),
            lr_history: Queue::default(),
            last_saved_net: Mutex::new(None),
        }
    }

    fn record_batch(&self, metrics: BatchMetrics) {
        let loss_scaled = (metrics.loss * SCALE) as i64;
        let acc_scaled = (metrics.accuracy * SCALE) as i64;

        self.current_loss_sum
            .fetch_add(loss_scaled, Ordering::Relaxed);
        self.current_accuracy_sum
            .fetch_add(acc_scaled, Ordering::Relaxed);
        self.current_count
            .fetch_add(metrics.processed_count, Ordering::Relaxed);
        self.total_positions_processed
            .fetch_add(metrics.processed_count as u64, Ordering::Relaxed);
        self.current_batch_in_super.fetch_add(1, Ordering::Relaxed);
    }

    fn finish_super_batch(&self) {
        let loss_sum_scaled = self.current_loss_sum.swap(0, Ordering::Relaxed);
        let acc_sum_scaled = self.current_accuracy_sum.swap(0, Ordering::Relaxed);
        let count = self.current_count.swap(0, Ordering::Relaxed);

        if count > 0 {
            let avg_loss = loss_sum_scaled as f32 / SCALE / count as f32;
            let avg_accuracy = acc_sum_scaled as f32 / SCALE / count as f32;
            let sb_num = self.current_super_batch.load(Ordering::Relaxed);

            self.loss_history.push(avg_loss);
            self.accuracy_history.push(avg_accuracy);

            // Update best metrics
            let avg_loss_scaled = (avg_loss * SCALE) as i64;
            let best_loss_scaled = self.best_loss.load(Ordering::Relaxed);
            if avg_loss_scaled < best_loss_scaled {
                self.best_loss.store(avg_loss_scaled, Ordering::Relaxed);
                self.best_loss_sb.store(sb_num, Ordering::Relaxed);
            }

            let avg_acc_scaled = (avg_accuracy * SCALE) as i64;
            let best_acc_scaled = self.best_accuracy.load(Ordering::Relaxed);
            if avg_acc_scaled > best_acc_scaled {
                self.best_accuracy.store(avg_acc_scaled, Ordering::Relaxed);
                self.best_accuracy_sb.store(sb_num, Ordering::Relaxed);
            }
        }

        self.current_super_batch.fetch_add(1, Ordering::Relaxed);
        self.current_batch_in_super.store(0, Ordering::Relaxed);
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
}

fn main() {
    let mut args = env::args();
    args.next();

    let input = args.next().expect("Missing input file");
    let phase_arg = args
        .next()
        .expect("Missing phase argument (--phase <mg|eg>)");

    let phase = Phase::from_arg(&phase_arg)
        .unwrap_or_else(|| panic!("Invalid phase argument: {phase_arg}. Use 'mg' or 'eg'."));

    let file = File::open(&input).unwrap();
    let data_positions = file.metadata().unwrap().len() as usize / TrainingPosition::SIZE;

    let network = PolicyNetwork::random();
    let momentum = PolicyNetwork::zeroed();
    let velocity = PolicyNetwork::zeroed();

    let config = TrainingConfig {
        input_file: input.clone(),
        phase,
        network_info: format!("{network}"),
        data_positions,
    };

    let total_steps = (TOTAL_SUPER_BATCHES * BATCHES_PER_SUPER_BATCH) as u32;

    match phase {
        Phase::MiddleGame => {
            let scheduler =
                StepLRScheduler::new(MG_LR, MG_LR_DROP_FACTOR, MG_LR_DROP_AT, total_steps);
            let optimizer = AdamWOptimizer::with_scheduler(scheduler).weight_decay(0.01);
            run_training_loop(
                phase, network, momentum, velocity, optimizer, &input, config,
            );
        }
        Phase::Endgame => {
            let scheduler =
                StepLRScheduler::new(EG_LR, EG_LR_DROP_FACTOR, EG_LR_DROP_AT, total_steps);
            let optimizer = AdamWOptimizer::with_scheduler(scheduler).weight_decay(0.0);
            run_training_loop(
                phase, network, momentum, velocity, optimizer, &input, config,
            );
        }
    }
}

fn run_training_loop<S: LRScheduler>(
    phase: Phase,
    mut network: Box<PolicyNetwork>,
    mut momentum: Box<PolicyNetwork>,
    mut velocity: Box<PolicyNetwork>,
    mut optimizer: AdamWOptimizer<S>,
    input: &str,
    config: TrainingConfig,
) {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let start_time = Instant::now();
    let stats = Arc::new(TrainingStats::new(start_time));
    let stop_signal = Arc::new(AtomicBool::new(false));

    // Spawn TUI update thread
    let stats_clone = Arc::clone(&stats);
    let stop_clone = Arc::clone(&stop_signal);
    let tui_thread = thread::spawn(move || {
        if let Err(e) = run_tui(&stats_clone, stop_clone.clone(), config) {
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
            input,
            phase,
            &stats,
        );

        stats.finish_super_batch();

        // Save network periodically
        if (sb + 1) % SAVE_EVERY_N_SUPER_BATCHES == 0 || sb + 1 == TOTAL_SUPER_BATCHES {
            let dir_name = format!("nets/policy-{phase}-{timestamp}-sb{:03}", sb + 1);
            fs::create_dir(&dir_name).expect("Failed to create network save directory");
            let dir = Path::new(&dir_name);
            network
                .to_boxed_and_quantized()
                .save_to_bin(dir, format!("{phase}-policy.bin").as_str());

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
    enable_raw_mode()?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::with_options(
        backend,
        TerminalOptions {
            viewport: Viewport::Inline(TUI_TOTAL_HEIGHT),
        },
    )?;

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
        Ok(())
    })();

    // Position cursor at the end of the viewport
    let viewport_area = terminal.get_frame().area();
    io::stdout().execute(cursor::MoveTo(0, viewport_area.bottom()))?;
    disable_raw_mode()?;
    io::stdout().execute(cursor::Show)?;
    result
}

fn render_tui(frame: &mut Frame, stats: &TrainingStats, config: &TrainingConfig) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .margin(1)
        .constraints([
            Constraint::Length(7),  // Progress section (with 2 sparklines)
            Constraint::Length(5),  // Info boxes
            Constraint::Length(10), // Loss chart
            Constraint::Length(10), // Accuracy chart
            Constraint::Length(1),  // Network info footer
        ])
        .split(frame.area());

    render_progress(frame, chunks[0], stats);
    render_info(frame, chunks[1], config, stats);
    render_loss_chart(frame, chunks[2], stats);
    render_accuracy_chart(frame, chunks[3], stats);

    // Footer: Network info (left) and Last saved (right)
    let footer_layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(chunks[4]);

    let network_info =
        Paragraph::new(format!("Network: {}", config.network_info)).alignment(Alignment::Left);
    frame.render_widget(network_info, footer_layout[0]);

    let last_saved = stats
        .last_saved_net
        .lock()
        .unwrap()
        .clone()
        .unwrap_or_else(|| "None".to_string());
    let saved_info =
        Paragraph::new(format!("Last saved: {}", last_saved)).alignment(Alignment::Right);
    frame.render_widget(saved_info, footer_layout[1]);
}

fn render_progress(frame: &mut Frame, area: ratatui::layout::Rect, stats: &TrainingStats) {
    let block = Block::default().borders(Borders::ALL).title("Progress");
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // Time/rate/ETA info
            Constraint::Length(1), // Overall progress bar
            Constraint::Length(1), // Current super batch progress bar
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

    let eta_secs = if batches_per_sec > 0.0 {
        ((total_batches - total_batches_done) as f64 / batches_per_sec) as u64
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
            "{:>4} / {:>4} ({:>5.1}%)",
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
            "{:>4} / {:>4} ({:>5.1}%)",
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

    // Sparkline for processing rate
    let guard = Guard::new();
    let recent_rates: Vec<u64> = stats.recent_rates.iter(&guard).copied().collect();
    if !recent_rates.is_empty() {
        let max_bars = layout[3].width as usize;
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
        frame.render_widget(sparkline, layout[3]);
    }

    // Sparkline for learning rate
    let lr_samples: Vec<f32> = stats.lr_history.iter(&guard).copied().collect();
    if !lr_samples.is_empty() {
        let sparkline_width = layout[4].width as usize;
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
            .style(Style::default().fg(Color::Yellow));
        frame.render_widget(lr_sparkline, layout[4]);
    }
}

fn render_info(
    frame: &mut Frame,
    area: ratatui::layout::Rect,
    config: &TrainingConfig,
    stats: &TrainingStats,
) {
    let columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area);

    // Left box: Dataset
    let left_block = Block::default().borders(Borders::ALL).title("Dataset");
    frame.render_widget(left_block, columns[0]);

    let left_inner = columns[0].inner(ratatui::layout::Margin {
        horizontal: 1,
        vertical: 1,
    });

    let left_content = format!(
        "Phase:     {}\nInput:     {}\nPositions: {:.1}M",
        config.phase,
        config.input_file,
        config.data_positions as f64 / 1_000_000.0,
    );
    let left_para = Paragraph::new(left_content);
    frame.render_widget(left_para, left_inner);

    // Right box: Metrics
    let right_block = Block::default().borders(Borders::ALL).title("Metrics");
    frame.render_widget(right_block, columns[1]);

    let right_inner = columns[1].inner(ratatui::layout::Margin {
        horizontal: 1,
        vertical: 1,
    });

    let (current_loss, current_acc) = stats.get_current_avg_metrics();
    let current_lr = f32::from_bits(stats.current_lr.load(Ordering::Relaxed));

    let right_content = format!(
        "Loss:     {:.4}\nAccuracy: {:.2}%\nLR:       {:.6}",
        current_loss,
        current_acc * 100.0,
        current_lr
    );
    let right_para = Paragraph::new(right_content);
    frame.render_widget(right_para, right_inner);
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

fn train_super_batch<S: LRScheduler>(
    network: &mut PolicyNetwork,
    momentum: &mut PolicyNetwork,
    velocity: &mut PolicyNetwork,
    optimizer: &mut AdamWOptimizer<S>,
    input: &str,
    phase: Phase,
    stats: &TrainingStats,
) {
    let file = File::open(input).unwrap();
    let mut buffer = BufReader::with_capacity(TrainingPosition::BUFFER_SIZE, file);

    let mut batches_processed = 0;

    while batches_processed < BATCHES_PER_SUPER_BATCH {
        let Ok(bytes) = buffer.fill_buf() else {
            break;
        };

        if bytes.is_empty() {
            // Reached end of file, restart from beginning
            drop(buffer);
            let file = File::open(input).unwrap();
            buffer = BufReader::with_capacity(TrainingPosition::BUFFER_SIZE, file);
            continue;
        }

        let data = TrainingPosition::read_buffer(bytes);

        for batch in data.chunks(BATCH_SIZE) {
            if batches_processed >= BATCHES_PER_SUPER_BATCH {
                break;
            }

            let mut gradients = PolicyNetwork::zeroed();
            let mut count = PolicyCount::zeroed();

            let batch_metrics = gradients_batch(network, &mut gradients, &mut count, batch, phase);

            gradients.scale_by_counts(&count);

            optimizer.step();
            network.adamw(&gradients, momentum, velocity, optimizer);

            stats.record_batch(batch_metrics);

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

        let consumed = bytes.len();
        buffer.consume(consumed);
    }
}

fn gradients_batch(
    network: &PolicyNetwork,
    gradients: &mut PolicyNetwork,
    count: &mut PolicyCount,
    batch: &[TrainingPosition],
    phase: Phase,
) -> BatchMetrics {
    let size = (batch.len() / THREADS) + 1;
    let mut total_metrics = BatchMetrics::default();

    thread::scope(|s| {
        batch
            .chunks(size)
            .map(|chunk| {
                s.spawn(move || {
                    let mut inner_gradients = PolicyNetwork::zeroed();
                    let mut inner_count = PolicyCount::zeroed();
                    let mut inner_metrics = BatchMetrics::default();

                    for position in chunk {
                        update_gradient(
                            position,
                            network,
                            &mut inner_gradients,
                            &mut inner_count,
                            &mut inner_metrics,
                            phase,
                        );
                    }
                    (inner_gradients, inner_count, inner_metrics)
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .for_each(|(inner_gradients, inner_count, inner_metrics)| {
                *gradients += &inner_gradients;
                *count += &inner_count;
                total_metrics += inner_metrics;
            });
    });

    total_metrics
}

fn update_gradient(
    position: &TrainingPosition,
    network: &PolicyNetwork,
    gradients: &mut PolicyNetwork,
    count: &mut PolicyCount,
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

    for idx in 0..moves.len() {
        let move_idx = move_idxes[idx];
        let actual_val = actual_policy[idx];
        let log_actual_val = actual_val.max(EPSILON).ln();

        let expected_primary_val = expected_primary[idx];
        metrics.loss -= expected_primary_val * log_actual_val;

        let soft_target_weight = match phase {
            Phase::MiddleGame => MG_SOFT_TARGET_WEIGHT,
            Phase::Endgame => EG_SOFT_TARGET_WEIGHT,
        };

        let expected_secondary_val = expected_secondary[idx];
        metrics.loss -= expected_secondary_val * log_actual_val * soft_target_weight;

        let combined_error = (actual_val - expected_primary_val)
            + (actual_val - expected_secondary_val) * soft_target_weight;
        network.backprop(&features, gradients, move_idx, combined_error);
        count.increment(move_idx);
    }

    if argmax(&expected_primary) == argmax(&actual_policy) {
        metrics.accuracy += 1.;
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

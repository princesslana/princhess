use std::fs::{self, File};
use std::io::{self, Read, Seek, SeekFrom};
use std::ops::AddAssign;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

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

use princhess::engine::SCALE;
use princhess::state::State;
use princhess_train::args::Args;
use princhess_train::data::TrainingPosition;
use princhess_train::neural::{
    AdamWOptimizer, FeedForwardNetwork, LRScheduler, LinearWarmupDecayLRScheduler, OutputLayer,
    SparseVector, Vector,
};
use princhess_train::system;
use princhess_train::tui::{self, RawModeGuard};
use princhess_train::value::ValueNetwork;

const BATCHES_PER_SUPER_BATCH: usize = 6_104;
const TOTAL_SUPER_BATCHES: usize = 75;
const BATCH_SIZE: usize = 16384;

const TUI_TOTAL_HEIGHT: u16 = 27;
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

    // File read progress (bytes consumed from current pass through the file)
    file_bytes_consumed: AtomicU64,

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
            file_bytes_consumed: AtomicU64::new(0),
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

    let scheduler = LinearWarmupDecayLRScheduler::new(LEARNING_RATE, 0.05, total_steps);
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
        );

        stats.finish_super_batch();

        // Save network periodically
        if (sb + 1) % SAVE_EVERY_N_SUPER_BATCHES == 0 || sb + 1 == TOTAL_SUPER_BATCHES {
            let dir_name = format!("nets/value-{timestamp}-sb{:03}", sb + 1);
            fs::create_dir(&dir_name).expect("Failed to create network save directory");
            let dir = Path::new(&dir_name);
            network.to_boxed_and_quantized().save_to_bin(dir);

            *stats.last_saved_net.lock().unwrap() = Some(dir_name);
        }
    }

    // Give TUI time to render final state before stopping
    thread::sleep(Duration::from_millis(200));

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
            Constraint::Length(8),  // Progress section (with 2 sparklines + file bar)
            Constraint::Length(4),  // Dataset / Network info boxes
            Constraint::Length(4),  // Info boxes
            Constraint::Length(10), // Loss chart
        ])
        .split(frame.area());

    render_progress(frame, chunks[0], stats, config);
    render_dataset_boxes(frame, chunks[1], stats, config);
    render_info(frame, chunks[2], stats);
    render_loss_chart(frame, chunks[3], stats);
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
            .style(Style::default().fg(Color::Yellow));
        frame.render_widget(lr_sparkline, layout[5]);
    }
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
            "Input: {}\nPositions: {:.1}M",
            config.input_file,
            config.data_positions as f64 / 1_000_000.0,
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

fn train_super_batch<S: LRScheduler>(
    network: &mut ValueNetwork,
    momentum: &mut ValueNetwork,
    velocity: &mut ValueNetwork,
    optimizer: &mut AdamWOptimizer<S>,
    config: &TrainingConfig,
    stats: &TrainingStats,
    file: &mut File,
) {
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

        stats
            .file_bytes_consumed
            .fetch_add(raw_buf.len() as u64, Ordering::Relaxed);

        let data = TrainingPosition::read_buffer(&raw_buf);

        for batch in data.chunks(BATCH_SIZE) {
            if batches_processed >= BATCHES_PER_SUPER_BATCH {
                break;
            }

            let mut gradients = ValueNetwork::zeroed();

            let batch_metrics = gradients_batch(network, &mut gradients, batch, config.threads);

            *gradients /= batch.len() as f32;

            optimizer.step();

            network.train_step(&gradients, momentum, velocity, optimizer, optimizer);

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

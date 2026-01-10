use std::env;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use bytemuck::allocation;
use chrono::Utc;
use crossterm::cursor;
use crossterm::event::{poll, read, Event, KeyCode};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode};
use crossterm::ExecutableCommand;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Alignment, Constraint, Direction, Layout};
use ratatui::style::{Color, Style};
use ratatui::text::Span;
use ratatui::widgets::{Block, Borders, Gauge, Paragraph};
use ratatui::{Frame, Terminal, TerminalOptions, Viewport};

use princhess::math::Rng;
use princhess_train::data::TrainingPosition;
use princhess_train::tui;

#[derive(Clone)]
struct FileInfo {
    path: PathBuf,
    is_shuffled: bool,
    position_count: usize,
}

impl FileInfo {
    fn new(path: PathBuf) -> io::Result<Self> {
        let is_shuffled = path
            .to_str()
            .map(|s| s.contains(".shuffled"))
            .unwrap_or(false);

        let metadata = fs::metadata(&path)?;
        let position_count = metadata.len() as usize / TrainingPosition::SIZE;

        Ok(Self {
            path,
            is_shuffled,
            position_count,
        })
    }
}

struct ProgressState {
    shuffle_start: Instant,
    shuffle_end_secs: AtomicU64, // Elapsed seconds when shuffle completed
    shuffle_complete: AtomicBool,
    shuffle_files_done: AtomicUsize,
    shuffle_files_total: AtomicUsize,
    shuffle_current_file: AtomicUsize,
    shuffle_load_progress: AtomicUsize,    // 0-100
    shuffle_shuffle_progress: AtomicUsize, // 0-100
    shuffle_write_progress: AtomicUsize,   // 0-100

    interleave_start: AtomicU64, // Elapsed secs from shuffle_start when interleave began
    interleave_complete: AtomicBool,
    interleave_positions_done: AtomicU64,
    interleave_positions_total: AtomicU64,

    stop_signal: AtomicBool,
}

impl ProgressState {
    fn new(shuffle_file_count: usize, interleave_total: u64) -> Self {
        Self {
            shuffle_start: Instant::now(),
            shuffle_end_secs: AtomicU64::new(0),
            shuffle_complete: AtomicBool::new(false),
            shuffle_files_done: AtomicUsize::new(0),
            shuffle_files_total: AtomicUsize::new(shuffle_file_count),
            shuffle_current_file: AtomicUsize::new(0),
            shuffle_load_progress: AtomicUsize::new(0),
            shuffle_shuffle_progress: AtomicUsize::new(0),
            shuffle_write_progress: AtomicUsize::new(0),

            interleave_start: AtomicU64::new(0),
            interleave_complete: AtomicBool::new(false),
            interleave_positions_done: AtomicU64::new(0),
            interleave_positions_total: AtomicU64::new(interleave_total),

            stop_signal: AtomicBool::new(false),
        }
    }

    // Atomic load helpers
    fn shuffle_files_done(&self) -> usize {
        self.shuffle_files_done.load(Ordering::Relaxed)
    }

    fn shuffle_files_total(&self) -> usize {
        self.shuffle_files_total.load(Ordering::Relaxed)
    }

    fn shuffle_current_file(&self) -> usize {
        self.shuffle_current_file.load(Ordering::Relaxed)
    }

    fn shuffle_phase_progress(&self) -> (usize, usize, usize) {
        (
            self.shuffle_load_progress.load(Ordering::Relaxed),
            self.shuffle_shuffle_progress.load(Ordering::Relaxed),
            self.shuffle_write_progress.load(Ordering::Relaxed),
        )
    }

    fn shuffle_complete(&self) -> bool {
        self.shuffle_complete.load(Ordering::Relaxed)
    }

    fn interleave_positions_done(&self) -> u64 {
        self.interleave_positions_done.load(Ordering::Relaxed)
    }

    fn interleave_positions_total(&self) -> u64 {
        self.interleave_positions_total.load(Ordering::Relaxed)
    }

    fn interleave_complete(&self) -> bool {
        self.interleave_complete.load(Ordering::Relaxed)
    }

    // Atomic store helpers
    fn reset_shuffle_phase_progress(&self) {
        self.shuffle_load_progress.store(0, Ordering::Relaxed);
        self.shuffle_shuffle_progress.store(0, Ordering::Relaxed);
        self.shuffle_write_progress.store(0, Ordering::Relaxed);
    }

    fn mark_shuffle_complete(&self) {
        let elapsed = self.shuffle_start.elapsed().as_secs();
        self.shuffle_end_secs.store(elapsed, Ordering::Relaxed);
        self.interleave_start.store(elapsed, Ordering::Relaxed);
        self.shuffle_complete.store(true, Ordering::Relaxed);
    }

    // Calculated properties
    fn shuffle_elapsed_secs(&self) -> u64 {
        if self.shuffle_complete() {
            self.shuffle_end_secs.load(Ordering::Relaxed)
        } else {
            self.shuffle_start.elapsed().as_secs()
        }
    }

    fn shuffle_progress_ratio(&self) -> f64 {
        let total = self.shuffle_files_total();
        if total > 0 {
            (self.shuffle_files_done() as f64 / total as f64).min(1.0)
        } else {
            0.0
        }
    }

    fn interleave_elapsed_secs(&self) -> u64 {
        let start = self.interleave_start.load(Ordering::Relaxed);
        if start > 0 {
            let total_elapsed = self.shuffle_start.elapsed().as_secs();
            total_elapsed.saturating_sub(start).max(1)
        } else {
            0
        }
    }

    fn interleave_rate(&self) -> f64 {
        let elapsed = self.interleave_elapsed_secs();
        if elapsed > 0 {
            self.interleave_positions_done() as f64 / elapsed as f64 / 1_000_000.0
        } else {
            0.0
        }
    }

    fn interleave_eta_secs(&self) -> u64 {
        if !self.shuffle_complete() || self.interleave_complete() {
            return 0;
        }

        let total = self.interleave_positions_total();
        let rate = self.interleave_rate();

        if total > 0 && rate > 0.0 {
            let remaining = total.saturating_sub(self.interleave_positions_done());
            (remaining as f64 / (rate * 1_000_000.0)) as u64
        } else {
            0
        }
    }

    fn interleave_progress_ratio(&self) -> f64 {
        let total = self.interleave_positions_total();
        if total > 0 {
            (self.interleave_positions_done() as f64 / total as f64).min(1.0)
        } else {
            0.0
        }
    }
}

fn shuffle_file(
    input_path: &Path,
    output_path: &Path,
    rng: &mut Rng,
    progress: &ProgressState,
) -> io::Result<()> {
    // Phase 1: Load
    let file = File::open(input_path)?;
    let file_size = file.metadata()?.len() as usize;
    let total_positions = file_size / TrainingPosition::SIZE;
    let mut positions = Vec::with_capacity(total_positions);
    let mut buffer = BufReader::with_capacity(TrainingPosition::BUFFER_SIZE, file);

    let mut loaded = 0;
    while let Ok(bytes) = buffer.fill_buf() {
        if bytes.is_empty() {
            break;
        }

        let data = TrainingPosition::read_buffer(bytes);
        loaded += data.len();
        positions.extend_from_slice(data);

        let pct = ((loaded as f64 / total_positions as f64) * 100.0) as usize;
        progress.shuffle_load_progress.store(pct, Ordering::Relaxed);

        let consumed = bytes.len();
        buffer.consume(consumed);
    }

    progress.shuffle_load_progress.store(100, Ordering::Relaxed);

    // Phase 2: Shuffle
    for i in 0..positions.len() - 1 {
        let j = rng.next_usize() % (positions.len() - i);
        positions.swap(i, i + j);

        if i % 100000 == 0 {
            let pct = ((i as f64 / positions.len() as f64) * 100.0) as usize;
            progress
                .shuffle_shuffle_progress
                .store(pct, Ordering::Relaxed);
        }
    }

    progress
        .shuffle_shuffle_progress
        .store(100, Ordering::Relaxed);

    // Phase 3: Write
    let mut writer = BufWriter::new(File::create(output_path)?);
    let mut write_buffer: Box<[TrainingPosition; TrainingPosition::BUFFER_COUNT]> =
        allocation::zeroed_box();

    let mut written = 0;
    while !positions.is_empty() {
        let chunk_size = positions.len().min(TrainingPosition::BUFFER_COUNT);
        write_buffer[..chunk_size].copy_from_slice(&positions[..chunk_size]);
        positions.drain(..chunk_size);
        TrainingPosition::write_buffer(&mut writer, &write_buffer[..chunk_size]);

        written += chunk_size;
        let pct = ((written as f64 / total_positions as f64) * 100.0) as usize;
        progress
            .shuffle_write_progress
            .store(pct, Ordering::Relaxed);
    }

    progress
        .shuffle_write_progress
        .store(100, Ordering::Relaxed);
    progress.shuffle_files_done.fetch_add(1, Ordering::Relaxed);

    Ok(())
}

fn interleave_files(
    input_paths: &[PathBuf],
    output_path: &Path,
    rng: &mut Rng,
    progress: &ProgressState,
) -> io::Result<()> {
    let mut inputs = Vec::new();
    let mut total = 0;

    for path in input_paths {
        let file = File::open(path)?;
        let count = file.metadata()?.len() as usize / TrainingPosition::SIZE;
        inputs.push((count, BufReader::new(file)));
        total += count;
    }

    let mut remaining = total;
    let mut written = 0;
    let mut writer = BufWriter::new(File::create(output_path)?);

    while remaining > 0 {
        let mut choice = rng.next_usize() % remaining;
        let mut idx = 0;
        while inputs[idx].0 <= choice {
            choice -= inputs[idx].0;
            idx += 1;
        }

        let (count, reader) = &mut inputs[idx];
        let mut value = [0; TrainingPosition::SIZE];
        reader.read_exact(&mut value)?;
        writer.write_all(&value)?;

        written += 1;
        remaining -= 1;
        *count -= 1;

        if written % 10000 == 0 {
            progress
                .interleave_positions_done
                .store(written as u64, Ordering::Relaxed);
        }

        if *count == 0 {
            inputs.remove(idx);
        }
    }

    progress
        .interleave_positions_done
        .store(total as u64, Ordering::Relaxed);

    Ok(())
}
fn render_tui(frame: &mut Frame, files: &[FileInfo], progress: &ProgressState) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .margin(1)
        .constraints([Constraint::Length(7), Constraint::Length(4)])
        .split(frame.area());

    render_shuffle_box(frame, chunks[0], files, progress);
    render_interleave_box(frame, chunks[1], progress);
}

fn render_shuffle_box(
    frame: &mut Frame,
    area: ratatui::layout::Rect,
    files: &[FileInfo],
    progress: &ProgressState,
) {
    let block = Block::default().borders(Borders::ALL).title("Shuffling");
    frame.render_widget(block, area);

    let inner = area.inner(ratatui::layout::Margin {
        horizontal: 1,
        vertical: 1,
    });

    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // Time/rate/ETA
            Constraint::Length(1), // Overall
            Constraint::Length(1), // Load
            Constraint::Length(1), // Shuffle
            Constraint::Length(1), // Write
        ])
        .split(inner);

    let shuffle_files_done = progress.shuffle_files_done();
    let shuffle_files_total = progress.shuffle_files_total();

    // Elapsed time
    frame.render_widget(
        Paragraph::new(tui::format_elapsed(progress.shuffle_elapsed_secs()))
            .alignment(Alignment::Left),
        layout[0],
    );

    // Overall progress
    let current_file_idx = progress.shuffle_current_file();
    let current_filename = files
        .get(current_file_idx)
        .and_then(|f| f.path.file_name())
        .and_then(|n| n.to_str())
        .unwrap_or("");

    let label = Span::styled(
        format!(
            "{} [{}/{}]",
            current_filename, shuffle_files_done, shuffle_files_total
        ),
        Style::default().fg(Color::White),
    );

    frame.render_widget(
        Gauge::default()
            .gauge_style(Style::default().fg(Color::Green))
            .ratio(progress.shuffle_progress_ratio())
            .label(label),
        layout[1],
    );

    // Load/Shuffle/Write bars
    let (load_pct, shuffle_pct, write_pct) = progress.shuffle_phase_progress();

    frame.render_widget(
        Gauge::default()
            .gauge_style(Style::default().fg(Color::Cyan))
            .ratio(load_pct as f64 / 100.0)
            .label(Span::styled(
                format!("Load    ({:>3}%)", load_pct),
                Style::default().fg(Color::White),
            )),
        layout[2],
    );

    frame.render_widget(
        Gauge::default()
            .gauge_style(Style::default().fg(Color::Yellow))
            .ratio(shuffle_pct as f64 / 100.0)
            .label(Span::styled(
                format!("Shuffle ({:>3}%)", shuffle_pct),
                Style::default().fg(Color::White),
            )),
        layout[3],
    );

    frame.render_widget(
        Gauge::default()
            .gauge_style(Style::default().fg(Color::Magenta))
            .ratio(write_pct as f64 / 100.0)
            .label(Span::styled(
                format!("Write   ({:>3}%)", write_pct),
                Style::default().fg(Color::White),
            )),
        layout[4],
    );
}

fn render_interleave_box(frame: &mut Frame, area: ratatui::layout::Rect, progress: &ProgressState) {
    let block = Block::default().borders(Borders::ALL).title("Interleaving");
    frame.render_widget(block, area);

    let inner = area.inner(ratatui::layout::Margin {
        horizontal: 1,
        vertical: 1,
    });

    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Length(1)])
        .split(inner);

    let interleave_done = progress.interleave_positions_done();
    let interleave_total = progress.interleave_positions_total();

    // Time/ETA
    let time_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(layout[0]);

    frame.render_widget(
        Paragraph::new(tui::format_elapsed(progress.interleave_elapsed_secs()))
            .alignment(Alignment::Left),
        time_chunks[0],
    );
    frame.render_widget(
        Paragraph::new(tui::format_eta(progress.interleave_eta_secs())).alignment(Alignment::Right),
        time_chunks[1],
    );

    // Overall progress

    let label = Span::styled(
        format!(
            "{}M / {}M ({:.0}%)",
            interleave_done / 1_000_000,
            interleave_total / 1_000_000,
            progress.interleave_progress_ratio() * 100.0
        ),
        Style::default().fg(Color::White),
    );

    frame.render_widget(
        Gauge::default()
            .gauge_style(Style::default().fg(Color::Green))
            .ratio(progress.interleave_progress_ratio())
            .label(label),
        layout[1],
    );
}

fn main() {
    let mut args = env::args();
    args.next();

    let mut cleanup = false;
    let mut force = false;
    let mut input_files = Vec::new();

    // Parse arguments
    for arg in args {
        match arg.as_str() {
            "--cleanup" => cleanup = true,
            "--force" => force = true,
            _ => input_files.push(PathBuf::from(arg)),
        }
    }

    if input_files.is_empty() {
        eprintln!("Usage: data-shuffle <files...> [--cleanup] [--force]");
        eprintln!();
        eprintln!("Options:");
        eprintln!("  --cleanup  Delete input files after successful shuffle");
        eprintln!("  --force    Required with --cleanup if any file >= 200M positions");
        std::process::exit(1);
    }

    // Gather file info
    let files: Vec<FileInfo> = input_files
        .into_iter()
        .map(|path| {
            FileInfo::new(path.clone()).unwrap_or_else(|e| {
                eprintln!("Error reading {}: {}", path.display(), e);
                std::process::exit(1);
            })
        })
        .collect();

    const SHUFFLE_LIMIT: usize = 25_000_000; // 25M positions
    const CLEANUP_SAFETY_THRESHOLD: usize = 150_000_000; // 150M positions

    // Safety check: files must have valid position counts
    for file in &files {
        if file.position_count == 0 {
            eprintln!("ERROR: Cannot shuffle empty file {}", file.path.display());
            std::process::exit(1);
        }
        if file.position_count % TrainingPosition::BUFFER_COUNT != 0 {
            eprintln!(
                "ERROR: File {} has invalid position count {}",
                file.path.display(),
                file.position_count
            );
            eprintln!(
                "       Position count must be a multiple of {}",
                TrainingPosition::BUFFER_COUNT
            );
            std::process::exit(1);
        }
    }

    // Safety check: files too large to shuffle
    for file in &files {
        if !file.is_shuffled && file.position_count > SHUFFLE_LIMIT {
            eprintln!(
                "ERROR: Cannot shuffle {} ({} positions)",
                file.path.display(),
                file.position_count
            );
            eprintln!("       Shuffle limit is {} positions", SHUFFLE_LIMIT);
            std::process::exit(1);
        }
    }

    // Safety check: cleanup with large files requires --force
    if cleanup && !force {
        if let Some(large_file) = files
            .iter()
            .find(|f| f.position_count >= CLEANUP_SAFETY_THRESHOLD)
        {
            eprintln!(
                "ERROR: Refusing to delete {} ({} positions)",
                large_file.path.display(),
                large_file.position_count
            );
            eprintln!(
                "       Files >= {}M positions require --force",
                CLEANUP_SAFETY_THRESHOLD / 1_000_000
            );
            std::process::exit(1);
        }
    }

    // Safety check: output directory exists
    if !Path::new("data").is_dir() {
        eprintln!("ERROR: 'data' directory does not exist");
        std::process::exit(1);
    }

    // Calculate totals for progress tracking
    let shuffle_file_count = files.iter().filter(|f| !f.is_shuffled).count();
    let total_positions: u64 = files.iter().map(|f| f.position_count as u64).sum();

    let progress = Arc::new(ProgressState::new(shuffle_file_count, total_positions));

    // Spawn TUI thread
    thread::scope(|s| {
        let tui_progress = progress.clone();
        let tui_files = files.clone();
        s.spawn(move || {
            if let Err(e) = run_tui(&tui_files, &tui_progress) {
                eprintln!("TUI failed: {}", e);
                tui_progress.stop_signal.store(true, Ordering::Relaxed);
            }
        });

        // Main worker thread
        let mut rng = Rng::default();
        let mut shuffled_paths = Vec::new();
        let mut temp_files = Vec::new();

        // Phase 1: Shuffle unshuffled files
        for (idx, file) in files.iter().enumerate() {
            if progress.stop_signal.load(Ordering::Relaxed) {
                return;
            }

            if file.is_shuffled {
                shuffled_paths.push(file.path.clone());
            } else {
                progress.shuffle_current_file.store(idx, Ordering::Relaxed);
                progress.reset_shuffle_phase_progress();

                let mut output_path = file.path.clone();
                let current_ext = output_path
                    .extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("data");
                output_path.set_extension(format!("{}.shuffled", current_ext));

                if let Err(e) = shuffle_file(&file.path, &output_path, &mut rng, &progress) {
                    eprintln!("Error shuffling {}: {}", file.path.display(), e);
                    progress.stop_signal.store(true, Ordering::Relaxed);
                    return;
                }

                shuffled_paths.push(output_path.clone());
                temp_files.push(output_path.clone());

                // Cleanup original file immediately after shuffling if requested
                if cleanup {
                    fs::remove_file(&file.path).ok();
                }
            }
        }

        progress.mark_shuffle_complete();

        // Phase 2: Interleave all files
        if !progress.stop_signal.load(Ordering::Relaxed) {
            let timestamp = Utc::now().format("%Y%m%d-%H%M").to_string();
            let output_filename = format!(
                "data/princhess-{}-{}m.data.shuffled",
                timestamp,
                total_positions / 1_000_000
            );
            let output_path = PathBuf::from(&output_filename);

            if let Err(e) = interleave_files(&shuffled_paths, &output_path, &mut rng, &progress) {
                eprintln!("Error interleaving files: {}", e);
                progress.stop_signal.store(true, Ordering::Relaxed);
                return;
            }

            progress.interleave_complete.store(true, Ordering::Relaxed);

            // Cleanup temp shuffled files after interleaving if requested
            if cleanup {
                for temp in &temp_files {
                    fs::remove_file(temp).ok();
                }
            }
        }
    });
}

fn run_tui(files: &[FileInfo], progress: &ProgressState) -> io::Result<()> {
    let stdout = io::stdout();
    enable_raw_mode()?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::with_options(
        backend,
        TerminalOptions {
            viewport: Viewport::Inline(13),
        },
    )?;

    let result = (|| -> io::Result<()> {
        loop {
            terminal.draw(|f| render_tui(f, files, progress))?;

            if progress.interleave_complete.load(Ordering::Relaxed)
                || progress.stop_signal.load(Ordering::Relaxed)
            {
                break;
            }

            if poll(Duration::from_millis(100))? {
                if let Event::Key(key) = read()? {
                    if key.code == KeyCode::Char('c')
                        && key
                            .modifiers
                            .contains(crossterm::event::KeyModifiers::CONTROL)
                    {
                        progress.stop_signal.store(true, Ordering::Relaxed);
                        break;
                    }
                }
            }
        }
        Ok(())
    })();

    let viewport_area = terminal.get_frame().area();
    io::stdout().execute(cursor::MoveTo(0, viewport_area.bottom()))?;
    disable_raw_mode()?;

    result
}

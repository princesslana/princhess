use std::io;
use std::time::Duration;

use crossterm::event::{self, Event, KeyCode, KeyModifiers};
use crossterm::{cursor, terminal, ExecutableCommand};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Style};
use ratatui::symbols::Marker;
use ratatui::text::Span;
use ratatui::widgets::{
    Axis, Block, Borders, Chart, Dataset, Gauge, GraphType, Paragraph, Sparkline,
};
use ratatui::{Frame, Terminal, TerminalOptions, Viewport};

struct RawModeGuard;

impl RawModeGuard {
    fn enable() -> io::Result<Self> {
        terminal::enable_raw_mode()?;
        Ok(Self)
    }
}

impl Drop for RawModeGuard {
    fn drop(&mut self) {
        terminal::disable_raw_mode().ok();
    }
}

/// # Errors
/// Returns an error if the terminal cannot be initialized or if reading input events fails.
pub fn run_inline_tui(
    height: u16,
    stop: impl Fn() -> bool,
    on_ctrl_c: impl Fn(),
    mut on_tick: impl FnMut(),
    mut render: impl FnMut(&mut Frame),
) -> io::Result<()> {
    let mut terminal = Terminal::with_options(
        CrosstermBackend::new(io::stdout()),
        TerminalOptions {
            viewport: Viewport::Inline(height),
        },
    )?;

    let _guard = RawModeGuard::enable()?;

    let result = (|| -> io::Result<()> {
        loop {
            terminal.draw(|f| render(f))?;
            on_tick();
            if stop() {
                break;
            }
            if event::poll(Duration::from_millis(100))? {
                if let Event::Key(key) = event::read()? {
                    if key.code == KeyCode::Char('c')
                        && key.modifiers.contains(KeyModifiers::CONTROL)
                    {
                        on_ctrl_c();
                        break;
                    }
                }
            }
        }
        Ok(())
    })();

    let viewport_area = terminal.get_frame().area();
    io::stdout().execute(cursor::MoveTo(0, viewport_area.bottom()))?;
    io::stdout().execute(cursor::Show)?;
    result
}

#[must_use]
pub fn format_elapsed(seconds: u64) -> String {
    let hours = seconds / 3600;
    let minutes = (seconds % 3600) / 60;
    let secs = seconds % 60;
    format!("{hours:2}h {minutes:02}m {secs:02}s")
}

#[must_use]
pub fn format_eta(seconds: u64) -> String {
    let hours = seconds / 3600;
    let minutes = (seconds % 3600) / 60;
    format!("ETA: {hours:2}h {minutes:02}m")
}

pub struct TrainingProgressView {
    pub elapsed_secs: u64,
    pub samples_per_sec: f64,
    pub eta_secs: u64,
    pub super_batch: usize,
    pub total_super_batches: usize,
    pub batch_in_super: usize,
    pub batches_per_super_batch: usize,
    pub positions_consumed: u64,
    pub data_positions: u64,
    pub recent_rates: Vec<u64>,
    pub lr_history: Vec<f32>,
    pub lr_samples_per_super_batch: usize,
}

#[allow(clippy::too_many_lines)]
pub fn render_training_progress(frame: &mut Frame, area: Rect, view: &TrainingProgressView) {
    let block = Block::default().borders(Borders::ALL).title("Progress");
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // Time / rate / ETA
            Constraint::Length(1), // Overall super-batch gauge
            Constraint::Length(1), // Current super-batch gauge
            Constraint::Length(1), // File read gauge
            Constraint::Length(1), // Processing rate sparkline
            Constraint::Length(1), // Learning rate sparkline
        ])
        .split(inner);

    let time_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(33),
            Constraint::Percentage(34),
            Constraint::Percentage(33),
        ])
        .split(layout[0]);

    frame.render_widget(
        Paragraph::new(format_elapsed(view.elapsed_secs)).alignment(Alignment::Left),
        time_chunks[0],
    );
    frame.render_widget(
        Paragraph::new(format!("{:.1}K pos/sec", view.samples_per_sec / 1000.0))
            .alignment(Alignment::Center),
        time_chunks[1],
    );
    frame.render_widget(
        Paragraph::new(format_eta(view.eta_secs)).alignment(Alignment::Right),
        time_chunks[2],
    );

    let overall_ratio = (view.super_batch as f64 / view.total_super_batches as f64).min(1.0);
    frame.render_widget(
        Gauge::default()
            .gauge_style(Style::default().fg(Color::Green))
            .ratio(overall_ratio)
            .label(Span::styled(
                format!(
                    "{:>6} / {:>6}  ({:>5.1}%)",
                    view.super_batch,
                    view.total_super_batches,
                    overall_ratio * 100.0,
                ),
                Style::default().fg(Color::White),
            )),
        layout[1],
    );

    let sb_ratio = (view.batch_in_super as f64 / view.batches_per_super_batch as f64).min(1.0);
    frame.render_widget(
        Gauge::default()
            .gauge_style(Style::default().fg(Color::Blue))
            .ratio(sb_ratio)
            .label(Span::styled(
                format!(
                    "{:>6} / {:>6}  ({:>5.1}%)",
                    view.batch_in_super,
                    view.batches_per_super_batch,
                    sb_ratio * 100.0,
                ),
                Style::default().fg(Color::White),
            )),
        layout[2],
    );

    let file_ratio = if view.data_positions > 0 {
        (view.positions_consumed as f64 / view.data_positions as f64).min(1.0)
    } else {
        0.0
    };
    frame.render_widget(
        Gauge::default()
            .gauge_style(Style::default().fg(Color::Yellow))
            .ratio(file_ratio)
            .label(Span::styled(
                format!(
                    "{:>6.1} / {:>6.1}M ({:>5.1}%)",
                    view.positions_consumed as f64 / 1_000_000.0,
                    view.data_positions as f64 / 1_000_000.0,
                    file_ratio * 100.0,
                ),
                Style::default().fg(Color::White),
            )),
        layout[3],
    );

    if !view.recent_rates.is_empty() {
        let max_bars = layout[4].width as usize;
        let data: Vec<u64> = view
            .recent_rates
            .iter()
            .rev()
            .take(max_bars)
            .rev()
            .copied()
            .collect();
        frame.render_widget(
            Sparkline::default()
                .data(&data)
                .style(Style::default().fg(Color::Cyan)),
            layout[4],
        );
    }

    if !view.lr_history.is_empty() {
        let width = layout[5].width as usize;
        let total_expected = view.total_super_batches * view.lr_samples_per_super_batch;
        let data: Vec<u64> = (0..width)
            .map(|i| {
                let idx = (i * total_expected) / width;
                if idx < view.lr_history.len() {
                    (view.lr_history[idx] * 1_000_000.0) as u64
                } else {
                    0
                }
            })
            .collect();
        frame.render_widget(
            Sparkline::default()
                .data(&data)
                .style(Style::default().fg(Color::Magenta)),
            layout[5],
        );
    }
}

pub struct DatasetBoxesView<'a> {
    pub input_file: &'a str,
    pub data_positions: usize,
    pub phase: Option<&'a str>,
    pub network_info: &'a str,
    pub last_saved: Option<&'a str>,
}

pub fn render_dataset_boxes(frame: &mut Frame, area: Rect, view: &DatasetBoxesView<'_>) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area);

    let dataset_block = Block::default().borders(Borders::ALL).title("Dataset");
    let dataset_inner = dataset_block.inner(cols[0]);
    frame.render_widget(dataset_block, cols[0]);
    let millions = view.data_positions as f64 / 1_000_000.0;
    let dataset_text = if let Some(phase) = view.phase {
        format!(
            "Input: {}\nPositions: {:.1}M  Phase: {phase}",
            view.input_file, millions
        )
    } else {
        format!("Input: {}\nPositions: {:.1}M", view.input_file, millions)
    };
    frame.render_widget(Paragraph::new(dataset_text), dataset_inner);

    let network_block = Block::default().borders(Borders::ALL).title("Network");
    let network_inner = network_block.inner(cols[1]);
    frame.render_widget(network_block, cols[1]);
    frame.render_widget(
        Paragraph::new(format!(
            "{}\nLast saved: {}",
            view.network_info,
            view.last_saved.unwrap_or("None"),
        )),
        network_inner,
    );
}

pub struct HistoryChartView<'a> {
    pub title: &'a str,
    pub data: &'a [f32],
    pub x_bound: f64,
    pub y_range_fallback: (f64, f64),
    pub y_max_clamp: Option<f64>,
    pub y_label_precision: usize,
    pub color: Color,
}

pub fn render_history_chart(frame: &mut Frame, area: Rect, view: &HistoryChartView<'_>) {
    let data_pairs: Vec<(f64, f64)> = view
        .data
        .iter()
        .enumerate()
        .map(|(i, &v)| ((i + 1) as f64, f64::from(v)))
        .collect();

    let (y_min, y_max) = if data_pairs.is_empty() {
        view.y_range_fallback
    } else {
        let max = data_pairs.iter().map(|(_, y)| *y).fold(0.0_f64, f64::max);
        let min = data_pairs.iter().map(|(_, y)| *y).fold(f64::MAX, f64::min);
        let range = max - min;
        if range < 1e-6 {
            let buffer = (max * 0.1).max(0.1);
            let y_max = match view.y_max_clamp {
                Some(clamp) => (max + buffer).min(clamp),
                None => max + buffer,
            };
            ((max - buffer).max(0.0), y_max)
        } else {
            let buffer = range * 0.1;
            let y_max = match view.y_max_clamp {
                Some(clamp) => (max + buffer).min(clamp),
                None => max + buffer,
            };
            ((min - buffer).max(0.0), y_max)
        }
    };

    let prec = view.y_label_precision;
    let datasets = vec![Dataset::default()
        .name(view.title)
        .marker(Marker::Braille)
        .graph_type(GraphType::Line)
        .style(Style::default().fg(view.color))
        .data(&data_pairs)];

    let chart = Chart::new(datasets)
        .block(Block::default().borders(Borders::ALL).title(view.title))
        .x_axis(Axis::default().bounds([0.0, view.x_bound]))
        .y_axis(Axis::default().bounds([y_min, y_max]).labels(vec![
            Span::raw(format!("{y_min:.prec$}")),
            Span::raw(format!("{y_max:.prec$}")),
        ]));

    frame.render_widget(chart, area);
}

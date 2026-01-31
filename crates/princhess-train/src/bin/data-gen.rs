#![warn(clippy::all, clippy::pedantic)]
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use std::array;
use std::collections::HashSet;
use std::fs::File;
use std::io::{self, BufWriter};
use std::ops::Neg;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use bytemuck::allocation;
use chrono::Utc;
use crossterm::cursor;
use crossterm::event::{poll, read, Event, KeyCode};
use crossterm::ExecutableCommand;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Alignment, Constraint, Direction, Layout};
use ratatui::style::{Color, Style};
use ratatui::text::Span;
use ratatui::widgets::{Block, Borders, Gauge, Paragraph, Sparkline};
use ratatui::{Frame, Terminal, TerminalOptions, Viewport};
use scc::{Guard, Queue};

use princhess::engine::{Engine, SCALE};
use princhess::graph::MoveEdge;
use princhess::math::{self, Rng};
use princhess::options::{EngineOptions, EvaluationOptions, MctsOptions};
use princhess::state::{self, State};

use princhess_train::args::Args;
use princhess_train::data::TrainingPosition;
use princhess_train::system;
use princhess_train::tui::{self, RawModeGuard};

const HASH_SIZE_MB: usize = 128;

const MAX_PLAYOUTS_PER_POSITION: u64 = 10000;
const KL_DIVERGENCE_THRESHOLD: f32 = 0.000_001;
const MAX_THREADS: u16 = 64;
const DFRC_PCT: u64 = 10;

const VARIATION_MIN_EVAL: f32 = 0.25;
const VARIATION_MAX_EVAL: f32 = 0.75;
const VARIATION_MIN_PHASE: usize = 16;

const ADJUDICATE_MAX_PIECE_COUNT: usize = 6;
const ADJUDICATE_DRAW_EVAL: f32 = 0.10;
const ADJUDICATE_DRAW_MOVES: usize = 8;
const ADJUDICATE_WIN_EVAL: f32 = 0.90;
const ADJUDICATE_WIN_MOVES: usize = 3;

const MAX_POSITIONS_PER_FILE: u64 = 20_000_000;
const DEFAULT_POSITIONS: u64 = 100_000_000;
const MAX_VARIATIONS: usize = 16;

const SAMPLE_INTERVAL_SECS: u64 = 5;
const RATE_WINDOW_SAMPLES: usize = 60;
const MAX_SAMPLES: usize = 512;

const ADJUSTMENT_WINDOW_SIZE: u64 = 8192;

const TARGET_PLAYOUTS: f32 = 500.0;
const PLAYOUTS_TOLERANCE: f32 = 25.0;
const CPUCT_ADJUSTMENT_STEP: u32 = 1;

const POLICY_TEMPERATURE: f32 = 1.4;

const TARGET_OPENING_UNBALANCED: f32 = 0.4;
const UNBALANCED_TOLERANCE: f32 = 0.02;
const OPENING_RANDOMNESS_ADJUSTMENT_STEP: u32 = 1;

// Histogram configuration: center point and bucket width (generates 11 thresholds for 12 buckets)
const POLICY_GINI_CENTER: f32 = 0.5;
const POLICY_GINI_WIDTH: f32 = 0.1;

const POLICY_KL_CENTER: f32 = 1.0;
const POLICY_KL_WIDTH: f32 = 0.2;

const EVAL_DISTRIBUTION_CENTER: f32 = 0.0;
const EVAL_DISTRIBUTION_WIDTH: f32 = 0.2;

const EVAL_RESULT_AGREEMENT_CENTER: f32 = 0.6;
const EVAL_RESULT_AGREEMENT_WIDTH: f32 = 0.1;

const POLICY_GINI_THRESHOLDS: [f32; 11] =
    generate_thresholds(POLICY_GINI_CENTER, POLICY_GINI_WIDTH);
const POLICY_KL_THRESHOLDS: [f32; 11] = generate_thresholds(POLICY_KL_CENTER, POLICY_KL_WIDTH);
const EVAL_DISTRIBUTION_THRESHOLDS: [f32; 11] =
    generate_thresholds(EVAL_DISTRIBUTION_CENTER, EVAL_DISTRIBUTION_WIDTH);
const EVAL_RESULT_AGREEMENT_THRESHOLDS: [f32; 11] =
    generate_thresholds(EVAL_RESULT_AGREEMENT_CENTER, EVAL_RESULT_AGREEMENT_WIDTH);

const EVAL_DELTA_BUCKETS: usize = 20;
const EVAL_DELTA_BUCKET_WIDTH: f32 = 0.05;

const TUI_GRID_HEIGHT: u16 = 9;
const TUI_HISTOGRAM_HEIGHT: u16 = 8;

fn tui_progress_box_height(threads: u16) -> u16 {
    3 + threads + 2
}

fn tui_total_height(threads: u16) -> u16 {
    tui_progress_box_height(threads) + TUI_GRID_HEIGHT + TUI_HISTOGRAM_HEIGHT + 2
}

fn load_atomic_array<const N: usize>(arr: &[AtomicU64; N]) -> [u64; N] {
    array::from_fn(|i| arr[i].load(Ordering::Relaxed))
}

fn zeroed_atomic_u64_array<const N: usize>() -> [AtomicU64; N] {
    array::from_fn(|_| AtomicU64::new(0))
}

fn zeroed_atomic_usize_array<const N: usize>() -> [AtomicUsize; N] {
    array::from_fn(|_| AtomicUsize::new(0))
}

fn to_bucket(value: f32, thresholds: &[f32]) -> usize {
    thresholds
        .iter()
        .position(|&t| value < t)
        .unwrap_or(thresholds.len())
}

const fn generate_thresholds(center: f32, width: f32) -> [f32; 11] {
    [
        center - 5.0 * width,
        center - 4.0 * width,
        center - 3.0 * width,
        center - 2.0 * width,
        center - width,
        center,
        center + width,
        center + 2.0 * width,
        center + 3.0 * width,
        center + 4.0 * width,
        center + 5.0 * width,
    ]
}

fn bucket_labels(thresholds: &[f32; 11]) -> [String; 6] {
    [
        format!("    -{:>4.1}", thresholds[1]),
        format!("{:>4.1}-{:>4.1}", thresholds[1], thresholds[3]),
        format!("{:>4.1}-{:>4.1}", thresholds[3], thresholds[5]),
        format!("{:>4.1}-{:>4.1}", thresholds[5], thresholds[7]),
        format!("{:>4.1}-{:>4.1}", thresholds[7], thresholds[9]),
        format!("{:>4.1}-    ", thresholds[9]),
    ]
}

/// Computes Gini impurity (1 - Σp²) of the visit distribution.
///
/// Returns a value in [0.0, 1.0] where:
/// - 0.0 = concentrated policy (all visits on one move)
/// - 1.0 = uniform policy (visits evenly distributed)
fn compute_policy_gini(visits: &[u8]) -> f32 {
    let total_visits: u64 = visits.iter().map(|&v| u64::from(v)).sum();
    math::gini(visits.iter().map(|&v| u32::from(v)), total_visits)
}

/// Computes KL divergence KL(P || Q) = Σ P(i) * ln(P(i) / Q(i)) between two distributions.
///
/// Uses Laplace smoothing (`ε=1.0`) to handle zero probabilities:
/// - `p = (p_values[i] + ε) / (p_total + num_moves * ε)`
/// - `q = (q_values[i] + ε) / (q_total + num_moves * ε)`
fn kl_divergence<T, U>(p_values: &[T], q_values: &[U]) -> f32
where
    T: Copy + Into<u64>,
    U: Copy + Into<u64>,
{
    let p_total: u64 = p_values.iter().map(|&x| x.into()).sum();
    let q_total: u64 = q_values.iter().map(|&x| x.into()).sum();

    let num_moves = p_values.len() as f32;
    let epsilon = 1.0;

    let p_sum = p_total as f32 + num_moves * epsilon;
    let q_sum = q_total as f32 + num_moves * epsilon;

    let mut kl = 0.0;

    for (&p_val, &q_val) in p_values.iter().zip(q_values.iter()) {
        let p = (p_val.into() as f32 + epsilon) / p_sum;
        let q = (q_val.into() as f32 + epsilon) / q_sum;
        kl += p * (p / q).ln();
    }

    kl
}

struct Stats {
    start: Instant,
    games: AtomicU64,
    positions: AtomicU64,
    skipped: AtomicU64,
    aborted: AtomicU64,
    white_wins: AtomicU64,
    black_wins: AtomicU64,
    draws: AtomicU64,
    blunders: AtomicU64,
    variations: AtomicUsize,
    nodes: AtomicUsize,
    playouts: AtomicUsize,
    depth: AtomicUsize,
    seldepth: AtomicUsize,
    opening_eval_sum: AtomicI64,
    opening_count: AtomicU64,
    variation_eval_sum: AtomicI64,
    variation_count: AtomicU64,
    recent_rates: Queue<u64>,
    last_sample_positions: AtomicU64,
    active_threads: AtomicUsize,
    thread_buffers: [AtomicUsize; MAX_THREADS as usize],
    policy_gini_buckets: [AtomicU64; 12],
    policy_gini_sum: AtomicU64,
    policy_kl_buckets: [AtomicU64; 12],
    policy_kl_sum: AtomicU64,
    eval_distribution_buckets: [AtomicU64; 12],
    eval_delta_distribution: [AtomicU64; EVAL_DELTA_BUCKETS],
    eval_result_agreement_buckets: [AtomicU64; 12],
    piece_count_distribution: [AtomicU64; 33],
    phase_distribution: [AtomicU64; 25],
    variation_phase_distribution: [AtomicU64; 25],
    duplicate_set: scc::HashSet<u64>,
    duplicate_piece_count_distribution: [AtomicU64; 33],
    window_playouts: AtomicU64,
    window_policy_kl: AtomicU64,
    window_opening_eval: AtomicU64,
    window_opening_count: AtomicU64,
    window_positions: AtomicU64,
    window_eval_result_disagreement: AtomicU64,
    eval_result_disagreement_sum: AtomicU64,
}

struct GameStats {
    pub positions: u64,
    pub skipped: u64,
    pub blunders: u64,
    pub variations: usize,
    pub nodes: usize,
    pub playouts: usize,
    pub depth: usize,
    pub seldepth: usize,
}

impl Stats {
    pub fn zero() -> Self {
        Self {
            start: Instant::now(),
            games: AtomicU64::new(0),
            positions: AtomicU64::new(0),
            skipped: AtomicU64::new(0),
            aborted: AtomicU64::new(0),
            white_wins: AtomicU64::new(0),
            black_wins: AtomicU64::new(0),
            draws: AtomicU64::new(0),
            blunders: AtomicU64::new(0),
            variations: AtomicUsize::new(0),
            nodes: AtomicUsize::new(0),
            playouts: AtomicUsize::new(0),
            depth: AtomicUsize::new(0),
            seldepth: AtomicUsize::new(0),
            opening_eval_sum: AtomicI64::new(0),
            opening_count: AtomicU64::new(0),
            variation_eval_sum: AtomicI64::new(0),
            variation_count: AtomicU64::new(0),
            recent_rates: Queue::default(),
            last_sample_positions: AtomicU64::new(0),
            active_threads: AtomicUsize::new(0),
            thread_buffers: zeroed_atomic_usize_array(),
            policy_gini_buckets: zeroed_atomic_u64_array(),
            policy_gini_sum: AtomicU64::new(0),
            policy_kl_buckets: zeroed_atomic_u64_array(),
            policy_kl_sum: AtomicU64::new(0),
            eval_distribution_buckets: zeroed_atomic_u64_array(),
            eval_delta_distribution: zeroed_atomic_u64_array(),
            eval_result_agreement_buckets: zeroed_atomic_u64_array(),
            piece_count_distribution: zeroed_atomic_u64_array(),
            phase_distribution: zeroed_atomic_u64_array(),
            variation_phase_distribution: zeroed_atomic_u64_array(),
            duplicate_set: scc::HashSet::new(),
            duplicate_piece_count_distribution: zeroed_atomic_u64_array(),
            window_playouts: AtomicU64::new(0),
            window_policy_kl: AtomicU64::new(0),
            window_opening_eval: AtomicU64::new(0),
            window_opening_count: AtomicU64::new(0),
            window_positions: AtomicU64::new(0),
            window_eval_result_disagreement: AtomicU64::new(0),
            eval_result_disagreement_sum: AtomicU64::new(0),
        }
    }

    fn add_game(&self, game: &GameStats) {
        self.games.fetch_add(1, Ordering::Relaxed);
        self.positions.fetch_add(game.positions, Ordering::Relaxed);
        self.skipped.fetch_add(game.skipped, Ordering::Relaxed);
        self.blunders.fetch_add(game.blunders, Ordering::Relaxed);
        self.variations
            .fetch_add(game.variations, Ordering::Relaxed);
        self.nodes.fetch_add(game.nodes, Ordering::Relaxed);
        self.playouts.fetch_add(game.playouts, Ordering::Relaxed);
        self.depth.fetch_add(game.depth, Ordering::Relaxed);
        self.seldepth.fetch_add(game.seldepth, Ordering::Relaxed);
    }

    pub fn add_white_win(&self, game: &GameStats) {
        self.add_game(game);
        self.white_wins.fetch_add(1, Ordering::Relaxed);
    }

    pub fn add_black_win(&self, game: &GameStats) {
        self.add_game(game);
        self.black_wins.fetch_add(1, Ordering::Relaxed);
    }

    pub fn add_draw(&self, game: &GameStats) {
        self.add_game(game);
        self.draws.fetch_add(1, Ordering::Relaxed);
    }

    pub fn add_aborted(&self) {
        self.aborted.fetch_add(1, Ordering::Relaxed);
    }

    /// Tracks opening position balance.
    /// Opening = initial seed position before game starts
    pub fn add_opening_eval(&self, eval: i64) {
        let abs_eval = eval.abs();
        self.opening_eval_sum.fetch_add(abs_eval, Ordering::Relaxed);
        self.opening_count.fetch_add(1, Ordering::Relaxed);
        self.window_opening_eval
            .fetch_add(abs_eval as u64, Ordering::Relaxed);
        self.window_opening_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Tracks variation position balance.
    /// Variation = alternative move branches explored during play
    pub fn add_variation_eval(&self, eval: i64) {
        let abs_eval = eval.abs();
        self.variation_eval_sum
            .fetch_add(abs_eval, Ordering::Relaxed);
        self.variation_count.fetch_add(1, Ordering::Relaxed);
    }

    fn view(
        &self,
        cpuct_quantized: &AtomicU32,
        opening_randomness_quantized: &AtomicU32,
        threads: u16,
        max_positions: u64,
    ) -> StatsView {
        let mut recent_rates = Vec::new();
        let guard = Guard::new();
        for rate in self.recent_rates.iter(&guard) {
            recent_rates.push(*rate);
        }

        StatsView {
            games: self.games.load(Ordering::Relaxed),
            positions: self.positions.load(Ordering::Relaxed),
            white_wins: self.white_wins.load(Ordering::Relaxed),
            draws: self.draws.load(Ordering::Relaxed),
            black_wins: self.black_wins.load(Ordering::Relaxed),
            skipped: self.skipped.load(Ordering::Relaxed),
            aborted: self.aborted.load(Ordering::Relaxed),
            blunders: self.blunders.load(Ordering::Relaxed),
            variations: self.variations.load(Ordering::Relaxed),
            nodes: self.nodes.load(Ordering::Relaxed),
            playouts: self.playouts.load(Ordering::Relaxed),
            depth: self.depth.load(Ordering::Relaxed),
            seldepth: self.seldepth.load(Ordering::Relaxed),
            opening_eval_sum: self.opening_eval_sum.load(Ordering::Relaxed),
            opening_count: self.opening_count.load(Ordering::Relaxed),
            variation_eval_sum: self.variation_eval_sum.load(Ordering::Relaxed),
            variation_count: self.variation_count.load(Ordering::Relaxed),
            elapsed_seconds: self.start.elapsed().as_secs().max(1),
            recent_rates,
            threads,
            max_positions,
            thread_buffers: array::from_fn(|i| self.thread_buffers[i].load(Ordering::Relaxed)),
            policy_gini_buckets: load_atomic_array(&self.policy_gini_buckets),
            policy_gini_sum: self.policy_gini_sum.load(Ordering::Relaxed),
            policy_kl_buckets: load_atomic_array(&self.policy_kl_buckets),
            policy_kl_sum: self.policy_kl_sum.load(Ordering::Relaxed),
            eval_distribution_buckets: load_atomic_array(&self.eval_distribution_buckets),
            eval_delta_distribution: load_atomic_array(&self.eval_delta_distribution),
            eval_result_agreement_buckets: load_atomic_array(&self.eval_result_agreement_buckets),
            piece_count_distribution: load_atomic_array(&self.piece_count_distribution),
            phase_distribution: load_atomic_array(&self.phase_distribution),
            variation_phase_distribution: load_atomic_array(&self.variation_phase_distribution),
            duplicate_count: self.positions.load(Ordering::Relaxed)
                - self.duplicate_set.len() as u64,
            duplicate_piece_count_distribution: load_atomic_array(
                &self.duplicate_piece_count_distribution,
            ),
            cpuct: cpuct_quantized.load(Ordering::Relaxed) as f32 / 100.0,
            opening_randomness: opening_randomness_quantized.load(Ordering::Relaxed) as f32 / 100.0,
            eval_result_disagreement_sum: self.eval_result_disagreement_sum.load(Ordering::Relaxed),
        }
    }
}

struct StatsView {
    games: u64,
    positions: u64,
    white_wins: u64,
    draws: u64,
    black_wins: u64,
    skipped: u64,
    aborted: u64,
    blunders: u64,
    variations: usize,
    nodes: usize,
    playouts: usize,
    depth: usize,
    seldepth: usize,
    opening_eval_sum: i64,
    opening_count: u64,
    variation_eval_sum: i64,
    variation_count: u64,
    elapsed_seconds: u64,
    recent_rates: Vec<u64>,
    threads: u16,
    max_positions: u64,
    thread_buffers: [usize; MAX_THREADS as usize],
    policy_gini_buckets: [u64; 12],
    policy_gini_sum: u64,
    policy_kl_buckets: [u64; 12],
    policy_kl_sum: u64,
    eval_distribution_buckets: [u64; 12],
    eval_delta_distribution: [u64; EVAL_DELTA_BUCKETS],
    eval_result_agreement_buckets: [u64; 12],
    piece_count_distribution: [u64; 33],
    phase_distribution: [u64; 25],
    variation_phase_distribution: [u64; 25],
    duplicate_count: u64,
    duplicate_piece_count_distribution: [u64; 33],
    cpuct: f32,
    opening_randomness: f32,
    eval_result_disagreement_sum: u64,
}

impl StatsView {
    fn per_game<T: Into<f32>>(&self, value: T) -> f32 {
        if self.games > 0 {
            value.into() / self.games as f32
        } else {
            0.0
        }
    }

    fn per_position<T: Into<f32>>(&self, value: T) -> f32 {
        if self.positions > 0 {
            value.into() / self.positions as f32
        } else {
            0.0
        }
    }

    fn positions_per_game(&self) -> u64 {
        if self.games > 0 {
            self.positions / self.games
        } else {
            0
        }
    }

    fn white_win_pct(&self) -> u16 {
        (self.per_game(self.white_wins as f32) * 100.0).round() as u16
    }

    fn draw_pct(&self) -> u16 {
        (self.per_game(self.draws as f32) * 100.0).round() as u16
    }

    fn black_win_pct(&self) -> u16 {
        (self.per_game(self.black_wins as f32) * 100.0).round() as u16
    }

    fn blunder_pct(&self) -> f32 {
        self.per_game(self.blunders as f32) * 100.0
    }

    fn variation_pct(&self) -> f32 {
        self.per_position(self.variations as f32) * 100.0
    }

    fn skipped_pct(&self) -> f32 {
        self.per_position(self.skipped as f32) * 100.0
    }

    fn aborted_pct(&self) -> f32 {
        self.per_game(self.aborted as f32) * 100.0
    }

    fn avg_nodes(&self) -> usize {
        self.per_position(self.nodes as f32) as usize
    }

    fn avg_playouts(&self) -> usize {
        self.per_position(self.playouts as f32) as usize
    }

    fn avg_depth(&self) -> usize {
        self.per_position(self.depth as f32) as usize
    }

    fn avg_seldepth(&self) -> usize {
        self.per_position(self.seldepth as f32) as usize
    }

    fn avg_opening_eval(&self) -> f32 {
        if self.opening_count > 0 {
            self.opening_eval_sum as f32 / self.opening_count as f32 / SCALE
        } else {
            0.0
        }
    }

    fn avg_variation_eval(&self) -> f32 {
        if self.variation_count > 0 {
            self.variation_eval_sum as f32 / self.variation_count as f32 / SCALE
        } else {
            0.0
        }
    }

    fn positions_millions(&self) -> f32 {
        self.positions as f32 / 1_000_000.0
    }

    fn avg_positions_per_hour(&self) -> u64 {
        if self.recent_rates.is_empty() {
            self.positions * 3600 / self.elapsed_seconds
        } else {
            let window = self.recent_rates.len().min(RATE_WINDOW_SAMPLES);
            let sum: u64 = self.recent_rates.iter().rev().take(window).sum();
            sum / window as u64
        }
    }

    fn positions_per_hour_millions(&self) -> f32 {
        self.avg_positions_per_hour() as f32 / 1_000_000.0
    }

    fn progress_ratio(&self, max_positions: u64) -> f64 {
        self.positions as f64 / max_positions as f64
    }

    fn avg_policy_gini(&self) -> f32 {
        let count: u64 = self.policy_gini_buckets.iter().sum();
        if count > 0 {
            self.policy_gini_sum as f32 / count as f32 / SCALE
        } else {
            0.0
        }
    }

    fn avg_policy_kl(&self) -> f32 {
        let count: u64 = self.policy_kl_buckets.iter().sum();
        if count > 0 {
            self.policy_kl_sum as f32 / count as f32 / SCALE
        } else {
            0.0
        }
    }

    fn avg_eval_result_disagreement(&self) -> f32 {
        if self.positions > 0 {
            self.eval_result_disagreement_sum as f32 / self.positions as f32 / SCALE
        } else {
            0.0
        }
    }
}

impl GameStats {
    pub fn zero() -> Self {
        Self {
            positions: 0,
            skipped: 0,
            blunders: 0,
            variations: 0,
            nodes: 0,
            playouts: 0,
            depth: 0,
            seldepth: 0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum GameResult {
    WhiteWin,
    Draw,
    BlackWin,
    Aborted,
}

impl From<GameResult> for i8 {
    fn from(result: GameResult) -> Self {
        match result {
            GameResult::WhiteWin => 1,
            GameResult::Draw => 0,
            GameResult::BlackWin => -1,
            GameResult::Aborted => unreachable!(),
        }
    }
}

impl Neg for GameResult {
    type Output = Self;

    fn neg(self) -> Self::Output {
        match self {
            GameResult::WhiteWin => GameResult::BlackWin,
            GameResult::BlackWin => GameResult::WhiteWin,
            GameResult::Draw => GameResult::Draw,
            GameResult::Aborted => GameResult::Aborted,
        }
    }
}

#[allow(clippy::too_many_lines)]
fn run_game(
    stats: &Stats,
    cpuct_quantized: &AtomicU32,
    opening_randomness_quantized: &AtomicU32,
    positions: &mut Vec<TrainingPosition>,
    rng: &mut Rng,
    stop_signal: &AtomicBool,
) {
    let mut variations = Vec::with_capacity(MAX_VARIATIONS + 1);
    let mut variations_count = 0;
    let mut seen_positions = HashSet::new();

    variations.push(random_start(rng, opening_randomness_quantized));

    let cpuct = cpuct_quantized.load(Ordering::Relaxed) as f32 / 100.0;

    let mcts_options = MctsOptions {
        cpuct,
        cpuct_tau: 0.5,
        cpuct_jitter: 0.0,
        cpuct_trend_adjustment: 0.0,
        cpuct_gini_base: 1.0,
        cpuct_gini_factor: 0.0,
        cpuct_gini_max: 1.0,
        policy_temperature: 1.0,
        policy_temperature_root: POLICY_TEMPERATURE,
    };

    let evaluation_options = EvaluationOptions {
        enable_material_scaling: false,
        enable_50mr_scaling: true,
    };

    let engine_options = EngineOptions {
        mcts_options,
        evaluation_options,
        hash_size_mb: HASH_SIZE_MB,
        ..EngineOptions::default()
    };

    while let Some(mut state) = variations.pop() {
        if stop_signal.load(Ordering::Relaxed) {
            return;
        }

        let mut game_stats = GameStats::zero();
        let mut engine = Engine::new(state.clone(), engine_options);

        if !state.is_available_move() {
            continue;
        }

        if state.drawn_by_fifty_move_rule()
            || state.is_repetition()
            || state.board().is_insufficient_material()
        {
            continue;
        }

        let mut game_positions = Vec::with_capacity(256);
        let mut is_first_move = true;
        let mut previous_eval = 0.0;
        let mut consecutive_draw_moves = 0;
        let mut consecutive_win_moves = 0;
        let result = loop {
            engine.set_root_state(state.clone());
            let legal_moves = engine.root_edges().len();

            if legal_moves > 1 {
                let mut previous_visits = vec![0u32; legal_moves];

                // Early stopping based on KL divergence convergence
                // Compare visit distributions between successive batches and stop when the
                // per-playout KL gain falls below threshold, indicating search has converged
                for _playout in 0..MAX_PLAYOUTS_PER_POSITION {
                    engine.playout_sync(1);

                    let current_visits: Vec<u32> =
                        engine.root_edges().iter().map(MoveEdge::visits).collect();

                    if let Some(gain) = kld_gain(&current_visits, &previous_visits) {
                        if gain < KL_DIVERGENCE_THRESHOLD {
                            break;
                        }
                    }

                    previous_visits = current_visits;
                }
            }

            let best_move = engine.best_move();

            if legal_moves == 1 || legal_moves > TrainingPosition::MAX_MOVES {
                game_stats.skipped += 1;
            } else {
                // Check for mate before creating TrainingPosition (which clamps the eval)
                let raw_eval = engine.mcts().best_edge().reward().average as f32 / SCALE;
                let raw_eval_stm = state.side_to_move().fold(raw_eval, -raw_eval);

                if raw_eval_stm.abs() > 1.0 {
                    break if raw_eval_stm > 0.0 {
                        GameResult::WhiteWin
                    } else {
                        GameResult::BlackWin
                    };
                }

                let position = TrainingPosition::from(engine.mcts());
                let eval = position.evaluation();

                // Track KL divergence between final visits and policy
                let policy_kl = policy_kl_divergence(&engine);
                let kl_bucket = to_bucket(policy_kl, &POLICY_KL_THRESHOLDS);
                stats.policy_kl_buckets[kl_bucket].fetch_add(1, Ordering::Relaxed);
                let kl_quantized = (policy_kl * SCALE) as u64;
                stats
                    .policy_kl_sum
                    .fetch_add(kl_quantized, Ordering::Relaxed);
                stats
                    .window_policy_kl
                    .fetch_add(kl_quantized, Ordering::Relaxed);

                if variations_count < MAX_VARIATIONS {
                    let variation = engine.most_visited_move();

                    if variation != best_move
                        && (VARIATION_MIN_EVAL..=VARIATION_MAX_EVAL).contains(&eval.abs())
                        && state.phase() >= VARIATION_MIN_PHASE
                    {
                        let explore_probability = 1.0 / (variations_count + 1) as f32;
                        if rng.next_f32() < explore_probability {
                            game_stats.variations += 1;
                            let mut variation_state = state.clone();
                            variation_state.make_move(variation);
                            stats.variation_phase_distribution[variation_state.phase()]
                                .fetch_add(1, Ordering::Relaxed);
                            variations.push(variation_state);
                            variations_count += 1;
                        }
                    }
                }

                if is_first_move {
                    if variations_count == 0 {
                        stats.add_opening_eval((eval * SCALE) as i64);
                    } else {
                        stats.add_variation_eval((eval * SCALE) as i64);
                    }
                }

                if !is_first_move {
                    let delta = eval - previous_eval;
                    let delta_bucket = ((delta.abs() / EVAL_DELTA_BUCKET_WIDTH) as usize)
                        .min(EVAL_DELTA_BUCKETS - 1);
                    stats.eval_delta_distribution[delta_bucket].fetch_add(1, Ordering::Relaxed);
                }
                previous_eval = eval;

                game_positions.push((position, state.hash()));

                game_stats.positions += 1;
                game_stats.nodes += engine.mcts().num_nodes();
                game_stats.playouts += engine.mcts().playouts();
                game_stats.depth += engine.mcts().depth();
                game_stats.seldepth += engine.mcts().max_depth();

                if eval.abs() <= ADJUDICATE_DRAW_EVAL {
                    consecutive_draw_moves += 1;
                    consecutive_win_moves = 0;
                } else if eval.abs() >= ADJUDICATE_WIN_EVAL {
                    consecutive_win_moves += 1;
                    consecutive_draw_moves = 0;
                } else {
                    consecutive_draw_moves = 0;
                    consecutive_win_moves = 0;
                }
            }

            is_first_move = false;

            state.make_move(best_move);

            if !state.is_available_move() {
                break if state.is_check() {
                    // The stm has been checkmated. Convert to white relative result
                    state
                        .side_to_move()
                        .fold(GameResult::BlackWin, GameResult::WhiteWin)
                } else {
                    GameResult::Draw
                };
            }

            if state.drawn_by_fifty_move_rule()
                || state.is_repetition()
                || state.board().is_insufficient_material()
            {
                break GameResult::Draw;
            }

            let piece_count = state.board().occupied().count();
            let base = ((ADJUDICATE_MAX_PIECE_COUNT + 1) as f32 - piece_count as f32)
                / (ADJUDICATE_MAX_PIECE_COUNT - 1) as f32;
            let adjudication_probability = base.powi(3);

            if adjudication_probability > 0.0 && rng.next_f32() < adjudication_probability {
                if consecutive_draw_moves >= ADJUDICATE_DRAW_MOVES {
                    break GameResult::Draw;
                }

                if consecutive_win_moves >= ADJUDICATE_WIN_MOVES {
                    break if previous_eval > 0.0 {
                        GameResult::WhiteWin
                    } else {
                        GameResult::BlackWin
                    };
                }
            }

            if !seen_positions.insert(state.hash()) {
                game_positions.clear();
                break GameResult::Aborted;
            }
        };

        let mut blunder = false;

        for (position, hash) in &mut game_positions {
            position.set_result(i8::from(result));

            let eval_white = position.evaluation();
            let eval_stm = position.stm_relative_evaluation();

            let disagreement = (eval_white - f32::from(i8::from(result))).abs();
            stats.eval_result_agreement_buckets
                [to_bucket(disagreement, &EVAL_RESULT_AGREEMENT_THRESHOLDS)]
            .fetch_add(1, Ordering::Relaxed);

            // Track disagreement for adaptive temperature tuning
            let disagreement_quantized = (disagreement * SCALE) as u64;
            stats
                .eval_result_disagreement_sum
                .fetch_add(disagreement_quantized, Ordering::Relaxed);
            stats
                .window_eval_result_disagreement
                .fetch_add(disagreement_quantized, Ordering::Relaxed);

            // Track data distributions for saved positions only
            let moves = position.moves();
            let visits: Vec<u8> = moves.iter().map(|(_, v)| *v).collect();
            let gini = compute_policy_gini(&visits);
            let gini_bucket = to_bucket(gini, &POLICY_GINI_THRESHOLDS);
            stats.policy_gini_buckets[gini_bucket].fetch_add(1, Ordering::Relaxed);
            stats
                .policy_gini_sum
                .fetch_add((gini * SCALE) as u64, Ordering::Relaxed);

            let eval_bucket = to_bucket(eval_stm, &EVAL_DISTRIBUTION_THRESHOLDS);
            stats.eval_distribution_buckets[eval_bucket].fetch_add(1, Ordering::Relaxed);

            let piece_count = position.piece_count();
            stats.piece_count_distribution[piece_count].fetch_add(1, Ordering::Relaxed);
            stats.phase_distribution[position.phase()].fetch_add(1, Ordering::Relaxed);

            if stats.duplicate_set.insert_sync(*hash).is_err() {
                stats.duplicate_piece_count_distribution[piece_count]
                    .fetch_add(1, Ordering::Relaxed);
            }

            blunder |= match result {
                GameResult::WhiteWin => eval_white < -0.5,
                GameResult::BlackWin => eval_white > 0.5,
                GameResult::Draw => eval_white.abs() > 0.75,
                GameResult::Aborted => false,
            }
        }

        let saved_positions: Vec<TrainingPosition> =
            game_positions.into_iter().map(|(pos, _)| pos).collect();
        positions.extend(saved_positions);

        if blunder {
            game_stats.blunders += 1;
        }

        match result {
            GameResult::WhiteWin => stats.add_white_win(&game_stats),
            GameResult::Draw => stats.add_draw(&game_stats),
            GameResult::BlackWin => stats.add_black_win(&game_stats),
            GameResult::Aborted => stats.add_aborted(),
        }

        if !matches!(result, GameResult::Aborted) {
            stats
                .window_playouts
                .fetch_add(game_stats.playouts as u64, Ordering::Relaxed);
            stats
                .window_positions
                .fetch_add(game_stats.positions, Ordering::Relaxed);
            adjust_parameters(stats, cpuct_quantized, opening_randomness_quantized);
        }
    }
}

fn random_start(rng: &mut Rng, opening_randomness_quantized: &AtomicU32) -> State {
    let opening_randomness = opening_randomness_quantized.load(Ordering::Relaxed) as f32 / 100.0;
    let (_moves, state) = state::generate_random_opening(rng, DFRC_PCT, opening_randomness);
    state
}

/// Computes KL divergence per additional playout between old and new visit distributions.
///
/// Returns KL divergence divided by the visit difference to normalize gain per playout.
/// Returns `None` if `old_visits` is zero or `new_visits` ≤ `old_visits`.
fn kld_gain(new_visits: &[u32], old_visits: &[u32]) -> Option<f32> {
    let new_parent_visits: u64 = new_visits.iter().map(|&x| u64::from(x)).sum();
    let old_parent_visits: u64 = old_visits.iter().map(|&x| u64::from(x)).sum();

    if old_parent_visits == 0 || new_parent_visits <= old_parent_visits {
        return None;
    }

    let kl = kl_divergence(old_visits, new_visits);

    let parent_visits_diff = new_parent_visits.saturating_sub(old_parent_visits) as f32;
    if parent_visits_diff > 0.0 {
        Some(kl / parent_visits_diff)
    } else {
        None
    }
}

/// Computes KL divergence between final visit distribution and policy distribution.
///
/// Measures how much the MCTS search diverged from the neural network policy.
/// Normalizes visits to the same scale as policies (SCALE) for fair Laplace smoothing.
fn policy_kl_divergence(engine: &Engine) -> f32 {
    let visits: Vec<u32> = engine.root_edges().iter().map(MoveEdge::visits).collect();
    let policies: Vec<u16> = engine.root_edges().iter().map(MoveEdge::policy).collect();

    // Scale visits to match policy scale for fair Laplace smoothing
    let total_visits: u64 = visits.iter().map(|&v| u64::from(v)).sum();
    let scaled_visits: Vec<u32> = if total_visits > 0 {
        visits
            .iter()
            .map(|&v| ((u64::from(v) * SCALE as u64) / total_visits) as u32)
            .collect()
    } else {
        visits
    };

    kl_divergence(&scaled_visits, &policies)
}

fn render_two_column_box(
    frame: &mut Frame,
    area: ratatui::layout::Rect,
    title: &str,
    left_content: String,
    right_content: String,
) {
    let block = Block::default().borders(Borders::ALL).title(title);
    frame.render_widget(block, area);

    let inner = area.inner(ratatui::layout::Margin {
        horizontal: 1,
        vertical: 1,
    });

    let columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(inner);

    let left = Paragraph::new(left_content);
    frame.render_widget(left, columns[0]);

    let right = Paragraph::new(right_content);
    frame.render_widget(right, columns[1]);
}

fn render_histogram(
    frame: &mut Frame,
    area: ratatui::layout::Rect,
    title: &str,
    buckets: &[u64; 12],
    labels: &[String; 6],
) {
    let total: u64 = buckets.iter().sum();
    if total == 0 {
        return;
    }

    let block = Block::default().borders(Borders::ALL).title(title);
    frame.render_widget(block, area);

    let inner = area.inner(ratatui::layout::Margin {
        horizontal: 1,
        vertical: 1,
    });

    let row_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1); 6])
        .split(inner);

    for row in 0..6 {
        let top_idx = row * 2;
        let bottom_idx = row * 2 + 1;

        let top_ratio = buckets[top_idx] as f64 / total as f64;
        let bottom_ratio = buckets[bottom_idx] as f64 / total as f64;

        let col_layout = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Length(10),
                Constraint::Length(4),
                Constraint::Min(0),
            ])
            .split(row_layout[row]);

        let label_widget =
            Paragraph::new(labels[row].as_str()).style(Style::default().fg(Color::Gray));
        frame.render_widget(label_widget, col_layout[0]);

        let combined_ratio = top_ratio + bottom_ratio;
        let combined_pct = (combined_ratio * 100.0).round() as u8;
        let pct_text = if (buckets[top_idx] > 0 || buckets[bottom_idx] > 0) && combined_pct == 0 {
            ">0%".to_string()
        } else {
            format!("{combined_pct:>2}%")
        };
        let pct_widget = Paragraph::new(pct_text).style(Style::default().fg(Color::Cyan));
        frame.render_widget(pct_widget, col_layout[1]);

        render_double_bar(frame, col_layout[2], top_ratio, bottom_ratio);
    }
}

fn render_double_bar(
    frame: &mut Frame,
    area: ratatui::layout::Rect,
    top_ratio: f64,
    bottom_ratio: f64,
) {
    let width = area.width as usize;
    // Scale so 50% fills the entire width (2x scale, capped at 1.0)
    let top_scaled = (top_ratio * 2.0).min(1.0);
    let bottom_scaled = (bottom_ratio * 2.0).min(1.0);
    let top_width = (top_scaled * width as f64).round() as usize;
    let bottom_width = (bottom_scaled * width as f64).round() as usize;

    let mut bar = String::new();
    for i in 0..width {
        let top_filled = i < top_width;
        let bottom_filled = i < bottom_width;

        let ch = match (top_filled, bottom_filled) {
            (true, true) => '█',   // Full block
            (true, false) => '▀',  // Upper half
            (false, true) => '▄',  // Lower half
            (false, false) => ' ', // Empty
        };
        bar.push(ch);
    }

    let widget = Paragraph::new(bar).style(Style::default().fg(Color::Cyan));
    frame.render_widget(widget, area);
}

#[allow(clippy::too_many_lines)]
fn render_tui(frame: &mut Frame, view: &StatsView) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .margin(1)
        .constraints([
            Constraint::Length(tui_progress_box_height(view.threads)),
            Constraint::Length(TUI_GRID_HEIGHT),
            Constraint::Length(TUI_HISTOGRAM_HEIGHT),
        ])
        .split(frame.area());

    // Progress box
    let progress_block = Block::default().borders(Borders::ALL).title("Progress");
    frame.render_widget(progress_block, chunks[0]);

    let progress_inner = chunks[0].inner(ratatui::layout::Margin {
        horizontal: 1,
        vertical: 1,
    });

    let progress_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(view.threads),
        ])
        .split(progress_inner);

    // Time/rate line
    let time_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(33),
            Constraint::Percentage(34),
            Constraint::Percentage(33),
        ])
        .split(progress_layout[0]);

    let elapsed =
        Paragraph::new(tui::format_elapsed(view.elapsed_seconds)).alignment(Alignment::Left);
    frame.render_widget(elapsed, time_chunks[0]);

    let rate = Paragraph::new(format!("{:.1}M/h", view.positions_per_hour_millions()))
        .alignment(Alignment::Center);
    frame.render_widget(rate, time_chunks[1]);

    let positions_remaining = view.max_positions.saturating_sub(view.positions);
    let positions_per_second = view.avg_positions_per_hour() / 3600;
    let eta_seconds = if positions_per_second == 0 {
        0
    } else {
        positions_remaining / positions_per_second
    };
    let eta = Paragraph::new(tui::format_eta(eta_seconds)).alignment(Alignment::Right);
    frame.render_widget(eta, time_chunks[2]);

    // Progress gauge
    let progress_ratio = view.progress_ratio(view.max_positions).min(1.0);
    let label = Span::styled(
        format!(
            "{:.1}M / {:.1}M ({:.1}%)",
            view.positions_millions(),
            view.max_positions as f32 / 1_000_000.0,
            progress_ratio * 100.0,
        ),
        Style::default().fg(Color::White),
    );
    let progress_gauge = Gauge::default()
        .gauge_style(Style::default().fg(Color::Green))
        .ratio(progress_ratio)
        .label(label);
    frame.render_widget(progress_gauge, progress_layout[1]);

    // Sparkline for rate trend
    if !view.recent_rates.is_empty() {
        let max_bars = progress_layout[2].width as usize;
        let data: Vec<u64> = if view.recent_rates.len() <= max_bars {
            view.recent_rates.clone()
        } else {
            view.recent_rates
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
        frame.render_widget(sparkline, progress_layout[2]);
    }

    // Thread buffer line gauges
    for (i, &buffer_size) in view
        .thread_buffers
        .iter()
        .take(view.threads as usize)
        .enumerate()
    {
        let ratio = (buffer_size as f64 / TrainingPosition::BUFFER_COUNT as f64).min(1.0);

        let line_gauge = Gauge::default()
            .gauge_style(Style::default().fg(Color::Blue))
            .ratio(ratio)
            .label(Span::styled(
                format!("T{i}"),
                Style::default().fg(Color::White),
            ));

        let thread_area = ratatui::layout::Rect {
            x: progress_layout[3].x,
            y: progress_layout[3].y + i as u16,
            width: progress_layout[3].width,
            height: 1,
        };
        frame.render_widget(line_gauge, thread_area);
    }

    let grid_rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(4), Constraint::Length(5)])
        .split(chunks[1]);

    let histogram_row = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
        ])
        .split(chunks[2]);

    let top_row = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(grid_rows[0]);

    let bottom_row = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(grid_rows[1]);

    let games_block = Block::default().borders(Borders::ALL).title("Games");
    frame.render_widget(games_block, top_row[0]);

    let games_inner = top_row[0].inner(ratatui::layout::Margin {
        horizontal: 1,
        vertical: 1,
    });

    let games_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Length(1)])
        .split(games_inner);

    let stats_columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(games_layout[0]);

    let white_pct = view.white_win_pct();
    let draw_pct = view.draw_pct();
    let black_pct = view.black_win_pct();

    let left_text = Paragraph::new(format!("Total: {}", view.games));
    frame.render_widget(left_text, stats_columns[0]);

    let right_text = Paragraph::new(format!("Pos/Game: {:>5}", view.positions_per_game()));
    frame.render_widget(right_text, stats_columns[1]);

    let bar_area = games_layout[1];

    // Ensure percentages always sum to 100 to avoid flickering
    let draw_constraint = 100u16.saturating_sub(white_pct + black_pct);

    let bar_layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(white_pct),
            Constraint::Percentage(draw_constraint),
            Constraint::Percentage(black_pct),
        ])
        .split(bar_area);

    if white_pct > 0 {
        let label = Span::styled(format!("{white_pct}%"), Style::default().fg(Color::Black));
        let white_bar = Gauge::default()
            .gauge_style(Style::default().fg(Color::White).bg(Color::White))
            .percent(100)
            .label(label);
        frame.render_widget(white_bar, bar_layout[0]);
    }

    if draw_pct > 0 {
        let label = Span::styled(format!("{draw_pct}%"), Style::default().fg(Color::Black));
        let draw_bar = Gauge::default()
            .gauge_style(
                Style::default()
                    .fg(Color::LightYellow)
                    .bg(Color::LightYellow),
            )
            .percent(100)
            .label(label);
        frame.render_widget(draw_bar, bar_layout[1]);
    }

    if black_pct > 0 {
        let label = Span::styled(format!("{black_pct}%"), Style::default().fg(Color::White));
        let black_bar = Gauge::default()
            .gauge_style(Style::default().fg(Color::LightGreen).bg(Color::LightGreen))
            .percent(100)
            .label(label);
        frame.render_widget(black_bar, bar_layout[2]);
    }

    render_two_column_box(
        frame,
        top_row[1],
        "Search Stats (avg per position)",
        format!(
            "Nodes:    {:>5}\nDepth:    {:>2}/{:<2}",
            view.avg_nodes(),
            view.avg_depth(),
            view.avg_seldepth(),
        ),
        format!(
            "Playouts: {:>5} [CPuct: {:.2}]\n",
            view.avg_playouts(),
            view.cpuct,
        ),
    );

    render_two_column_box(
        frame,
        bottom_row[0],
        "Quality Metrics",
        format!(
            "Blunders: {:>5.1}%\nSkipped:  {:>5.1}%\nOpening:   {:>+5.2} [Adv: {:.2}]",
            view.blunder_pct(),
            view.skipped_pct(),
            view.avg_opening_eval(),
            view.opening_randomness,
        ),
        format!(
            "Variations: {:>5.1}%\nAborted:    {:>5.1}%\nEval:        {:>+5.2}",
            view.variation_pct(),
            view.aborted_pct(),
            view.avg_variation_eval(),
        ),
    );

    render_histogram(
        frame,
        histogram_row[0],
        &format!("Policy Gini (μ={:.2})", view.avg_policy_gini()),
        &view.policy_gini_buckets,
        &bucket_labels(&POLICY_GINI_THRESHOLDS),
    );
    render_histogram(
        frame,
        histogram_row[1],
        &format!("Policy KL (μ={:.2})", view.avg_policy_kl()),
        &view.policy_kl_buckets,
        &bucket_labels(&POLICY_KL_THRESHOLDS),
    );
    render_histogram(
        frame,
        histogram_row[2],
        "Eval Distribution (STM)",
        &view.eval_distribution_buckets,
        &bucket_labels(&EVAL_DISTRIBUTION_THRESHOLDS),
    );
    render_histogram(
        frame,
        histogram_row[3],
        &format!(
            "Eval-Result Agreement (μ={:.2})",
            view.avg_eval_result_disagreement()
        ),
        &view.eval_result_agreement_buckets,
        &bucket_labels(&EVAL_RESULT_AGREEMENT_THRESHOLDS),
    );

    // Distribution sparklines in bottom_row[1]
    let distribution_block = Block::default()
        .borders(Borders::ALL)
        .title("Distributions");
    frame.render_widget(distribution_block, bottom_row[1]);

    let distribution_inner = bottom_row[1].inner(ratatui::layout::Margin {
        horizontal: 1,
        vertical: 1,
    });

    let main_split = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(distribution_inner);

    let sparkline_layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(13), Constraint::Min(0)])
        .split(main_split[0]);

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
        ])
        .split(sparkline_layout[1]);

    let labels = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
        ])
        .split(sparkline_layout[0]);

    // Piece count label and sparkline
    let piece_label = Paragraph::new("Piece Count:").style(Style::default().fg(Color::Gray));
    frame.render_widget(piece_label, labels[0]);

    let piece_data: Vec<u64> = view.piece_count_distribution[2..33].to_vec();
    let piece_sparkline = Sparkline::default()
        .data(&piece_data)
        .style(Style::default().fg(Color::Green));
    frame.render_widget(piece_sparkline, rows[0]);

    // Phase label and sparkline
    let phase_label = Paragraph::new("Phase:").style(Style::default().fg(Color::Gray));
    frame.render_widget(phase_label, labels[1]);

    let phase_data: Vec<u64> = view.phase_distribution.to_vec();
    let phase_sparkline = Sparkline::default()
        .data(&phase_data)
        .style(Style::default().fg(Color::Cyan));
    frame.render_widget(phase_sparkline, rows[1]);

    // Variation phase label and sparkline
    let var_phase_label = Paragraph::new("Var Phase:").style(Style::default().fg(Color::Gray));
    frame.render_widget(var_phase_label, labels[2]);

    let var_phase_data: Vec<u64> = view.variation_phase_distribution.to_vec();
    let var_phase_sparkline = Sparkline::default()
        .data(&var_phase_data)
        .style(Style::default().fg(Color::Yellow));
    frame.render_widget(var_phase_sparkline, rows[2]);

    let duplicate_layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(13), Constraint::Min(0)])
        .split(main_split[1]);

    let duplicate_rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
        ])
        .split(duplicate_layout[1]);

    let duplicate_labels = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
        ])
        .split(duplicate_layout[0]);

    let duplicate_pct = if view.positions > 0 {
        (view.duplicate_count as f32 / view.positions as f32) * 100.0
    } else {
        0.0
    };

    let dup_label = Paragraph::new("Duplicates:").style(Style::default().fg(Color::Gray));
    frame.render_widget(dup_label, duplicate_labels[0]);

    let dup_pct =
        Paragraph::new(format!("{duplicate_pct:>5.1}%")).style(Style::default().fg(Color::Gray));
    frame.render_widget(dup_pct, duplicate_rows[0]);

    let dup_sparkline_label = Paragraph::new("Dup by PC:").style(Style::default().fg(Color::Gray));
    frame.render_widget(dup_sparkline_label, duplicate_labels[1]);

    let dup_data: Vec<u64> = view.duplicate_piece_count_distribution[2..33].to_vec();
    let dup_sparkline = Sparkline::default()
        .data(&dup_data)
        .style(Style::default().fg(Color::Red));
    frame.render_widget(dup_sparkline, duplicate_rows[1]);

    let eval_delta_label = Paragraph::new("Eval Delta:").style(Style::default().fg(Color::Gray));
    frame.render_widget(eval_delta_label, duplicate_labels[2]);

    let eval_delta_data: Vec<u64> = view.eval_delta_distribution.to_vec();
    let eval_delta_sparkline = Sparkline::default()
        .data(&eval_delta_data)
        .style(Style::default().fg(Color::Magenta));
    frame.render_widget(eval_delta_sparkline, duplicate_rows[2]);
}

fn run_tui(
    stats: &Stats,
    cpuct_quantized: &AtomicU32,
    opening_randomness_quantized: &AtomicU32,
    stop_signal: &Arc<AtomicBool>,
    threads: u16,
    max_positions: u64,
) -> io::Result<()> {
    let stdout = io::stdout();
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::with_options(
        backend,
        TerminalOptions {
            viewport: Viewport::Inline(tui_total_height(threads)),
        },
    )?;

    let _guard = RawModeGuard::enable()?;

    let result = (|| -> io::Result<()> {
        let mut last_sample_time = Instant::now();

        loop {
            let view = stats.view(
                cpuct_quantized,
                opening_randomness_quantized,
                threads,
                max_positions,
            );
            terminal.draw(|f| render_tui(f, &view))?;

            if stats.active_threads.load(Ordering::Relaxed) == 0
                || stop_signal.load(Ordering::Relaxed)
            {
                break;
            }

            let now = Instant::now();
            let elapsed = now.duration_since(last_sample_time).as_secs();

            if elapsed >= SAMPLE_INTERVAL_SECS {
                last_sample_time = now;

                let current_positions = stats.positions.load(Ordering::Relaxed);
                let last_positions = stats.last_sample_positions.load(Ordering::Relaxed);

                if last_positions > 0 {
                    let positions_diff = current_positions.saturating_sub(last_positions);
                    let rate_per_hour = (positions_diff * 3600) / elapsed;

                    let _ = stats.recent_rates.push(rate_per_hour);

                    while stats.recent_rates.len() > MAX_SAMPLES {
                        let _ = stats.recent_rates.pop();
                    }
                }

                stats
                    .last_sample_positions
                    .store(current_positions, Ordering::Relaxed);
            }

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

    result
}

/// Adjusts `CPuct` based on average playouts (primary) and disagreement (secondary when playouts stable).
fn adjust_cpuct(avg_playouts: f32, cpuct_quantized: &AtomicU32) {
    // Adjust based on playouts
    if avg_playouts > TARGET_PLAYOUTS + PLAYOUTS_TOLERANCE {
        cpuct_quantized.fetch_sub(CPUCT_ADJUSTMENT_STEP, Ordering::Relaxed);
    } else if avg_playouts < TARGET_PLAYOUTS - PLAYOUTS_TOLERANCE {
        cpuct_quantized.fetch_add(CPUCT_ADJUSTMENT_STEP, Ordering::Relaxed);
    }
}

/// Adjusts opening randomness based on opening position balance.
fn adjust_opening_randomness(
    avg_opening_unbalanced: f32,
    opening_randomness_quantized: &AtomicU32,
) {
    if avg_opening_unbalanced > TARGET_OPENING_UNBALANCED + UNBALANCED_TOLERANCE {
        opening_randomness_quantized
            .fetch_sub(OPENING_RANDOMNESS_ADJUSTMENT_STEP, Ordering::Relaxed);
    } else if avg_opening_unbalanced < TARGET_OPENING_UNBALANCED - UNBALANCED_TOLERANCE {
        opening_randomness_quantized
            .fetch_add(OPENING_RANDOMNESS_ADJUSTMENT_STEP, Ordering::Relaxed);
    }
}

fn adjust_parameters(
    stats: &Stats,
    cpuct_quantized: &AtomicU32,
    opening_randomness_quantized: &AtomicU32,
) {
    let window_positions = stats.window_positions.load(Ordering::Relaxed);
    if window_positions < ADJUSTMENT_WINDOW_SIZE {
        return;
    }

    // Try to claim this window for adjustment
    if stats
        .window_positions
        .compare_exchange(window_positions, 0, Ordering::AcqRel, Ordering::Relaxed)
        .is_err()
    {
        return;
    }

    // Swap out accumulated values
    let window_playouts = stats.window_playouts.swap(0, Ordering::Relaxed);
    let window_opening_eval = stats.window_opening_eval.swap(0, Ordering::Relaxed);
    let window_opening_count = stats.window_opening_count.swap(0, Ordering::Relaxed);

    // Calculate averages
    let avg_playouts = window_playouts as f32 / window_positions as f32;
    let avg_opening_unbalanced = if window_opening_count > 0 {
        window_opening_eval as f32 / window_opening_count as f32 / SCALE
    } else {
        0.0
    };

    // Adjust parameters
    adjust_opening_randomness(avg_opening_unbalanced, opening_randomness_quantized);
    adjust_cpuct(avg_playouts, cpuct_quantized);
}

fn main() {
    let mut args = Args::from_env();
    let threads = args
        .flag("-t", "--threads")
        .unwrap_or_else(system::default_thread_count)
        .min(MAX_THREADS);
    let max_positions = args.flag("-p", "--positions").unwrap_or(DEFAULT_POSITIONS);

    assert!(
        threads > 0,
        "Thread count must be at least 1, got {threads}"
    );
    assert!(
        max_positions <= MAX_POSITIONS_PER_FILE * u64::from(threads),
        "Requested {max_positions} positions but maximum is {} ({}M positions/file * {} threads)",
        MAX_POSITIONS_PER_FILE * u64::from(threads),
        MAX_POSITIONS_PER_FILE / 1_000_000,
        threads
    );

    let stats = Arc::new(Stats::zero());
    let stop_signal = Arc::new(AtomicBool::new(false));

    // Calculate initial CPuct based on matching effective exploration at 500 visits
    let default_options = MctsOptions::default();
    let visits_at_target = TARGET_PLAYOUTS + 1.0;
    let effective_cpuct_at_target =
        default_options.cpuct * visits_at_target.powf(default_options.cpuct_tau);
    let data_gen_tau = 0.5;
    let initial_cpuct = effective_cpuct_at_target / visits_at_target.powf(data_gen_tau);

    let cpuct_quantized = Arc::new(AtomicU32::new((initial_cpuct * 100.0) as u32));
    let opening_randomness_quantized = Arc::new(AtomicU32::new(0)); // Initial opening randomness 0.00

    let timestamp = Utc::now().format("%Y%m%d-%H%M").to_string();

    thread::scope(|s| {
        let tui_stats = stats.clone();
        let tui_stop = stop_signal.clone();
        let tui_cpuct = cpuct_quantized.clone();
        let tui_opening_randomness = opening_randomness_quantized.clone();
        s.spawn(move || {
            if let Err(e) = run_tui(
                &tui_stats,
                &tui_cpuct,
                &tui_opening_randomness,
                &tui_stop,
                threads,
                max_positions,
            ) {
                eprintln!("TUI failed: {e}");
                tui_stop.store(true, Ordering::Relaxed);
            }
        });

        stats
            .active_threads
            .store(threads as usize, Ordering::Relaxed);

        for t in 0..threads {
            thread::sleep(Duration::from_millis(u64::from(10 * t)));

            let mut writer = BufWriter::new(
                File::create(format!("data/princhess-{timestamp}-{t}.data")).unwrap(),
            );

            let stats = stats.clone();
            let stop = stop_signal.clone();
            let cpuct = cpuct_quantized.clone();
            let opening_randomness = opening_randomness_quantized.clone();
            let mut rng = Rng::default();

            s.spawn(move || {
                let mut positions = Vec::new();

                let mut buffer: Box<[TrainingPosition; TrainingPosition::BUFFER_COUNT]> =
                    allocation::zeroed_box();

                while stats.positions.load(Ordering::Relaxed) < max_positions
                    && !stop.load(Ordering::Relaxed)
                {
                    while positions.len() < TrainingPosition::BUFFER_COUNT
                        && !stop.load(Ordering::Relaxed)
                    {
                        run_game(
                            &stats,
                            &cpuct,
                            &opening_randomness,
                            &mut positions,
                            &mut rng,
                            &stop,
                        );
                        stats.thread_buffers[t as usize].store(positions.len(), Ordering::Relaxed);
                    }

                    if positions.len() >= TrainingPosition::BUFFER_COUNT {
                        buffer.copy_from_slice(
                            positions.drain(..TrainingPosition::BUFFER_COUNT).as_slice(),
                        );

                        TrainingPosition::write_buffer(&mut writer, &buffer[..]).unwrap();
                        stats.thread_buffers[t as usize].store(positions.len(), Ordering::Relaxed);
                    }
                }

                stats.active_threads.fetch_sub(1, Ordering::Relaxed);
            });
        }
    });
}

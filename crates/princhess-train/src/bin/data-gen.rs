use bytemuck::allocation;
use crossterm::{
    cursor,
    event::{poll, read, Event, KeyCode},
    terminal::{disable_raw_mode, enable_raw_mode},
    ExecutableCommand,
};
use princhess::engine::{Engine, SCALE};
use princhess::math::{self, Rng};
use princhess::options::{EngineOptions, MctsOptions};
use princhess::state::State;
use ratatui::{
    backend::CrosstermBackend,
    layout::{Alignment, Constraint, Direction, Layout},
    style::{Color, Style},
    text::Span,
    widgets::{Block, Borders, Gauge, Paragraph, Sparkline},
    Frame, Terminal, TerminalOptions, Viewport,
};
use std::array;
use std::collections::HashSet;
use std::fs::File;
use std::io::{self, BufWriter};
use std::ops::Neg;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use princhess_train::data::TrainingPosition;
use scc::{Guard, Queue};

const HASH_SIZE_MB: usize = 128;

const MAX_PLAYOUTS_PER_POSITION: u64 = 10000;
const KL_DIVERGENCE_THRESHOLD: f64 = 0.000002;
const THREADS: u64 = 5;
const DFRC_PCT: u64 = 10;

const VARIATION_MIN_EVAL: f32 = 0.25;
const VARIATION_MAX_EVAL: f32 = 0.75;
const VARIATION_MIN_PHASE: usize = 8;

const MAX_POSITIONS_PER_FILE: u64 = 20_000_000;
const MAX_POSITIONS_TOTAL: u64 = MAX_POSITIONS_PER_FILE * THREADS;
const MAX_VARIATIONS: usize = 16;

const SAMPLE_INTERVAL_SECS: u64 = 5;
const RATE_WINDOW_SAMPLES: usize = 60;
const MAX_SAMPLES: usize = 512;

const POLICY_GINI_THRESHOLDS: [f32; 5] = [0.3, 0.5, 0.7, 0.8, 0.9];
const EVAL_DISTRIBUTION_THRESHOLDS: [f32; 5] = [-0.6, -0.3, 0.0, 0.3, 0.6];
const EVAL_DELTA_THRESHOLDS: [f32; 5] = [0.1, 0.2, 0.3, 0.4, 0.5];
const EVAL_RESULT_AGREEMENT_THRESHOLDS: [f32; 5] = [0.2, 0.4, 0.6, 1.0, 1.5];

const TUI_PROGRESS_BOX_HEIGHT: u16 = 3 + THREADS as u16 + 2;
const TUI_GRID_HEIGHT: u16 = 9;
const TUI_HISTOGRAM_HEIGHT: u16 = 8;
const TUI_TOTAL_HEIGHT: u16 = TUI_PROGRESS_BOX_HEIGHT + TUI_GRID_HEIGHT + TUI_HISTOGRAM_HEIGHT + 2;

fn load_atomic_array<const N: usize>(arr: &[AtomicU64; N]) -> [u64; N] {
    array::from_fn(|i| arr[i].load(Ordering::Relaxed))
}

fn to_bucket(value: f32, thresholds: &[f32]) -> usize {
    thresholds
        .iter()
        .position(|&t| value < t)
        .unwrap_or(thresholds.len())
}

fn bucket_labels(thresholds: &[f32; 5]) -> [String; 6] {
    [
        format!("    -{:>4.1}", thresholds[0]),
        format!("{:>4.1}-{:>4.1}", thresholds[0], thresholds[1]),
        format!("{:>4.1}-{:>4.1}", thresholds[1], thresholds[2]),
        format!("{:>4.1}-{:>4.1}", thresholds[2], thresholds[3]),
        format!("{:>4.1}-{:>4.1}", thresholds[3], thresholds[4]),
        format!("{:>4.1}-    ", thresholds[4]),
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
    thread_buffers: [AtomicUsize; THREADS as usize],
    policy_gini_buckets: [AtomicU64; 6],
    eval_distribution_buckets: [AtomicU64; 6],
    eval_delta_buckets: [AtomicU64; 6],
    eval_result_agreement_buckets: [AtomicU64; 6],
    piece_count_distribution: [AtomicU64; 33],
    phase_distribution: [AtomicU64; 25],
    variation_phase_distribution: [AtomicU64; 25],
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
            thread_buffers: Default::default(),
            policy_gini_buckets: Default::default(),
            eval_distribution_buckets: Default::default(),
            eval_delta_buckets: Default::default(),
            eval_result_agreement_buckets: Default::default(),
            piece_count_distribution: array::from_fn(|_| AtomicU64::new(0)),
            phase_distribution: Default::default(),
            variation_phase_distribution: Default::default(),
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
    }

    /// Tracks variation position balance.
    /// Variation = alternative move branches explored during play
    pub fn add_variation_eval(&self, eval: i64) {
        let abs_eval = eval.abs();
        self.variation_eval_sum
            .fetch_add(abs_eval, Ordering::Relaxed);
        self.variation_count.fetch_add(1, Ordering::Relaxed);
    }

    fn view(&self) -> StatsView {
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
            thread_buffers: array::from_fn(|i| self.thread_buffers[i].load(Ordering::Relaxed)),
            policy_gini_buckets: load_atomic_array(&self.policy_gini_buckets),
            eval_distribution_buckets: load_atomic_array(&self.eval_distribution_buckets),
            eval_delta_buckets: load_atomic_array(&self.eval_delta_buckets),
            eval_result_agreement_buckets: load_atomic_array(&self.eval_result_agreement_buckets),
            piece_count_distribution: load_atomic_array(&self.piece_count_distribution),
            phase_distribution: load_atomic_array(&self.phase_distribution),
            variation_phase_distribution: load_atomic_array(&self.variation_phase_distribution),
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
    thread_buffers: [usize; THREADS as usize],
    policy_gini_buckets: [u64; 6],
    eval_distribution_buckets: [u64; 6],
    eval_delta_buckets: [u64; 6],
    eval_result_agreement_buckets: [u64; 6],
    piece_count_distribution: [u64; 33],
    phase_distribution: [u64; 25],
    variation_phase_distribution: [u64; 25],
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

    fn white_win_pct(&self) -> u64 {
        (self.per_game(self.white_wins as f32) * 100.0) as u64
    }

    fn draw_pct(&self) -> u64 {
        (self.per_game(self.draws as f32) * 100.0) as u64
    }

    fn black_win_pct(&self) -> u64 {
        (self.per_game(self.black_wins as f32) * 100.0) as u64
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
        if !self.recent_rates.is_empty() {
            let window = self.recent_rates.len().min(RATE_WINDOW_SAMPLES);
            let sum: u64 = self.recent_rates.iter().rev().take(window).sum();
            sum / window as u64
        } else {
            self.positions * 3600 / self.elapsed_seconds
        }
    }

    fn positions_per_hour_millions(&self) -> f32 {
        self.avg_positions_per_hour() as f32 / 1_000_000.0
    }

    fn elapsed_formatted(&self) -> String {
        let hours = self.elapsed_seconds / 3600;
        let minutes = (self.elapsed_seconds % 3600) / 60;
        let secs = self.elapsed_seconds % 60;
        format!("{:02}:{:02}:{:02}", hours, minutes, secs)
    }

    fn eta_formatted(&self, max_positions: u64) -> String {
        if self.positions == 0 {
            return "--:--:--".to_string();
        }
        let positions_remaining = max_positions.saturating_sub(self.positions);

        let positions_per_second = self.avg_positions_per_hour() / 3600;

        if positions_per_second == 0 {
            return "--:--:--".to_string();
        }
        let eta_seconds = positions_remaining / positions_per_second;
        let hours = eta_seconds / 3600;
        let minutes = (eta_seconds % 3600) / 60;
        let secs = eta_seconds % 60;
        format!("{:02}:{:02}:{:02}", hours, minutes, secs)
    }

    fn progress_ratio(&self, max_positions: u64) -> f64 {
        self.positions as f64 / max_positions as f64
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

fn run_game(
    stats: &Stats,
    positions: &mut Vec<TrainingPosition>,
    rng: &mut Rng,
    stop_signal: &AtomicBool,
) {
    let mut variations = Vec::with_capacity(MAX_VARIATIONS + 1);
    let mut variations_count = 0;
    let mut seen_positions = HashSet::new();

    variations.push(random_start(rng));

    let mcts_options = MctsOptions {
        cpuct: 2.82,
        cpuct_tau: 0.5,
        policy_temperature: 1.0,
        policy_temperature_root: 1.4,
        cpuct_trend_adjustment: 0.0,
        cpuct_gini_base: 1.0,
        cpuct_gini_factor: 0.0,
        cpuct_gini_max: 1.0,
    };

    let engine_options = EngineOptions {
        mcts_options,
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

                    let current_visits: Vec<u32> = engine
                        .root_edges()
                        .iter()
                        .map(|edge| edge.visits())
                        .collect();

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
                let position = TrainingPosition::from(engine.mcts());
                let eval = position.evaluation();

                if variations_count < MAX_VARIATIONS {
                    let variation = engine.most_visited_move();

                    if variation != best_move
                        && (VARIATION_MIN_EVAL..=VARIATION_MAX_EVAL).contains(&eval.abs())
                        && state.phase() >= VARIATION_MIN_PHASE
                    {
                        let explore_probability = 1.0 / (variations_count + 1) as f32;
                        if rng.next_f32() < explore_probability {
                            game_stats.variations += 1;
                            let mut state = state.clone();
                            state.make_move(variation);
                            stats.variation_phase_distribution[state.phase()]
                                .fetch_add(1, Ordering::Relaxed);
                            variations.push(state);
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
                    let delta_bucket = to_bucket(delta.abs(), &EVAL_DELTA_THRESHOLDS);
                    stats.eval_delta_buckets[delta_bucket].fetch_add(1, Ordering::Relaxed);
                }
                previous_eval = eval;

                game_positions.push((position, state.board().occupied().count(), state.phase()));

                game_stats.positions += 1;
                game_stats.nodes += engine.mcts().num_nodes();
                game_stats.playouts += engine.mcts().playouts();
                game_stats.depth += engine.mcts().depth();
                game_stats.seldepth += engine.mcts().max_depth();
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

            if !seen_positions.insert(state.hash()) {
                game_positions.clear();
                break GameResult::Aborted;
            }
        };

        let mut blunder = false;

        for (position, piece_count, phase) in game_positions.iter_mut() {
            position.set_result(i8::from(result));

            let eval = position.evaluation();
            stats.eval_result_agreement_buckets[to_bucket(
                (eval - f32::from(i8::from(result))).abs(),
                &EVAL_RESULT_AGREEMENT_THRESHOLDS,
            )]
            .fetch_add(1, Ordering::Relaxed);

            // Track data distributions for saved positions only
            let moves = position.moves();
            let visits: Vec<u8> = moves.iter().map(|(_, v)| *v).collect();
            let gini = compute_policy_gini(&visits);
            let gini_bucket = to_bucket(gini, &POLICY_GINI_THRESHOLDS);
            stats.policy_gini_buckets[gini_bucket].fetch_add(1, Ordering::Relaxed);

            let eval_bucket = to_bucket(eval, &EVAL_DISTRIBUTION_THRESHOLDS);
            stats.eval_distribution_buckets[eval_bucket].fetch_add(1, Ordering::Relaxed);

            stats.piece_count_distribution[*piece_count].fetch_add(1, Ordering::Relaxed);
            stats.phase_distribution[*phase].fetch_add(1, Ordering::Relaxed);

            blunder |= match result {
                GameResult::WhiteWin => eval < -0.5,
                GameResult::BlackWin => eval > 0.5,
                GameResult::Draw => eval.abs() > 0.75,
                GameResult::Aborted => false,
            }
        }

        let saved_positions: Vec<TrainingPosition> =
            game_positions.into_iter().map(|(pos, _, _)| pos).collect();
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
    }
}

fn random_start(rng: &mut Rng) -> State {
    let (_moves, state) = princhess::state::generate_random_opening(rng, DFRC_PCT);
    state
}

/// Computes KL divergence per additional playout between old and new visit distributions.
///
/// Uses Laplace smoothing (ε=1.0) to handle zero visits:
/// - p = (old_visits + ε) / (old_total + num_moves * ε)
/// - q = (new_visits + ε) / (new_total + num_moves * ε)
/// - KL(p||q) = Σ p * ln(p/q)
///
/// Returns KL divergence divided by the visit difference to normalize gain per playout.
/// Returns None if old_visits is zero or new_visits ≤ old_visits.
fn kld_gain(new_visits: &[u32], old_visits: &[u32]) -> Option<f64> {
    let new_parent_visits: u64 = new_visits.iter().map(|&x| u64::from(x)).sum();
    let old_parent_visits: u64 = old_visits.iter().map(|&x| u64::from(x)).sum();

    if old_parent_visits == 0 || new_parent_visits <= old_parent_visits {
        return None;
    }

    let num_moves = new_visits.len() as f64;
    let epsilon = 1.0;

    let new_total = new_parent_visits as f64 + num_moves * epsilon;
    let old_total = old_parent_visits as f64 + num_moves * epsilon;

    let mut gain = 0.0;

    for (&new_v, &old_v) in new_visits.iter().zip(old_visits.iter()) {
        let q = (f64::from(new_v) + epsilon) / new_total;
        let p = (f64::from(old_v) + epsilon) / old_total;

        gain += p * (p / q).ln();
    }

    let parent_visits_diff = new_parent_visits.saturating_sub(old_parent_visits) as f64;
    if parent_visits_diff > 0.0 {
        Some(gain / parent_visits_diff)
    } else {
        None
    }
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
    buckets: &[u64; 6],
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

    let gauge_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1); 6])
        .split(inner);

    for (i, (&count, label)) in buckets.iter().zip(labels.iter()).enumerate() {
        let pct = (count as f32 * 100.0 / total as f32) as u64;

        let row_layout = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Length(10), Constraint::Min(0)])
            .split(gauge_layout[i]);

        let label_widget = Paragraph::new(label.as_str()).style(Style::default().fg(Color::Gray));
        frame.render_widget(label_widget, row_layout[0]);

        let gauge_label = Span::styled(format!("{:>3}%", pct), Style::default().fg(Color::White));
        let gauge = Gauge::default()
            .gauge_style(Style::default().fg(Color::Cyan))
            .ratio(pct as f64 / 100.0)
            .label(gauge_label);
        frame.render_widget(gauge, row_layout[1]);
    }
}

fn render_tui(frame: &mut Frame, view: &StatsView) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .margin(1)
        .constraints([
            Constraint::Length(3 + THREADS as u16 + 2),
            Constraint::Length(9),
            Constraint::Length(8),
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
            Constraint::Length(1),              // Time/rate line
            Constraint::Length(1),              // Progress gauge
            Constraint::Length(1),              // Sparkline
            Constraint::Length(THREADS as u16), // Thread gauges
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

    let elapsed = Paragraph::new(view.elapsed_formatted()).alignment(Alignment::Left);
    frame.render_widget(elapsed, time_chunks[0]);

    let rate = Paragraph::new(format!("{:.1}M/h", view.positions_per_hour_millions()))
        .alignment(Alignment::Center);
    frame.render_widget(rate, time_chunks[1]);

    let eta = Paragraph::new(view.eta_formatted(MAX_POSITIONS_TOTAL)).alignment(Alignment::Right);
    frame.render_widget(eta, time_chunks[2]);

    // Progress gauge
    let progress_ratio = view.progress_ratio(MAX_POSITIONS_TOTAL).min(1.0);
    let label = Span::styled(
        format!(
            "{:.1}M / {:.1}M ({:.1}%)",
            view.positions_millions(),
            MAX_POSITIONS_TOTAL as f32 / 1_000_000.0,
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
    for (i, &buffer_size) in view.thread_buffers.iter().enumerate() {
        let ratio = (buffer_size as f64 / TrainingPosition::BUFFER_COUNT as f64).min(1.0);

        let line_gauge = Gauge::default()
            .gauge_style(Style::default().fg(Color::Blue))
            .ratio(ratio)
            .label(Span::styled(
                format!("T{}", i),
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
    let white_constraint = white_pct as u16;
    let black_constraint = black_pct as u16;
    let draw_constraint = 100u16.saturating_sub(white_constraint + black_constraint);

    let bar_layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(white_constraint),
            Constraint::Percentage(draw_constraint),
            Constraint::Percentage(black_constraint),
        ])
        .split(bar_area);

    if white_pct > 0 {
        let label = Span::styled(format!("{}%", white_pct), Style::default().fg(Color::Black));
        let white_bar = Gauge::default()
            .gauge_style(Style::default().fg(Color::White).bg(Color::White))
            .percent(100)
            .label(label);
        frame.render_widget(white_bar, bar_layout[0]);
    }

    if draw_pct > 0 {
        let label = Span::styled(format!("{}%", draw_pct), Style::default().fg(Color::Black));
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
        let label = Span::styled(format!("{}%", black_pct), Style::default().fg(Color::White));
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
            "Nodes:   {:>5}\nDepth:   {:>5}",
            view.avg_nodes(),
            view.avg_depth(),
        ),
        format!(
            "Playouts: {:>5}\nSeldepth: {:>5}",
            view.avg_playouts(),
            view.avg_seldepth(),
        ),
    );

    render_two_column_box(
        frame,
        bottom_row[0],
        "Quality Metrics",
        format!(
            "Blunders: {:>5.1}%\nSkipped:  {:>5.1}%\nOpening:   {:>+5.2}",
            view.blunder_pct(),
            view.skipped_pct(),
            view.avg_opening_eval(),
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
        "Policy Gini",
        &view.policy_gini_buckets,
        &bucket_labels(&POLICY_GINI_THRESHOLDS),
    );
    render_histogram(
        frame,
        histogram_row[1],
        "Eval Distribution",
        &view.eval_distribution_buckets,
        &bucket_labels(&EVAL_DISTRIBUTION_THRESHOLDS),
    );
    render_histogram(
        frame,
        histogram_row[2],
        "Eval Delta",
        &view.eval_delta_buckets,
        &bucket_labels(&EVAL_DELTA_THRESHOLDS),
    );
    render_histogram(
        frame,
        histogram_row[3],
        "Eval-Result Agreement",
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

    let sparkline_layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(13), Constraint::Min(0)])
        .split(distribution_inner);

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
}

fn run_tui(stats: &Stats, stop_signal: Arc<AtomicBool>) -> io::Result<()> {
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
            let view = stats.view();
            terminal.draw(|f| render_tui(f, &view))?;

            if view.positions >= MAX_POSITIONS_TOTAL || stop_signal.load(Ordering::Relaxed) {
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
    disable_raw_mode()?;

    result
}

fn main() {
    let stats = Arc::new(Stats::zero());
    let stop_signal = Arc::new(AtomicBool::new(false));

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    thread::scope(|s| {
        let tui_stats = stats.clone();
        let tui_stop = stop_signal.clone();
        s.spawn(move || {
            if let Err(e) = run_tui(&tui_stats, tui_stop.clone()) {
                eprintln!("TUI failed: {e}");
                tui_stop.store(true, Ordering::Relaxed);
            }
        });

        for t in 0..THREADS {
            thread::sleep(Duration::from_millis(10 * t));

            let mut writer = BufWriter::new(
                File::create(format!("data/princhess-{timestamp}-{t}.data")).unwrap(),
            );

            let stats = stats.clone();
            let stop = stop_signal.clone();
            let mut rng = Rng::default();

            s.spawn(move || {
                let mut positions = Vec::new();

                let mut buffer: Box<[TrainingPosition; TrainingPosition::BUFFER_COUNT]> =
                    allocation::zeroed_box();

                while stats.positions.load(Ordering::Relaxed) < MAX_POSITIONS_TOTAL
                    && !stop.load(Ordering::Relaxed)
                {
                    while positions.len() < TrainingPosition::BUFFER_COUNT
                        && !stop.load(Ordering::Relaxed)
                    {
                        run_game(&stats, &mut positions, &mut rng, &stop);
                        stats.thread_buffers[t as usize].store(positions.len(), Ordering::Relaxed);
                    }

                    if positions.len() >= TrainingPosition::BUFFER_COUNT {
                        buffer.copy_from_slice(
                            positions.drain(..TrainingPosition::BUFFER_COUNT).as_slice(),
                        );

                        TrainingPosition::write_buffer(&mut writer, &buffer[..]);
                        stats.thread_buffers[t as usize].store(positions.len(), Ordering::Relaxed);
                    }
                }
            });
        }
    });
}

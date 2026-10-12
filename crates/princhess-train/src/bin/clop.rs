#![warn(clippy::all, clippy::pedantic)]
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss, clippy::cast_possible_wrap)]

use std::f64::consts::{LN_10, TAU};
use std::fs::{self, File};
use std::iter;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::symbols::Marker;
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Axis, Block, Borders, Chart, Dataset, Gauge, GraphType, Padding, Paragraph,
};
use ratatui::Frame;

use princhess::math::Rng;
use princhess_train::args::Args;
use princhess_train::system;
use princhess_train::tui;

const DEFAULT_GAMES: u64 = 15_000;
const DEFAULT_TC: &str = "8+0.08";
const DEFAULT_BOX_PCT: f64 = 50.0;
// Each candidate plays one game pair against the reference, so a batch is 8 games per
// concurrent slot, which keeps end-of-batch idle time to a few percent
const CANDIDATES_PER_SLOT: u64 = 4;
// Box half-widths below this are mostly lost to integer rounding of UCI spin values
const MIN_HALF_WIDTH: f64 = 2.0;
// Sample the box uniformly for this share of the games before focusing on the fitted optimum
const UNIFORM_FRACTION: f64 = 0.1;

// Gaussian prior precision on each coefficient, in logits. Weak, but keeps early fits finite
const PRIOR_PRECISION: f64 = 1.0;
const NEWTON_ITERATIONS: usize = 50;
const NEWTON_TOLERANCE: f64 = 1e-9;
const ELO_PER_LOGIT: f64 = 400. / LN_10;

const ARGMAX_RANDOM_POINTS: usize = 256;
const ARGMAX_ASCENT_STEPS: usize = 100;
const ESTIMATE_DRAWS: usize = 200;

const DEFAULT_ENGINE: &str = "target/release/princhess";
const BOOK: &str = "books/UHO_Lichess_4852_v1.epd";
const SYZYGY: &str = "syzygy";
const HASH_MB: u32 = 128;
const WORK_DIR: &str = "target/clop";
const REFERENCE: &str = "ref";

const POLL_INTERVAL: Duration = Duration::from_millis(500);
const FASTCHESS_STOP_TIMEOUT: Duration = Duration::from_secs(10);
const CHART_HEIGHT: u16 = 20;
const PARAM_COLORS: [Color; 8] = [
    Color::Cyan,
    Color::Yellow,
    Color::Magenta,
    Color::Green,
    Color::Red,
    Color::Blue,
    Color::LightCyan,
    Color::LightYellow,
];

#[derive(Clone)]
struct Param {
    name: String,
    start: f64,
    half_width: f64,
    // (games played, estimated optimum) after each batch
    history: Vec<(f64, f64)>,
}

impl Param {
    fn value(&self, u: f64) -> f64 {
        self.start + u * self.half_width
    }
}

/// A candidate's game pair, with its parameters normalised to [-1, 1] across the box and its mean
/// score against the reference. Pairs are kept whole because with unbalanced openings the two
/// games are strongly anti-correlated, so a pair's score varies much less than two games would.
#[derive(Clone)]
struct Pair {
    u: Vec<f64>,
    score: f64,
}

/// Logistic quadratic model of the candidate's score against the reference:
/// `P(score) = sigmoid(q(u))`. There is no constant term, since at the start values the candidate
/// is the reference and scores exactly 50%. In logits, so Elo is `q * ELO_PER_LOGIT`.
#[derive(Clone)]
struct Model {
    params: usize,
    // Linear terms, then quadratic terms u_i * u_j for i <= j
    beta: Vec<f64>,
    // Cholesky factor of the posterior precision (the negated Hessian of the log posterior,
    // scaled by the dispersion)
    chol: Vec<Vec<f64>>,
}

impl Model {
    fn coefficients(params: usize) -> usize {
        params + params * (params + 1) / 2
    }

    fn features(u: &[f64]) -> Vec<f64> {
        let mut phi = u.to_vec();
        for i in 0..u.len() {
            for j in i..u.len() {
                phi.push(u[i] * u[j]);
            }
        }
        phi
    }

    /// Maximum a posteriori fit by Newton's method.
    fn fit(params: usize, pairs: &[Pair], warm: Option<&Model>) -> Self {
        let m = Self::coefficients(params);
        let phis: Vec<Vec<f64>> = pairs.iter().map(|g| Self::features(&g.u)).collect();
        let mut beta = warm.map_or_else(|| vec![0.; m], |w| w.beta.clone());
        let mut chol = Vec::new();

        for _ in 0..NEWTON_ITERATIONS {
            let mut grad: Vec<f64> = beta.iter().map(|b| -PRIOR_PRECISION * b).collect();
            let mut hess = vec![vec![0.; m]; m];
            for (i, row) in hess.iter_mut().enumerate() {
                row[i] = PRIOR_PRECISION;
            }

            // Each pair counts as two games
            for (phi, pair) in phis.iter().zip(pairs) {
                let p = sigmoid(dot(&beta, phi));
                let w = 2. * p * (1. - p);
                for i in 0..m {
                    grad[i] += 2. * (pair.score - p) * phi[i];
                    for j in 0..=i {
                        hess[i][j] += w * phi[i] * phi[j];
                    }
                }
            }

            chol = cholesky(&hess);
            let step = backward(&chol, &forward(&chol, &grad));
            for (b, s) in beta.iter_mut().zip(&step) {
                *b += s;
            }
            if step.iter().all(|s| s.abs() < NEWTON_TOLERANCE) {
                break;
            }
        }

        // The logistic variance assumes independent wins and losses, but draws and the
        // anti-correlation within pairs make pair scores less variable. Scale the posterior by
        // the observed dispersion (quasi-likelihood) so its uncertainty matches the actual spread.
        let dof = pairs.len().saturating_sub(m);
        if dof > 0 {
            let pearson: f64 = phis
                .iter()
                .zip(pairs)
                .map(|(phi, pair)| {
                    let p = sigmoid(dot(&beta, phi));
                    2. * (pair.score - p).powi(2) / (p * (1. - p))
                })
                .sum();
            let scale = (dof as f64 / pearson).sqrt();
            for x in chol.iter_mut().flatten() {
                *x *= scale;
            }
        }

        Self { params, beta, chol }
    }

    /// Coefficients drawn from the Laplace approximation of the posterior.
    fn draw(&self, rng: &mut Rng) -> Vec<f64> {
        let z: Vec<f64> = self.beta.iter().map(|_| normal(rng)).collect();
        let offset = backward(&self.chol, &z);
        self.beta.iter().zip(&offset).map(|(b, o)| b + o).collect()
    }

    /// Standard deviation of `q(u)` under the posterior, in logits.
    fn sd(&self, u: &[f64]) -> f64 {
        let y = forward(&self.chol, &Self::features(u));
        dot(&y, &y).sqrt()
    }

    fn q(beta: &[f64], u: &[f64]) -> f64 {
        dot(beta, &Self::features(u))
    }

    /// The coefficient of `u_i * u_j`.
    fn quadratic(&self, i: usize, j: usize) -> f64 {
        let (i, j) = (i.min(j), i.max(j));
        // Quadratic terms of rows before i, then the offset within row i
        let row_start = i * self.params - i * (i.saturating_sub(1)) / 2;
        self.beta[self.params + row_start + j - i]
    }

    /// Maximises `q` over the box: best of random points, refined by projected gradient ascent.
    fn argmax(params: usize, beta: &[f64], rng: &mut Rng) -> Vec<f64> {
        let mut best = vec![0.; params];
        let mut best_q = 0.;
        for _ in 0..ARGMAX_RANDOM_POINTS {
            let u: Vec<f64> = (0..params).map(|_| uniform(rng) * 2. - 1.).collect();
            let q = Self::q(beta, &u);
            if q > best_q {
                best = u;
                best_q = q;
            }
        }

        // The gradient's Lipschitz constant is at most twice the norm of the quadratic terms,
        // so its inverse is a step size that never overshoots
        let lipschitz = 2. * beta[params..].iter().map(|b| b * b).sum::<f64>().sqrt();
        let step = 1. / lipschitz.max(f64::MIN_POSITIVE);

        let mut u = best;
        for _ in 0..ARGMAX_ASCENT_STEPS {
            let mut grad = beta[..params].to_vec();
            let mut k = params;
            for i in 0..params {
                for j in i..params {
                    grad[i] += beta[k] * u[j];
                    grad[j] += beta[k] * u[i];
                    k += 1;
                }
            }
            for (x, g) in u.iter_mut().zip(&grad) {
                *x = (*x + step * g).clamp(-1., 1.);
            }
        }
        u
    }
}

#[derive(Clone)]
struct Estimate {
    // Normalised optimum, and the spread of the optimum across posterior draws in UCI units
    best: Vec<f64>,
    spread: Vec<f64>,
    gain: f64,
    gain_sd: f64,
}

impl Estimate {
    fn new(model: &Model, params: &[Param], rng: &mut Rng) -> Self {
        let best = Model::argmax(model.params, &model.beta, rng);

        let draws: Vec<Vec<f64>> = (0..ESTIMATE_DRAWS)
            .map(|_| Model::argmax(model.params, &model.draw(rng), rng))
            .collect();
        let spread = params
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let mean = draws.iter().map(|d| d[i]).sum::<f64>() / ESTIMATE_DRAWS as f64;
                let var = draws.iter().map(|d| (d[i] - mean).powi(2)).sum::<f64>()
                    / ESTIMATE_DRAWS as f64;
                var.sqrt() * p.half_width
            })
            .collect();

        Self {
            gain: Model::q(&model.beta, &best) * ELO_PER_LOGIT,
            gain_sd: model.sd(&best) * ELO_PER_LOGIT,
            best,
            spread,
        }
    }
}

#[derive(Clone)]
struct Clop {
    engine: PathBuf,
    tc: String,
    concurrency: u64,
    total_batches: u64,
    batches_done: u64,
    params: Vec<Param>,
    pairs: Vec<Pair>,
    model: Option<Model>,
    estimate: Option<Estimate>,
}

impl Clop {
    fn from_args(args: &mut Args) -> Self {
        let games = args.flag("-g", "--games").unwrap_or(DEFAULT_GAMES);
        let tc = match args.flag::<u64>("-n", "--nodes") {
            Some(nodes) => format!("tc=inf nodes={nodes}"),
            None => format!(
                "tc={}",
                args.flag::<String>("--tc", "--tc")
                    .unwrap_or_else(|| DEFAULT_TC.to_string())
            ),
        };
        let concurrency = args
            .flag("-c", "--concurrency")
            .unwrap_or_else(|| u64::from(system::default_thread_count()));
        let box_pct = args
            .flag("--box-pct", "--box-pct")
            .unwrap_or(DEFAULT_BOX_PCT);
        let engine = args
            .flag::<String>("-e", "--engine")
            .unwrap_or_else(|| DEFAULT_ENGINE.to_string());

        let names = args.positional();
        assert!(!names.is_empty(), "No parameters given to tune");

        let defaults = uci_spin_defaults(Path::new(&engine));
        let params = names
            .into_iter()
            .map(|name| {
                let Some(&(_, start)) = defaults.iter().find(|(n, _)| *n == name) else {
                    panic!("Engine has no spin option named {name}");
                };
                Param {
                    half_width: (start.abs() * box_pct / 100.).max(MIN_HALF_WIDTH),
                    name,
                    start,
                    history: vec![(0., start)],
                }
            })
            .collect();

        fs::create_dir_all(WORK_DIR).unwrap();

        let batch_games = concurrency * CANDIDATES_PER_SLOT * 2;

        Self {
            engine: PathBuf::from(engine),
            tc,
            concurrency,
            total_batches: games.div_ceil(batch_games),
            batches_done: 0,
            params,
            pairs: Vec::new(),
            model: None,
            estimate: None,
        }
    }

    fn candidates(&self) -> u64 {
        self.concurrency * CANDIDATES_PER_SLOT
    }

    fn batch_games(&self) -> u64 {
        self.candidates() * 2
    }

    fn total_games(&self) -> u64 {
        self.total_batches * self.batch_games()
    }

    fn is_finished(&self) -> bool {
        self.batches_done >= self.total_batches
    }

    fn games_played(&self) -> u64 {
        self.pairs.len() as u64 * 2
    }

    fn is_uniform(&self) -> bool {
        (self.games_played() as f64) < UNIFORM_FRACTION * self.total_games() as f64
    }

    /// Picks a candidate point: uniform over the box at first, then the optimum of a quadratic
    /// drawn from the posterior, so sampling concentrates as the optimum becomes certain.
    fn candidate(&self, rng: &mut Rng) -> Vec<f64> {
        match &self.model {
            Some(model) if !self.is_uniform() => Model::argmax(model.params, &model.draw(rng), rng),
            _ => self.params.iter().map(|_| uniform(rng) * 2. - 1.).collect(),
        }
    }

    /// Plays one gauntlet of candidates against the reference and refits the model. Returns
    /// `Ok(false)` if interrupted, leaving the state unchanged.
    fn run_batch(
        &mut self,
        rng: &mut Rng,
        interrupt: &AtomicBool,
        progress: &BatchProgress,
    ) -> Result<bool, String> {
        // Values are rounded before normalising, so the model sees what was actually played
        let candidates: Vec<Vec<i64>> = (0..self.candidates())
            .map(|_| {
                self.candidate(rng)
                    .iter()
                    .zip(&self.params)
                    .map(|(&u, p)| stochastic_round(p.value(u), rng))
                    .collect()
            })
            .collect();

        progress.reset(self.batches_done, self.batch_games());
        let fastchess = Fastchess {
            clop: self,
            seed: rng.next_u32(),
        };
        let Some(results) = fastchess.run(&candidates, interrupt, progress)? else {
            return Ok(false);
        };

        let mut totals = vec![0.; candidates.len()];
        for (candidate, score) in results {
            totals[candidate] += score;
        }
        for (values, total) in candidates.iter().zip(totals) {
            let u = values
                .iter()
                .zip(&self.params)
                .map(|(&v, p)| (v as f64 - p.start) / p.half_width)
                .collect();
            self.pairs.push(Pair {
                u,
                score: total / 2.,
            });
        }

        let model = Model::fit(self.params.len(), &self.pairs, self.model.as_ref());
        let estimate = Estimate::new(&model, &self.params, rng);

        self.batches_done += 1;
        let games = self.games_played() as f64;
        for (p, &u) in self.params.iter_mut().zip(&estimate.best) {
            let value = p.value(u);
            p.history.push((games, value));
        }

        self.model = Some(model);
        self.estimate = Some(estimate);

        Ok(true)
    }
}

#[derive(Clone, Default)]
struct BatchState {
    batch: u64,
    started: u64,
    finished: u64,
    total: u64,
}

/// Live state of the running batch, read from the fastchess log.
#[derive(Default)]
struct BatchProgress {
    state: Mutex<BatchState>,
}

impl BatchProgress {
    fn reset(&self, batch: u64, total: u64) {
        *self.state.lock().unwrap() = BatchState {
            batch,
            total,
            ..BatchState::default()
        };
    }

    fn update(&self, log: &str) {
        let mut state = self.state.lock().unwrap();
        state.started = log.matches("Started game").count() as u64;
        state.finished = parse_finished_games(log).len() as u64;
    }

    fn get(&self) -> BatchState {
        self.state.lock().unwrap().clone()
    }
}

struct Fastchess<'a> {
    clop: &'a Clop,
    seed: u32,
}

impl Fastchess<'_> {
    fn args(&self, candidates: &[Vec<i64>]) -> Vec<String> {
        let engine = |name: String| {
            vec![
                "-engine".to_string(),
                format!("cmd={}", self.clop.engine.display()),
                format!("name={name}"),
            ]
        };

        // The reference runs at the engine's defaults, which are the start values
        let mut args = engine(REFERENCE.to_string());
        for (i, values) in candidates.iter().enumerate() {
            args.extend(engine(format!("c{i}")));
            args.extend(
                values
                    .iter()
                    .zip(&self.clop.params)
                    .map(|(v, p)| format!("option.{}={v}", p.name)),
            );
        }
        args.extend(["-each".into(), "proto=uci".into()]);
        args.extend(self.clop.tc.split_whitespace().map(String::from));
        args.extend([
            format!("option.Hash={HASH_MB}"),
            "option.Threads=1".into(),
            format!("option.SyzygyPath={SYZYGY}"),
            "-tournament".into(),
            "gauntlet".into(),
            "-seeds".into(),
            "1".into(),
            "-openings".into(),
            format!("file={BOOK}"),
            "format=epd".into(),
            "order=random".into(),
            "-srand".into(),
            self.seed.to_string(),
            "-games".into(),
            "2".into(),
            "-repeat".into(),
            "-rounds".into(),
            "1".into(),
            "-concurrency".into(),
            self.clop.concurrency.to_string(),
            "-config".into(),
            format!("outname={WORK_DIR}/fastchess.json"),
        ]);
        args
    }

    /// Runs the gauntlet, returning each game's candidate index and score, or `Ok(None)` if
    /// interrupted.
    fn run(
        &self,
        candidates: &[Vec<i64>],
        interrupt: &AtomicBool,
        progress: &BatchProgress,
    ) -> Result<Option<Vec<(usize, f64)>>, String> {
        let log_path = Path::new(WORK_DIR).join("fastchess.log");
        let log = File::create(&log_path).map_err(|e| e.to_string())?;
        let mut child = Command::new("fastchess")
            .args(self.args(candidates))
            .stdin(Stdio::null())
            .stdout(log.try_clone().map_err(|e| e.to_string())?)
            .stderr(log)
            .spawn()
            .map_err(|e| format!("Failed to start fastchess: {e}"))?;

        let exit = loop {
            if interrupt.load(Ordering::Relaxed) {
                stop_fastchess(&mut child);
                return Ok(None);
            }
            if let Some(exit) = child.try_wait().map_err(|e| e.to_string())? {
                break exit;
            }
            if let Ok(out) = fs::read_to_string(&log_path) {
                progress.update(&out);
            }
            thread::sleep(POLL_INTERVAL);
        };

        let out = fs::read_to_string(&log_path).map_err(|e| e.to_string())?;
        let fail = |msg: String| format!("{msg}, see {}", log_path.display());

        if !exit.success() {
            return Err(fail(format!("fastchess exited with {exit}")));
        }

        let results = parse_finished_games(&out);
        let expected = self.clop.batch_games();
        if results.len() as u64 != expected {
            return Err(fail(format!(
                "Expected {expected} games, got {}",
                results.len()
            )));
        }

        Ok(Some(results))
    }
}

fn main() {
    let mut args = Args::from_env();
    let mut clop = Clop::from_args(&mut args);

    assert!(
        Path::new(BOOK).exists(),
        "Opening book {BOOK} not found, run from the repository root"
    );

    let shared = Arc::new(Mutex::new(clop.clone()));
    let progress = Arc::new(BatchProgress::default());
    let interrupt = Arc::new(AtomicBool::new(false));
    let done = Arc::new(AtomicBool::new(false));

    let tui_thread = {
        let shared = Arc::clone(&shared);
        let progress = Arc::clone(&progress);
        let interrupt = Arc::clone(&interrupt);
        let done = Arc::clone(&done);
        thread::spawn(move || {
            let height = tui_height(shared.lock().unwrap().params.len());
            let mut rate = GameRate::new(Instant::now());
            if let Err(e) = tui::run_inline_tui(
                height,
                || done.load(Ordering::Relaxed) || interrupt.load(Ordering::Relaxed),
                || interrupt.store(true, Ordering::Relaxed),
                || {},
                |f| render(f, &shared.lock().unwrap(), &progress, &mut rate),
            ) {
                eprintln!("TUI failed: {e}");
                interrupt.store(true, Ordering::Relaxed);
            }
        })
    };

    let mut rng = Rng::default();
    let mut failure = None;

    while !clop.is_finished() && !interrupt.load(Ordering::Relaxed) {
        match clop.run_batch(&mut rng, &interrupt, &progress) {
            Ok(true) => *shared.lock().unwrap() = clop.clone(),
            Ok(false) => break,
            Err(e) => {
                failure = Some(e);
                break;
            }
        }
    }

    done.store(true, Ordering::Relaxed);
    tui_thread.join().unwrap();

    // Panic only once the TUI has restored the terminal, so the message is readable
    if let Some(e) = failure {
        panic!("{e}");
    }
}

fn uci_spin_defaults(engine: &Path) -> Vec<(String, f64)> {
    let out = Command::new(engine)
        .arg("uci")
        .output()
        .unwrap_or_else(|e| panic!("Failed to run {}: {e}", engine.display()));

    // option name <Name> type spin default <D> min <Min> max <Max>
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|line| {
            let tokens: Vec<&str> = line.split_whitespace().collect();
            match tokens[..] {
                ["option", "name", name, "type", "spin", "default", default, ..] => {
                    Some((name.to_string(), default.parse().ok()?))
                }
                _ => None,
            }
        })
        .collect()
}

/// Asks fastchess to stop with SIGINT so it shuts down its engines; SIGKILL would orphan them.
/// Falls back to SIGKILL if it hasn't exited in time.
fn stop_fastchess(child: &mut Child) {
    let interrupted = Command::new("kill")
        .args(["-INT", &child.id().to_string()])
        .status()
        .is_ok_and(|s| s.success());

    if interrupted {
        let deadline = Instant::now() + FASTCHESS_STOP_TIMEOUT;
        while Instant::now() < deadline {
            if let Ok(Some(_)) = child.try_wait() {
                return;
            }
            thread::sleep(POLL_INTERVAL);
        }
    }

    child.kill().ok();
    child.wait().ok();
}

/// Parses "Finished game N (white vs black): 1-0 {...}" lines into the candidate's index and its
/// score against the reference.
fn parse_finished_games(out: &str) -> Vec<(usize, f64)> {
    out.lines()
        .filter_map(|line| {
            let rest = line.strip_prefix("Finished game")?;
            let (_, rest) = rest.split_once('(')?;
            let (players, rest) = rest.split_once("): ")?;
            let (white, black) = players.split_once(" vs ")?;
            let white_score = match rest.split_whitespace().next()? {
                "1-0" => 1.,
                "0-1" => 0.,
                "1/2-1/2" => 0.5,
                _ => return None,
            };
            let (candidate, score) = if white == REFERENCE {
                (black, 1. - white_score)
            } else {
                (white, white_score)
            };
            Some((candidate.strip_prefix('c')?.parse().ok()?, score))
        })
        .collect()
}

/// Rounds to a neighbouring integer with probability proportional to proximity, so integer
/// UCI values are unbiased on average.
fn stochastic_round(x: f64, rng: &mut Rng) -> i64 {
    let floor = x.floor();
    floor as i64 + i64::from(uniform(rng) < x - floor)
}

fn uniform(rng: &mut Rng) -> f64 {
    (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64
}

/// Standard normal by Box-Muller.
fn normal(rng: &mut Rng) -> f64 {
    let r = (-2. * (1. - uniform(rng)).ln()).sqrt();
    r * (TAU * uniform(rng)).cos()
}

fn sigmoid(x: f64) -> f64 {
    1. / (1. + (-x).exp())
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Lower triangular `L` with `L * L^T = a`, reading only the lower triangle of `a`.
fn cholesky(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let mut l = vec![vec![0.; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let sum = a[i][j] - dot(&l[i][..j], &l[j][..j]);
            l[i][j] = if i == j { sum.sqrt() } else { sum / l[j][j] };
        }
    }
    l
}

/// Solves `L * x = b`.
fn forward(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let mut x = vec![0.; b.len()];
    for i in 0..b.len() {
        x[i] = (b[i] - dot(&l[i][..i], &x[..i])) / l[i][i];
    }
    x
}

/// Solves `L^T * x = b`.
fn backward(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut x = vec![0.; n];
    for i in (0..n).rev() {
        let sum: f64 = (i + 1..n).map(|j| l[j][i] * x[j]).sum();
        x[i] = (b[i] - sum) / l[i][i];
    }
    x
}

fn tui_height(params: usize) -> u16 {
    // Progress + params table (header, rows, borders) + chart + margins
    6 + params as u16 + 3 + CHART_HEIGHT + 2
}

/// Games/h and ETA, recomputed only when a game finishes so they don't drift in between.
struct GameRate {
    start: Instant,
    games: u64,
    games_per_hour: f64,
    eta_secs: u64,
}

impl GameRate {
    fn new(start: Instant) -> Self {
        Self {
            start,
            games: 0,
            games_per_hour: 0.,
            eta_secs: 0,
        }
    }

    fn update(&mut self, games: u64, games_total: u64) {
        if games == self.games {
            return;
        }
        self.games = games;
        self.games_per_hour = games as f64 * 3600. / self.start.elapsed().as_secs_f64();
        self.eta_secs =
            (games_total.saturating_sub(games) as f64 / self.games_per_hour * 3600.) as u64;
    }
}

fn render(frame: &mut Frame, clop: &Clop, progress: &BatchProgress, rate: &mut GameRate) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .margin(1)
        .constraints([
            Constraint::Length(6),
            Constraint::Length(clop.params.len() as u16 + 3),
            Constraint::Length(CHART_HEIGHT),
        ])
        .split(frame.area());

    let middle = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Ratio(1, 2), Constraint::Ratio(1, 2)])
        .split(chunks[1]);

    render_progress(frame, chunks[0], clop, progress, rate);
    render_params(frame, middle[0], clop);
    render_surface(frame, middle[1], clop);
    render_chart(frame, chunks[2], clop);
}

fn render_progress(
    frame: &mut Frame,
    area: Rect,
    clop: &Clop,
    progress: &BatchProgress,
    rate: &mut GameRate,
) {
    let block = Block::default().borders(Borders::ALL).title("Progress");
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
        ])
        .split(inner);
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(33),
            Constraint::Percentage(34),
            Constraint::Percentage(33),
        ])
        .split(rows[0]);

    let batch = progress.get();
    // Once a batch is recorded its games are in `clop.pairs`, so only count a running batch
    let running = if batch.batch == clop.batches_done {
        batch.finished
    } else {
        0
    };
    let games_done = clop.games_played() + running;
    let games_total = clop.total_games();

    rate.update(games_done, games_total);

    frame.render_widget(
        Paragraph::new(tui::format_elapsed(rate.start.elapsed().as_secs()))
            .alignment(Alignment::Left),
        cols[0],
    );
    frame.render_widget(
        Paragraph::new(format!("{:.0} games/h", rate.games_per_hour)).alignment(Alignment::Center),
        cols[1],
    );
    frame.render_widget(
        Paragraph::new(tui::format_eta(rate.eta_secs)).alignment(Alignment::Right),
        cols[2],
    );

    let ratio = (games_done as f64 / games_total as f64).min(1.);
    frame.render_widget(
        Gauge::default()
            .gauge_style(Style::default().fg(Color::Green))
            .ratio(ratio)
            .label(Span::styled(
                format!(
                    "{games_done:>6} / {games_total:>6}  ({:>5.1}%)",
                    ratio * 100.
                ),
                Style::default().fg(Color::White),
            )),
        rows[1],
    );
    render_batch_bar(frame, rows[2], &batch);

    if let Some(e) = &clop.estimate {
        frame.render_widget(
            Paragraph::new(format!("Gain {:+.1} ± {:.1} Elo", e.gain, e.gain_sd))
                .style(Style::default().add_modifier(Modifier::BOLD))
                .alignment(Alignment::Center),
            rows[3],
        );
    }
}

/// Gauge for the running batch, with games in progress shaded between finished and unstarted.
fn render_batch_bar(frame: &mut Frame, area: Rect, batch: &BatchState) {
    let done = batch.finished;
    let ratio = if batch.total > 0 {
        done as f64 / batch.total as f64
    } else {
        0.
    };
    let label: Vec<char> = format!("{done:>6} / {:>6}  ({:>5.1}%)", batch.total, ratio * 100.)
        .chars()
        .collect();

    let width = usize::from(area.width);
    let cells = |games: u64| {
        (games as usize * width)
            .checked_div(batch.total as usize)
            .unwrap_or(0)
    };
    let done_cells = cells(done);
    let started_cells = cells(batch.started).max(done_cells);
    let label_start = width.saturating_sub(label.len()) / 2;

    let spans = (0..width)
        .map(|i| {
            let bg = if i < done_cells {
                Color::Blue
            } else if i < started_cells {
                Color::Cyan
            } else {
                Color::Reset
            };
            let ch = i
                .checked_sub(label_start)
                .and_then(|j| label.get(j))
                .copied()
                .unwrap_or(' ');
            Span::styled(ch.to_string(), Style::default().fg(Color::White).bg(bg))
        })
        .collect::<Vec<_>>();

    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

fn render_box(frame: &mut Frame, area: Rect, title: &str, paragraph: Paragraph) {
    let block = Block::default()
        .borders(Borders::ALL)
        .padding(Padding::horizontal(1))
        .title(title);
    let inner = block.inner(area);
    frame.render_widget(block, area);
    frame.render_widget(paragraph, inner);
}

fn render_params(frame: &mut Frame, area: Rect, clop: &Clop) {
    let width = clop.params.iter().map(|p| p.name.len()).max().unwrap_or(0);
    let mut lines = vec![Line::raw(format!(
        "{:<width$} {:>8} {:>8} {:>8} {:>8} {:>7}",
        "", "Start", "Low", "High", "Best", "±"
    ))];
    // Names are coloured to match their chart lines, so the table doubles as the legend
    for (i, (p, &color)) in clop
        .params
        .iter()
        .zip(PARAM_COLORS.iter().cycle())
        .enumerate()
    {
        let (best, spread) = clop.estimate.as_ref().map_or_else(
            || (String::new(), String::new()),
            |e| {
                (
                    format!("{:.2}", p.value(e.best[i])),
                    format!("{:.2}", e.spread[i]),
                )
            },
        );
        lines.push(Line::from(vec![
            Span::styled(format!("{:<width$}", p.name), Style::default().fg(color)),
            Span::raw(format!(
                " {:>8.2} {:>8.2} {:>8.2} {best:>8} {spread:>7}",
                p.start,
                p.value(-1.),
                p.value(1.),
            )),
        ]));
    }
    render_box(frame, area, "Parameters", Paragraph::new(lines));
}

/// The quadratic terms in Elo at the box edge: the diagonal is each parameter's curvature
/// (negative means it has a peak), off-diagonals are interactions between parameters.
fn render_surface(frame: &mut Frame, area: Rect, clop: &Clop) {
    let marker = |i: usize| {
        Span::styled(
            "■",
            Style::default().fg(PARAM_COLORS[i % PARAM_COLORS.len()]),
        )
    };
    let n = clop.params.len();

    let mut header = vec![Span::raw(" ")];
    for j in 0..n {
        header.extend([Span::raw(" ".repeat(7)), marker(j)]);
    }
    let mut lines = vec![Line::from(header)];

    if let Some(model) = &clop.model {
        for i in 0..n {
            let mut row = vec![marker(i)];
            for j in 0..n {
                let cell = if j < i {
                    String::new()
                } else {
                    format!("{:+.1}", model.quadratic(i, j) * ELO_PER_LOGIT)
                };
                row.push(Span::raw(format!(" {cell:>7}")));
            }
            lines.push(Line::from(row));
        }
    }

    let paragraph = Paragraph::new(lines).alignment(Alignment::Center);
    render_box(frame, area, "Surface", paragraph);
}

fn render_chart(frame: &mut Frame, area: Rect, clop: &Clop) {
    // Plot the change from start as a percentage so parameters share one axis
    let series: Vec<Vec<(f64, f64)>> = clop
        .params
        .iter()
        .map(|p| {
            p.history
                .iter()
                .map(|&(g, value)| (g, change_pct(p, value)))
                .collect()
        })
        .collect();

    // Fixed to the widest box, so estimates jumping between edges early on don't rescale the axis
    let y_min = clop
        .params
        .iter()
        .map(|p| change_pct(p, p.value(-1.)))
        .fold(0., f64::min);
    let y_max = clop
        .params
        .iter()
        .map(|p| change_pct(p, p.value(1.)))
        .fold(0., f64::max);

    let x_max = clop.total_games() as f64;
    let zero = [(0., 0.), (x_max, 0.)];

    // The zero line goes first so parameter lines draw over it, in the axes' default style
    let datasets = iter::once(
        Dataset::default()
            .marker(Marker::Braille)
            .graph_type(GraphType::Line)
            .data(&zero),
    )
    .chain(
        series
            .iter()
            .zip(&clop.params)
            .zip(PARAM_COLORS.iter().cycle())
            .map(|((data, p), &color)| {
                Dataset::default()
                    .name(p.name.clone())
                    .marker(Marker::Braille)
                    .graph_type(GraphType::Line)
                    .style(Style::default().fg(color))
                    .data(data)
            }),
    )
    .collect();

    let chart = Chart::new(datasets)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Best change from start"),
        )
        .x_axis(Axis::default().bounds([0., x_max]))
        .y_axis(Axis::default().bounds([y_min, y_max]).labels(vec![
            Span::raw(format!("{y_min:+.0}%")),
            Span::raw(format!("{y_max:+.0}%")),
        ]));

    frame.render_widget(chart, area);
}

/// Change from the start value as a percentage, or in raw units if the start is zero.
fn change_pct(p: &Param, value: f64) -> f64 {
    if p.start == 0. {
        value
    } else {
        (value - p.start) / p.start.abs() * 100.
    }
}
